"""
DAPE Pipeline — Entry Point and unified process_form() API.
"""

import argparse
import glob
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ai_extraction import DictionaryStore, OpenRouterGeminiClient, closest_dictionary_match
from ai_extraction.confidence import compute_confidence
from project.alignment.aligner import TemplateAligner
from project.differential.analyzer import DifferentialAnalyzer
from project.extraction.field_extractor import FieldExtractor
from project.orchestrator import DAPEOrchestrator
from project.output.audit_logger import AuditLogger
from project.output.exporter import DataExporter
from project.output.relational_exporter import RelationalXLSXExporter
from project.output.structurer import OutputStructurer
from project.preprocessing.baseline_metrics import baseline_metrics
from project.preprocessing.binarization import binarization
from project.preprocessing.border_removal import border_removal
from project.preprocessing.dpi import kernel_sizes
from project.preprocessing.fusion import fuse
from project.preprocessing.grayscale import to_grayscale
from project.preprocessing.illumination import illumination_normalization
from project.preprocessing.io import load_image
from project.preprocessing.skew_analysis import skew_analysis
from project.preprocessing.structure_prep import structure_prep
from project.template_analyzer.registry_builder import TemplateRegistryBuilder


def _load_config(config_name: str | None = None, config_path: str | None = None) -> tuple[dict, str]:
    if config_path:
        p = Path(config_path)
    else:
        if not config_name:
            raise ValueError("Provide config_name or config_path")
        p = Path("configs") / f"{config_name}.json"
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh), str(p)


def _to_png_bytes(gray_image: np.ndarray) -> bytes:
    ok, buff = cv2.imencode(".png", gray_image)
    if not ok:
        return b""
    return buff.tobytes()


def _safe_crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    if img is None or img.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    ih, iw = img.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(iw, x1 + max(1, int(w)))
    y2 = min(ih, y1 + max(1, int(h)))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1), dtype=np.uint8)
    return img[y1:y2, x1:x2]


def _field_type(field: dict) -> str:
    t = (field.get("field_type") or "").strip().lower()
    if t in {"checkbox", "handwritten", "printed"}:
        return t
    exp = (field.get("expected_type") or "string").lower()
    return "checkbox" if exp == "boolean" else "handwritten"


def _validation_ok(value, field: dict) -> tuple[bool, str]:
    v = field.get("validation", {}) or {}
    s = "" if value is None else str(value)
    min_len = v.get("min_length")
    regex = v.get("regex")
    if min_len is not None and len(s.strip()) < int(min_len):
        return False, f"min_length<{min_len}"
    if regex:
        import re

        if not re.match(regex, s.strip()):
            return False, "regex_mismatch"
    return True, "passed"


def process_form(
    image_path: str,
    config_name: str | None = None,
    config_path: str | None = None,
    output_dir: str = "outputs",
    log_dir: str = "logs",
    dictionaries_dir: str = "dictionaries",
    dpi: int = 300,
    tesseract_cmd: str | None = None,
    original_filename: str | None = None,
    job_id: str | None = None,
) -> dict:
    config, resolved_config_path = _load_config(config_name, config_path)
    mode = (config.get("extraction_mode") or "differential").strip().lower()
    if mode not in {"differential", "ai", "hybrid"}:
        mode = "differential"

    fields = config.get("fields", [])
    form_type = config.get("form_type", Path(resolved_config_path).stem)
    weights = config.get("confidence_weights", {}) or {}
    thresholds = config.get("thresholds", {}) or {}
    w_lp = float(weights.get("w_lp", 0.6))
    w_dict = float(weights.get("w_dict", 0.4))
    t_auto = float(thresholds.get("auto_accept", 0.85))
    t_review = float(thresholds.get("review", 0.70))
    hybrid_switch_conf = float(thresholds.get("hybrid_switch_confidence", 0.80))
    dict_low_lp_guard = float(thresholds.get("dictionary_low_lp_guard", 0.80))
    min_field_height_px = int(thresholds.get("min_field_height_px", 20))

    jid = job_id or str(uuid.uuid4())
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    crops_dir = output_path / "crops" / jid
    crops_dir.mkdir(parents=True, exist_ok=True)

    store = DictionaryStore(dictionaries_dir)
    dictionaries = store.load()

    stats: dict = {"dpi": dpi}
    images: dict = {}

    image, h, w, aspect = load_image(image_path)
    gray = to_grayscale(image)
    stats.update({"original_width": w, "original_height": h, "aspect_ratio": aspect})
    stats.update(baseline_metrics(gray))

    ks = kernel_sizes(dpi)
    stats.update(skew_analysis(gray, hough_threshold=ks["skew_hough_threshold"]))
    normalized, illum = illumination_normalization(
        gray,
        stats.get("grayscale_std", 0.0),
        blur_size=ks["illumination_blur"],
        clahe_tile=ks["clahe_tile"],
    )
    stats.update(illum)
    binary, bin_stats = binarization(normalized, block_size=ks["binarization_block"])
    stats.update(bin_stats)
    cropped, crop_stats = border_removal(binary, stats.get("threshold_stability", 0.0))
    stats.update(crop_stats)

    template_img = None
    aligned = gray
    mask = np.zeros_like(gray, dtype=np.uint8)

    if mode in {"differential", "hybrid"}:
        template_path = config.get("template_path")
        if not template_path:
            raise ValueError("template_path is required for differential/hybrid mode")
        template_img = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if template_img is None:
            raise FileNotFoundError(f"Template image not found: {template_path}")

        stats.update(structure_prep(cropped, (template_img.shape[1], template_img.shape[0])))
        stats["fusion_score"] = fuse(stats)

        aligner = TemplateAligner(max_features=ks["max_features"])
        aligned, _, align_meta = aligner.align(gray, template_img)
        stats.update({f"align_{k}": v for k, v in align_meta.items()})

        differ = DifferentialAnalyzer(
            diff_threshold=ks["diff_threshold"],
            min_region_area=ks["min_region_area"],
        )
        mask, diff_meta = differ.analyze(aligned, template_img)
        stats.update({f"diff_{k}": v for k, v in diff_meta.items() if not hasattr(v, "shape")})

    images["gray"] = gray
    images["aligned"] = aligned
    images["interaction_mask"] = mask

    diff_defs = [
        {
            "id": f["name"],
            "type": _field_type(f),
            "x": int((f.get("bounding_box") or {}).get("x", 0)),
            "y": int((f.get("bounding_box") or {}).get("y", 0)),
            "w": int((f.get("bounding_box") or {}).get("w", 1)),
            "h": int((f.get("bounding_box") or {}).get("h", 1)),
            "required": bool(f.get("critical", False)),
        }
        for f in fields
    ]

    diff_map: dict[str, dict] = {}
    if mode in {"differential", "hybrid"}:
        extractor = FieldExtractor(tesseract_cmd=tesseract_cmd)
        for row in extractor.extract_fields(aligned, mask, diff_defs):
            diff_map[row["field_id"]] = row

    ai_map: dict[str, dict] = {}
    if mode in {"ai", "hybrid"}:
        client = OpenRouterGeminiClient()
        if mode == "ai":
            png = _to_png_bytes(normalized)
            out = client.extract_fields_from_image(png, form_type, fields)
            ai_map = {k: {"value": v.value, "C_lp": v.c_lp} for k, v in out.items()}
        else:
            handwriting_fields = [f for f in fields if _field_type(f) != "checkbox"]
            for f in handwriting_fields:
                bb = f.get("bounding_box", {})
                x, y, w0, h0 = int(bb.get("x", 0)), int(bb.get("y", 0)), int(bb.get("w", 1)), int(bb.get("h", 1))
                roi = _safe_crop(aligned, x, y, w0, h0)
                roi_mask = _safe_crop(mask, x, y, w0, h0)
                crop = cv2.bitwise_and(roi, roi, mask=roi_mask)
                png = _to_png_bytes(crop)
                ai = client.extract_single_field_from_crop(png, f, form_type)
                ai_map[f["name"]] = {"value": ai.value, "C_lp": ai.c_lp}

    final_fields = []
    pending = []
    extraction_entries = []

    for f in fields:
        name = f["name"]
        bb = f.get("bounding_box", {})
        x, y, w0, h0 = int(bb.get("x", 0)), int(bb.get("y", 0)), int(bb.get("w", 1)), int(bb.get("h", 1))
        ftype = _field_type(f)

        diff_row = diff_map.get(name, {})
        ai_row = ai_map.get(name, {})

        if mode == "differential":
            raw = diff_row.get("value", "")
            c_lp = float(diff_row.get("confidence", 0.0))
            source = "differential"
        elif mode == "ai":
            raw = ai_row.get("value", "")
            c_lp = float(ai_row.get("C_lp", 0.5))
            source = "ai"
        else:
            if ftype == "checkbox":
                raw = diff_row.get("value", False)
                c_lp = float(diff_row.get("confidence", 0.0))
                source = "differential"
            else:
                d_conf = float(diff_row.get("confidence", 0.0))
                d_val = diff_row.get("value", "")
                a_val = ai_row.get("value", "")
                a_conf = float(ai_row.get("C_lp", 0.5))
                use_ai = (not str(d_val).strip()) or d_conf < hybrid_switch_conf
                raw = a_val if use_ai else d_val
                c_lp = a_conf if use_ai else d_conf
                source = "ai" if use_ai else "differential"

        dict_file = f.get("dictionary")
        dict_entries = dictionaries.get(dict_file, []) if dict_file else []
        best_match, min_distance = closest_dictionary_match(str(raw), dict_entries)
        c_final, c_dict, dictionary_distance = compute_confidence(
            str(raw), c_lp, best_match, bool(f.get("critical") and dict_file), w_lp, w_dict
        )
        min_distance = min(min_distance, dictionary_distance)

        if c_final >= t_auto:
            status = "accepted"
        elif c_final >= t_review:
            status = "accepted_spot_check"
        else:
            status = "pending_review"

        if h0 < min_field_height_px:
            status = "pending_review"
            sem_reason = "bounding_box_too_small"
        if dict_file and min_distance > 2 and c_lp < dict_low_lp_guard:
            status = "pending_review"

        sem_ok, sem_reason = _validation_ok(raw, f)
        if not sem_ok:
            status = "pending_review"

        needs_review = status == "pending_review"
        crop = _safe_crop(aligned if aligned is not None else gray, x, y, w0, h0)
        crop_name = f"{name}.png"
        crop_path = crops_dir / crop_name
        cv2.imwrite(str(crop_path), crop)

        out = {
            "field_id": name,
            "field_type": ftype,
            "x": x,
            "y": y,
            "w": w0,
            "h": h0,
            "value": raw,
            "final_value": raw,
            "confidence": float(c_final),
            "C_lp": float(c_lp),
            "C_dict": float(c_dict),
            "closest_dictionary_match": best_match,
            "min_distance": int(min_distance),
            "validation_status": status,
            "validation_reason": sem_reason,
            "needs_review": needs_review,
            "corrected": False,
            "source": source,
            "crop_path": str(crop_path),
        }
        final_fields.append(out)

        extraction_entries.append(
            {
                "field_name": name,
                "raw_value": raw,
                "C_lp": float(c_lp),
                "C_dict": float(c_dict),
                "C_final": float(c_final),
                "status": status,
                "final_value": raw,
                "reviewer": None,
                "correction": None,
            }
        )
        if needs_review:
            pending.append(out)

    schema = {f["name"]: f["name"] for f in fields}
    structurer = OutputStructurer(schema)
    structured = structurer.structure(final_fields, jid, form_type, stats)

    exporter = DataExporter()
    xlsx = RelationalXLSXExporter()
    base = str(Path(output_dir) / jid)
    export_paths = exporter.export_all(structured, base)
    export_paths["xlsx"] = xlsx.export(structured, base + ".xlsx")

    logger = AuditLogger(log_dir=log_dir, audit_jsonl_path=str(Path(output_dir) / "audit.jsonl"))
    audit_path = logger.log(
        form_id=jid,
        template_id=form_type,
        processing_stats=stats,
        validated_fields=final_fields,
        export_paths=export_paths,
        original_filename=original_filename or Path(image_path).name,
        extra={
            "job_id": jid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "form_type": form_type,
            "original_filename": original_filename or Path(image_path).name,
            "extractions": extraction_entries,
            "review_summary": {
                "pending_review_count": len(pending),
                "accepted_count": sum(1 for f in final_fields if f["validation_status"] == "accepted"),
            },
        },
    )

    return {
        "job_id": jid,
        "status": "pending_review" if pending else "completed",
        "structured_output": structured,
        "fields": final_fields,
        "pending_fields": pending,
        "export_paths": export_paths,
        "audit_log_path": audit_path,
        "config_path": resolved_config_path,
    }


def parse_args():
    p = argparse.ArgumentParser(description="DAPE Form Processing Pipeline")

    # Mode
    p.add_argument("--register", action="store_true", help="Register a blank template before processing")
    p.add_argument("--batch", action="store_true", help="Process all forms in --forms-dir")

    # Input
    p.add_argument("--image", default=None, help="Path to scanned form image (single-form mode)")
    p.add_argument("--forms-dir", default="form", help="Directory of forms for batch mode (default: form/)")
    p.add_argument("--template-id", default="student_academic_record", help="Template ID to use (must exist in registry.json)")
    p.add_argument("--template-image", default=None, help="Blank template image path (--register mode only)")

    # Pipeline settings
    p.add_argument("--dpi", type=int, default=300, help="Actual scan DPI — scales all kernels (default: 300)")
    p.add_argument("--threshold", type=float, default=0.60, help="Confidence threshold for HITL escalation (default: 0.60)")
    p.add_argument("--no-hitl", action="store_true", help="Disable HITL review (auto-accept all extractions)")
    p.add_argument("--hitl-port", type=int, default=5050, help="Port for HITL Flask review UI (default: 5050)")

    # Paths
    p.add_argument("--registry", default="templates/registry.json")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--tesseract-cmd", default=None, help="Explicit path to Tesseract binary")

    return p.parse_args()


def main():
    args = parse_args()

    if args.register:
        if not args.template_image:
            print("[ERROR] --template-image required in --register mode")
            return
        builder = TemplateRegistryBuilder()
        builder.build(
            template_id=args.template_id,
            image_path=args.template_image,
            registry_path=args.registry,
        )
        print(f"\nTemplate '{args.template_id}' registered.")
        print("Review templates/registry.json to adjust field coordinates if needed.")
        print(f"Then run:  python main.py --image <form.tif> --template-id {args.template_id} --dpi {args.dpi}\n")
        return

    orchestrator = DAPEOrchestrator(
        registry_path=args.registry,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        confidence_threshold=args.threshold,
        enable_hitl=not args.no_hitl,
        hitl_host="127.0.0.1",
        hitl_port=args.hitl_port,
        tesseract_cmd=args.tesseract_cmd,
        dpi=args.dpi,
    )

    if args.batch:
        exts = ["tif", "tiff", "jpg", "jpeg", "png"]
        paths = []
        for ext in exts:
            paths += glob.glob(f"{args.forms_dir}/*.{ext}")
            paths += glob.glob(f"{args.forms_dir}/*.{ext.upper()}")
        paths = sorted(set(paths))

        if not paths:
            print(f"[ERROR] No images found in '{args.forms_dir}'")
            return

        print(f"\nBatch processing {len(paths)} forms (template: {args.template_id}, DPI: {args.dpi})\n")
        results = orchestrator.process_batch(paths, args.template_id)

        print(f"\n{'='*55}")
        print("  BATCH COMPLETE")
        print(f"{'='*55}")
        ok = [r for r in results if "error" not in r]
        err = [r for r in results if "error" in r]
        print(f"  Processed : {len(ok)}")
        print(f"  Errors    : {len(err)}")
        for r in err:
            print(f"  ERROR {r.get('form_id','?')}: {r['error']}")
        return

    if not args.image:
        print("[ERROR] Provide --image <path> or use --batch / --register")
        return

    print(f"\nProcessing : {args.image}")
    print(f"Template   : {args.template_id}")
    print(f"DPI        : {args.dpi}")
    print(f"HITL       : {'disabled' if args.no_hitl else f'enabled (port {args.hitl_port})'}\n")

    result = orchestrator.process(image_path=args.image, template_id=args.template_id)

    print(f"\n{'='*55}")
    print("  RESULT")
    print(f"{'='*55}")
    data = result["structured_output"]["data"]
    for k, v in data.items():
        print(f"  {k:<28} {v}")
    print(f"\n  Exports   : {result['export_paths']}")
    print(f"  Audit log : {result['audit_log_path']}")
    esc = result["stats"]
    print(f"  Flagged   : {esc.get('esc_flagged_count',0)}")
    print(f"  Corrected : {esc.get('esc_corrected_count',0)}")
    print()


if __name__ == "__main__":
    main()
