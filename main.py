"""DAPE pipeline entrypoint."""

import argparse
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ai_extraction import (
    DictionaryStore,
    GeminiClient,
    best_match,
    build_multi_image_prompt,
    compute_C_dict,
    compute_C_final,
)
from project.alignment.aligner import TemplateAligner
from project.differential.analyzer import DifferentialAnalyzer
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


def _load_template_extraction(config: dict) -> dict:
    path = config.get("template_extraction")
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict) and "fields" in payload and isinstance(payload["fields"], dict):
        return payload["fields"]
    return payload if isinstance(payload, dict) else {}


def _to_png_bytes(image: np.ndarray) -> bytes:
    ok, buff = cv2.imencode(".png", image)
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


def _validation_ok(value, field: dict) -> tuple[bool, str]:
    rules = field.get("validation", {}) or {}
    s = "" if value is None else str(value)

    min_len = rules.get("min_length")
    if min_len is not None and len(s.strip()) < int(min_len):
        return False, f"min_length<{min_len}"

    regex = rules.get("regex")
    if regex and not re.match(regex, s.strip()):
        return False, "regex_mismatch"

    try:
        if rules.get("min") is not None or rules.get("max") is not None:
            n = float(s)
            if rules.get("min") is not None and n < float(rules["min"]):
                return False, f"value_below_min_{rules['min']}"
            if rules.get("max") is not None and n > float(rules["max"]):
                return False, f"value_above_max_{rules['max']}"
    except Exception:
        if rules.get("min") is not None or rules.get("max") is not None:
            return False, "numeric_parse_failed"

    return True, "passed"


def _is_checkbox(field: dict) -> bool:
    t = (field.get("expected_type") or "").lower()
    return t in {"checkbox_group", "checkbox", "boolean"}


def _differs_from_template(value, baseline) -> bool:
    if value is None:
        return False
    return str(value).strip() != str(baseline if baseline is not None else "").strip()


def process_form(
    image_path: str,
    config_name: str | None = None,
    config_path: str | None = None,
    output_dir: str = "outputs",
    log_dir: str = "logs",
    dictionaries_dir: str = "dictionaries",
    dpi: int = 300,
    original_filename: str | None = None,
    job_id: str | None = None,
) -> dict:
    config, resolved_config_path = _load_config(config_name, config_path)
    fields = config.get("fields", [])
    form_type = config.get("form_type", Path(resolved_config_path).stem)
    template_extraction = _load_template_extraction(config)

    weights = config.get("confidence_weights", {}) or {}
    thresholds = config.get("thresholds", {}) or {}
    w_lp = float(weights.get("w_lp", 0.6))
    w_dict = float(weights.get("w_dict", 0.4))
    t_auto = float(thresholds.get("auto_accept", 0.85))
    t_review = float(thresholds.get("review", 0.70))

    jid = job_id or str(uuid.uuid4())
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    crops_dir = output_path / "crops" / jid
    crops_dir.mkdir(parents=True, exist_ok=True)

    store = DictionaryStore(dictionaries_dir)
    dictionaries = store.load()

    stats: dict = {"dpi": dpi}

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

    template_path = config.get("template_path")
    if not template_path:
        raise ValueError("template_path is required")
    template_img = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template_img is None:
        raise FileNotFoundError(f"Template image not found: {template_path}")

    stats.update(structure_prep(cropped, (template_img.shape[1], template_img.shape[0])))
    stats["fusion_score"] = fuse(stats)

    aligner = TemplateAligner(max_features=ks["max_features"])
    aligned, _, align_meta = aligner.align(gray, template_img)
    stats.update({f"align_{k}": v for k, v in align_meta.items()})

    differ = DifferentialAnalyzer(diff_threshold=ks["diff_threshold"], min_region_area=ks["min_region_area"])
    mask, diff_meta = differ.analyze(aligned, template_img)
    stats.update({f"diff_{k}": v for k, v in diff_meta.items() if not hasattr(v, "shape")})

    # NEW: Define editor-to-actual scaling multipliers
    # We assume the UI editor works on an 800x1100 fixed canvas (Portrait A4)
    ref_w, ref_h = 800.0, 1100.0
    actual_h, actual_w = aligned.shape[:2]
    scale_x, scale_y = actual_w / ref_w, actual_h / ref_h

    critical_crops: dict[str, bytes] = {}
    for field in fields:
        if not field.get("critical"):
            continue
        bb = field.get("bounding_box", {})
        # Scale coordinates from 900x600 -> actual resolution
        x = int(bb.get("x", 0) * scale_x)
        y = int(bb.get("y", 0) * scale_y)
        w0 = int(bb.get("w", 1) * scale_x)
        h0 = int(bb.get("h", 1) * scale_y)
        
        roi = _safe_crop(aligned, x, y, w0, h0)
        roi_mask = _safe_crop(mask, x, y, w0, h0)
        ink = np.full_like(roi, 255)
        ink[roi_mask > 0] = roi[roi_mask > 0]
        critical_crops[field["name"]] = _to_png_bytes(ink)
        cv2.imwrite(str(crops_dir / f"{field['name']}.png"), ink)

    full_image_bytes = _to_png_bytes(aligned)
    prompt_items = build_multi_image_prompt(config, full_image_bytes, critical_crops)
    images = [p["image"] for p in prompt_items]
    prompts = [p["prompt"] for p in prompt_items]

    ai_payload = GeminiClient().extract_from_images(images=images, prompts=prompts)
    ai_fields = ai_payload.get("fields", {}) if isinstance(ai_payload, dict) else {}
    ai_conf_map = ((ai_payload.get("meta") or {}).get("C_lp") or {}) if isinstance(ai_payload, dict) else {}

    final_fields = []
    pending = []
    extraction_entries = []

    for field in fields:
        name = field["name"]
        bb = field.get("bounding_box", {})
        x = int(bb.get("x", 0) * scale_x)
        y = int(bb.get("y", 0) * scale_y)
        w0 = int(bb.get("w", 1) * scale_x)
        h0 = int(bb.get("h", 1) * scale_y)

        raw = ai_fields.get(name)
        c_lp = float(ai_conf_map.get(name, 0.5))

        if not field.get("critical"):
            baseline = template_extraction.get(name)
            if _is_checkbox(field):
                raw = bool(raw) if raw is not None else None
            else:
                raw = raw if _differs_from_template(raw, baseline) else None

        dict_file = field.get("dictionary")
        dict_entries = dictionaries.get(dict_file, []) if dict_file else []
        matched, distance = best_match(str(raw or ""), dict_entries)
        c_dict = compute_C_dict(str(raw or ""), matched, distance)
        c_final = compute_C_final(c_lp, c_dict if field.get("critical") else 0.0, w_lp, w_dict if field.get("critical") else 0.0)

        if c_final >= t_auto:
            status = "accepted"
        elif c_final >= t_review:
            status = "spot_check"
        else:
            status = "pending_review"

        sem_ok, sem_reason = _validation_ok(raw, field)
        if not sem_ok:
            status = "pending_review"

        crop = _safe_crop(aligned, x, y, w0, h0)
        crop_path = crops_dir / f"{name}.png"
        if not crop_path.exists():
            cv2.imwrite(str(crop_path), crop)

        out = {
            "field_id": name,
            "field_type": "checkbox" if _is_checkbox(field) else "string",
            "x": x,
            "y": y,
            "w": w0,
            "h": h0,
            "value": raw,
            "final_value": raw,
            "confidence": float(c_final),
            "C_lp": float(c_lp),
            "C_dict": float(c_dict),
            "C_final": float(c_final),
            "closest_dictionary_match": matched,
            "min_distance": int(distance),
            "validation_status": status,
            "validation_reason": sem_reason,
            "needs_review": status == "pending_review",
            "corrected": False,
            "source": "hybrid_ai",
            "crop_path": str(crop_path),
            "critical": bool(field.get("critical")),
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
        if out["needs_review"]:
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
                "spot_check_count": sum(1 for f in final_fields if f["validation_status"] == "spot_check"),
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
    p = argparse.ArgumentParser(description="DAPE form processing")
    p.add_argument("--image", required=True, help="Path to filled form image")
    p.add_argument("--config-name", default=None, help="Config name in configs/<name>.json")
    p.add_argument("--config-path", default=None, help="Absolute/relative config path")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--dictionaries-dir", default="dictionaries")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    result = process_form(
        image_path=args.image,
        config_name=args.config_name,
        config_path=args.config_path,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        dictionaries_dir=args.dictionaries_dir,
        dpi=args.dpi,
    )
    print(json.dumps({"job_id": result["job_id"], "status": result["status"], "audit": result["audit_log_path"]}, indent=2))


if __name__ == "__main__":
    main()
