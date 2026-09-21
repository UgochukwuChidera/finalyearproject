"""DAPE pipeline orchestration — form processing, field extraction, export."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import threading as _threading

import cv2
import numpy as np

MIN_DESKEW_ANGLE_DEG = 0.2

logger = logging.getLogger(__name__)

_TEMPLATE_CACHE: dict[str, tuple[float, np.ndarray]] = {}
_TEMPLATE_LOCK = _threading.Lock()
_DICT_CACHE: dict[str, tuple[float, dict[str, list[str]]]] = {}
_DICT_LOCK = _threading.Lock()


def _load_template_cached(template_path: str) -> np.ndarray:
    p = Path(template_path)
    try:
        mtime = p.stat().st_mtime if p.exists() else 0.0
    except OSError:
        mtime = 0.0
    with _TEMPLATE_LOCK:
        entry = _TEMPLATE_CACHE.get(str(p))
        if entry is not None and entry[0] == mtime:
            return entry[1].copy()
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Template image not found: {template_path}")
    with _TEMPLATE_LOCK:
        _TEMPLATE_CACHE[str(p)] = (mtime, img.copy())
    return img


def _load_dictionaries_cached(dictionaries_dir: str) -> dict[str, list[str]]:
    d = Path(dictionaries_dir)
    max_mtime = 0.0
    try:
        if d.exists():
            for csv_path in d.glob("*.csv"):
                try:
                    max_mtime = max(max_mtime, csv_path.stat().st_mtime)
                except OSError:
                    pass
            try:
                max_mtime = max(max_mtime, d.stat().st_mtime)
            except OSError:
                pass
    except OSError:
        pass
    with _DICT_LOCK:
        entry = _DICT_CACHE.get(str(d))
        if entry is not None and entry[0] == max_mtime:
            return entry[1]
    store = DictionaryStore(dictionaries_dir)
    data = store.load()
    with _DICT_LOCK:
        _DICT_CACHE[str(d)] = (max_mtime, data)
    return data


def _scale_bbox(
    bb: dict[str, Any],
    actual_w: int,
    actual_h: int,
    ref_w: float,
    ref_h: float,
    normalized: bool = False,
) -> tuple[int, int, int, int]:
    x = float(bb.get("x", 0))
    y = float(bb.get("y", 0))
    w = float(bb.get("w", 1))
    h = float(bb.get("h", 1))
    if normalized or (abs(ref_w - 1000) < 1e-6 and abs(ref_h - 1000) < 1e-6):
        x = x / 1000.0 * actual_w
        y = y / 1000.0 * actual_h
        w = w / 1000.0 * actual_w
        h = h / 1000.0 * actual_h
    else:
        if max(x, y, w, h) <= 1000 and (x + w > ref_w or y + h > ref_h):
            logger.debug(
                "bbox %s overflows canvas %.0fx%.0f, treating as normalized 0-1000",
                bb,
                ref_w,
                ref_h,
            )
            x = x / 1000.0 * actual_w
            y = y / 1000.0 * actual_h
            w = w / 1000.0 * actual_w
            h = h / 1000.0 * actual_h
        else:
            sx = actual_w / ref_w if ref_w else 1.0
            sy = actual_h / ref_h if ref_h else 1.0
            x *= sx
            y *= sy
            w *= sx
            h *= sy
    return int(x), int(y), int(w), int(h)


from ai_extraction import (
    DictionaryStore,
    GeminiClient,
    best_match,
    build_multi_image_prompt,
    compute_C_dict,
    compute_C_final,
)
from ai_extraction.gemini_client import get_active_model
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


def _load_config(
    config_name: str | None = None, config_path: str | None = None
) -> tuple[dict, str]:
    if config_path:
        p = Path(config_path)
    else:
        if not config_name:
            raise ValueError("Provide config_name or config_path")
        p = Path("configs") / f"{config_name}.json"
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    from utils.config_validator import validate_config

    try:
        data = validate_config(data)
    except ValueError as e:
        raise ValueError(f"Invalid config {p}: {e}") from e
    return data, str(p)


def _load_template_extraction(config: dict) -> dict:
    path = config.get("template_extraction")
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if (
        isinstance(payload, dict)
        and "fields" in payload
        and isinstance(payload["fields"], dict)
    ):
        return payload["fields"]
    return payload if isinstance(payload, dict) else {}


def _to_png_bytes(image: np.ndarray) -> bytes:
    ok, buff = cv2.imencode(".png", image)
    if not ok:
        return b""
    return buff.tobytes()


def _safe_crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    if img is None:
        raise ValueError("safe_crop: image is None")
    if img.size == 0:
        raise ValueError("safe_crop: image is empty")
    ih, iw = img.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(iw, x1 + max(1, int(w)))
    y2 = min(ih, y1 + max(1, int(h)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"safe_crop: invalid crop region x={x} y={y} w={w} h={h} for image {iw}x{ih}"
        )
    return img[y1:y2, x1:x2]


def _deskew_image(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    if gray is None:
        raise ValueError("deskew_image: image is None")
    if gray.size == 0:
        raise ValueError("deskew_image: image is empty")
    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        gray,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def _validation_ok(value: object, field: dict) -> tuple[bool, str]:
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


def _differs_from_template(value: object, baseline: object) -> bool:
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
    progress_callback: Callable[[str, str], None] | None = None,
    api_key: str | None = None,
) -> dict:
    def update_status(stage: str, message: str) -> None:
        if progress_callback:
            progress_callback(stage, message)

    update_status("INITIALIZING", "Loading configuration and resources...")
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

    dictionaries = _load_dictionaries_cached(dictionaries_dir)

    stats: dict[str, Any] = {"dpi": dpi}

    update_status("PREPROCESSING", "Analyzing image metrics and deskewing...")
    image, h, w, aspect = load_image(image_path)
    gray = to_grayscale(image)
    assert (
        gray is not None
    )  # load_image guarantees image, to_grayscale never returns None for valid input
    stats.update({"original_width": w, "original_height": h, "aspect_ratio": aspect})
    stats.update(baseline_metrics(gray))  # type: ignore[arg-type]

    ks = kernel_sizes(dpi)
    skew_meta = skew_analysis(gray, hough_threshold=int(ks["skew_hough_threshold"]))  # type: ignore[arg-type]
    stats.update(skew_meta)
    if (config.get("preprocessing") or {}).get("deskew", False):
        angle = float(skew_meta.get("skew_angle", 0.0))
        if abs(angle) > MIN_DESKEW_ANGLE_DEG:
            gray = _deskew_image(gray, -angle)  # type: ignore[arg-type]
            stats["deskew_applied"] = True
            stats["deskew_angle_used"] = -angle
        else:
            stats["deskew_applied"] = False
    else:
        stats["deskew_applied"] = False
    normalized, illum = illumination_normalization(
        gray,  # type: ignore[arg-type]
        stats.get("grayscale_std", 0.0),
        blur_size=int(ks["illumination_blur"]),  # type: ignore[arg-type]
        clahe_tile=int(ks["clahe_tile"]),  # type: ignore[arg-type]
    )
    stats.update(illum)
    binary, bin_stats = binarization(normalized, block_size=int(ks["binarization_block"]))  # type: ignore[arg-type]
    stats.update(bin_stats)
    cropped, crop_stats = border_removal(binary, stats.get("threshold_stability", 0.0))  # type: ignore[arg-type]
    stats.update(crop_stats)

    update_status("ALIGNMENT", "Aligning form with template blueprint...")
    template_path = config.get("template_path")
    if not template_path:
        raise ValueError("template_path is required")
    template_img = _load_template_cached(str(template_path))

    stats.update(
        structure_prep(cropped, (template_img.shape[1], template_img.shape[0]))  # type: ignore[arg-type]
    )
    stats["fusion_score"] = fuse(stats)

    aligner = TemplateAligner(max_features=int(ks["max_features"]))  # type: ignore[arg-type]
    aligned, _, align_meta = aligner.align(gray, template_img)  # type: ignore[arg-type]
    stats.update({f"align_{k}": v for k, v in align_meta.items()})

    update_status("ANALYSIS", "Detecting ink signatures and form differences...")
    differ = DifferentialAnalyzer(
        diff_threshold=int(ks["diff_threshold"]),  # type: ignore[arg-type]
        min_region_area=int(ks["min_region_area"]),  # type: ignore[arg-type]
        morph_open=int(ks["morph_open"]),  # type: ignore[arg-type]
        morph_close=int(ks["morph_close"]),  # type: ignore[arg-type]
        morph_dilate=int(ks["morph_dilate"]),  # type: ignore[arg-type]
    )
    mask, diff_meta = differ.analyze(aligned, template_img)
    stats.update(
        {f"diff_{k}": v for k, v in diff_meta.items() if not hasattr(v, "shape")}
    )

    editor_canvas = config.get("editor_canvas", {}) or {}
    ref_w = float(editor_canvas.get("width", 800.0))
    ref_h = float(editor_canvas.get("height", 1100.0))
    ref_w = ref_w if ref_w > 0 else 800.0
    ref_h = ref_h if ref_h > 0 else 1100.0
    actual_h, actual_w = aligned.shape[:2]
    is_normalized = bool(
        config.get("bbox_normalized") or config.get("coordinate_space") == "normalized"
    )

    critical_crops: dict[str, bytes] = {}
    for field in fields:
        if not field.get("critical"):
            continue
        bb = field.get("bounding_box", {})
        x, y, w0, h0 = _scale_bbox(
            bb, actual_w, actual_h, ref_w, ref_h, normalized=is_normalized
        )

        roi = _safe_crop(aligned, x, y, w0, h0)
        roi_mask = _safe_crop(mask, x, y, w0, h0)
        ink = np.full_like(roi, 255)
        ink[roi_mask > 0] = roi[roi_mask > 0]
        critical_crops[field["name"]] = _to_png_bytes(ink)
        cv2.imwrite(str(crops_dir / f"{field['name']}.png"), ink)

    update_status(
        "EXTRACTION", f"Running Neural Extraction with {get_active_model()}..."
    )
    full_image_bytes = _to_png_bytes(aligned)
    prompt_items = build_multi_image_prompt(config, full_image_bytes, critical_crops)
    images = [p["image"] for p in prompt_items]
    prompts = [p["prompt"] for p in prompt_items]

    client = GeminiClient(api_key=api_key)
    ai_payload = client.extract_from_images(images=images, prompts=prompts)
    api_error = ai_payload.get("error") if isinstance(ai_payload, dict) else None
    if api_error:
        raise RuntimeError(f"AI extraction failed: {api_error}")
    ai_meta = (ai_payload.get("meta") or {}) if isinstance(ai_payload, dict) else {}
    ai_fields = ai_payload.get("fields", {}) if isinstance(ai_payload, dict) else {}
    if not isinstance(ai_fields, dict):
        ai_fields = {}
    ai_conf_map = ai_meta.get("C_lp") or {}
    has_logprobs = ai_meta.get("has_logprobs", False)

    ai_debug = {
        "model": client.model,
        "has_logprobs": has_logprobs,
        "fields_extracted": sorted(ai_fields.keys()),
        "raw_response_preview": ai_meta.get("raw_response_preview", ""),
        "logprobs_debug": ai_meta.get("logprobs_debug", ""),
        "overall_confidence": ai_meta.get("overall_confidence"),
        "api_error": None,
    }
    if not ai_fields:
        logger.warning(
            "AI extraction produced no fields for job %s. "
            "Check ai_debug.raw_response_preview for the model output.",
            jid,
        )

    update_status("VALIDATION", "Applying business logic and scoring results...")
    final_fields: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    extraction_entries: list[dict[str, Any]] = []
    text_diff_changed_count = 0

    for field in fields:
        name = field["name"]
        bb = field.get("bounding_box", {})
        x, y, w0, h0 = _scale_bbox(
            bb, actual_w, actual_h, ref_w, ref_h, normalized=is_normalized
        )

        raw = ai_fields.get(name)
        if name in ai_conf_map:
            c_lp = float(ai_conf_map[name])
        elif has_logprobs:
            c_lp = 0.7
            logger.warning(
                "Field '%s' had no C_lp computed despite logprobs being available (possible mapping issue)",
                name,
            )
        else:
            c_lp = 0.65
            logger.debug(
                "Field '%s' using default confidence since logprobs not available", name
            )

        if not field.get("critical"):
            baseline = template_extraction.get(name)
            if _is_checkbox(field):
                raw = bool(raw) if raw is not None else None
            else:
                raw = raw if _differs_from_template(raw, baseline) else None
            text_diff_changed = _differs_from_template(raw, baseline)
        else:
            text_diff_changed = False

        dict_file = field.get("dictionary")
        dict_entries = dictionaries.get(dict_file, []) if dict_file else []
        matched, distance = best_match(str(raw or ""), dict_entries)
        c_dict = compute_C_dict(str(raw or ""), matched, distance)
        c_final = compute_C_final(
            c_lp,
            c_dict if field.get("critical") else 0.0,
            w_lp,
            w_dict if field.get("critical") else 0.0,
        )

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

        out: dict[str, Any] = {
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
            "text_diff_changed": bool(text_diff_changed),
            "crop_path": str(crop_path),
            "critical": bool(field.get("critical")),
        }
        final_fields.append(out)
        if out["text_diff_changed"]:
            text_diff_changed_count += 1

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
    stats["text_diff_changed_count"] = text_diff_changed_count
    stats["text_diff_unchanged_count"] = max(
        0, len(final_fields) - text_diff_changed_count
    )
    structured = structurer.structure(final_fields, jid, form_type, stats)

    update_status("EXPORTING", "Generating structured reports and audit logs...")
    exporter = DataExporter()
    xlsx = RelationalXLSXExporter()
    base = str(Path(output_dir) / jid)
    export_paths = exporter.export_all(structured, base)
    export_paths["xlsx"] = xlsx.export(structured, base + ".xlsx")

    audit_logger = AuditLogger(
        log_dir=log_dir, audit_jsonl_path=str(Path(output_dir) / "audit.jsonl")
    )
    audit_path = audit_logger.log(
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
                "accepted_count": sum(
                    1 for f in final_fields if f["validation_status"] == "accepted"
                ),
                "spot_check_count": sum(
                    1 for f in final_fields if f["validation_status"] == "spot_check"
                ),
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
        "ai_debug": ai_debug,
    }
