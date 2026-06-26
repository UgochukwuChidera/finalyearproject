"""API utility and static routes: dictionaries, audits, templates, settings, previews."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import cv2
from flask import jsonify, render_template, request, send_file, send_from_directory
from werkzeug.utils import safe_join, secure_filename

from ai_extraction.gemini_client import get_active_model
from .bp import bp
from .common import (
    _root_dir, _uploads_dir, _outputs_dir, _dict_dir, _load_models_config, _save_models_config,
    _get_batch_settings, _save_batch_settings, _reset_job_semaphore,
    _safe_config_name, _config_path, _list_configs,
    _init_jobs_internal, _read_audit_entries, _norm_text,
    JOBS, JOBS_LOCK,
)


@bp.route("/api/dictionaries/upload", methods=["POST"])
def api_upload_dictionary() -> Any:
    file = request.files.get("dictionary_file")
    if not file:
        return jsonify({"error": "dictionary_file is required"}), 400

    safe_name = secure_filename(file.filename or "")
    if not safe_name.endswith(".csv"):
        return jsonify({"error": "CSV only"}), 400

    joined = safe_join(str(_dict_dir()), safe_name)
    if not joined:
        return jsonify({"error": "Invalid path"}), 400
    Path(joined).parent.mkdir(parents=True, exist_ok=True)
    file.save(joined)
    return jsonify({"status": "ok", "filename": safe_name})


@bp.route("/audits", methods=["GET"])
def audits() -> str:
    entries = _read_audit_entries(limit=500)
    q = (request.args.get("q") or "").strip()
    sort_key = (request.args.get("sort") or "timestamp").strip()
    sort_dir = (request.args.get("dir") or "desc").strip().lower()

    if q:
        qn = _norm_text(q)
        def _match(a: dict) -> bool:
            pending = ((a.get("review_summary") or {}).get("pending_review_count", 0))
            haystack = [
                a.get("job_id"),
                a.get("form_type"),
                a.get("timestamp"),
                str(pending),
            ]
            return any(qn in _norm_text(v) for v in haystack)
        entries = [a for a in entries if _match(a)]

    key_funcs = {
        "job_id": lambda a: _norm_text(a.get("job_id")),
        "form_type": lambda a: _norm_text(a.get("form_type")),
        "timestamp": lambda a: _norm_text(a.get("timestamp")),
        "pending": lambda a: int(((a.get("review_summary") or {}).get("pending_review_count", 0)) or 0),
    }
    if sort_key not in key_funcs:
        sort_key = "timestamp"
    if sort_dir not in {"asc", "desc"}:
        sort_dir = "desc"
    entries = sorted(entries, key=key_funcs[sort_key], reverse=(sort_dir == "desc"))

    return render_template("audits.html", audits=entries, q=q, sort=sort_key, direction=sort_dir)


@bp.route("/api/template-preview", methods=["GET"])
def api_template_preview() -> Any:
    template_path = (request.args.get("template_path") or "").strip()
    if not template_path:
        return jsonify({"error": "template_path is required"}), 400
    rel = template_path.replace("\\", "/")
    if rel.startswith("templates/"):
        rel = rel[len("templates/") :]
    joined = safe_join(str(_root_dir() / "templates"), rel)
    if not joined:
        return jsonify({"error": "Invalid template path"}), 400
    target = Path(joined)
    if not target.exists():
        return jsonify({"error": "Template not found"}), 404
    image = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return jsonify({"error": "Unable to load template image"}), 422
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return jsonify({"error": "Unable to render template preview"}), 500
    return send_file(io.BytesIO(encoded.tobytes()), mimetype="image/png")


@bp.route("/templates/<path:filename>")
def templates_static(filename: str) -> Any:
    return send_from_directory(_root_dir() / "templates", filename)


@bp.route("/uploads/<path:filename>")
def uploads_static(filename: str) -> Any:
    return send_from_directory(_uploads_dir(), filename)


@bp.route("/outputs/<path:filename>")
def outputs_static(filename: str) -> Any:
    return send_from_directory(_outputs_dir(), filename)


@bp.route("/api/utils/convert-to-png", methods=["POST"])
def api_utils_convert_to_png() -> Any:
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    import numpy as np
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Failed to decode image"}), 422

        ok, encoded = cv2.imencode(".png", img)
        if not ok:
            return jsonify({"error": "Failed to encode to PNG"}), 500

        return send_file(io.BytesIO(encoded.tobytes()), mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/evaluation", methods=["GET"])
def evaluation() -> str:
    results_path = _root_dir() / "evaluation" / "results" / "full_results.json"
    results: dict = {}
    if results_path.exists():
        try:
            with results_path.open("r", encoding="utf-8") as fh:
                results = json.load(fh)
        except (json.JSONDecodeError, OSError):
            results = {}
    return render_template("evaluation.html", results=results)


@bp.route("/settings", methods=["GET"])
def settings() -> str:
    cfg = _load_models_config()
    return render_template("settings.html", models_config=cfg, active_model=get_active_model())


@bp.route("/api/models", methods=["GET"])
def api_models_get() -> Any:
    cfg = _load_models_config()
    cfg["resolved_active"] = get_active_model()
    return jsonify(cfg)


@bp.route("/api/models", methods=["POST"])
def api_models_update() -> Any:
    payload = request.get_json(silent=True) or {}

    active = (payload.get("active_model") or "").strip()
    api_key = (payload.get("api_key") or "").strip()
    models = payload.get("models")

    cfg = _load_models_config()

    if active:
        cfg["active_model"] = active

    if api_key is not None:
        cfg["api_key"] = api_key

    if isinstance(models, list):
        cleaned = []
        for m in models:
            if not isinstance(m, dict):
                continue
            mid = (m.get("id") or "").strip()
            if not mid:
                continue
            cleaned.append({
                "id": mid,
                "label": (m.get("label") or mid).strip(),
                "description": (m.get("description") or "").strip(),
            })
        if cleaned:
            cfg["models"] = cleaned

    _save_models_config(cfg)
    return jsonify({"status": "ok", "active_model": cfg.get("active_model")})


@bp.route("/api/settings/batch", methods=["GET"])
def api_batch_settings_get() -> Any:
    return jsonify(_get_batch_settings())


@bp.route("/api/settings/batch", methods=["POST"])
def api_batch_settings_update() -> Any:
    payload = request.get_json(silent=True) or {}
    updates: dict = {}

    max_concurrent = payload.get("max_concurrent")
    if max_concurrent is not None:
        updates["max_concurrent"] = max(1, min(int(max_concurrent), 20))

    rpm = payload.get("requests_per_minute")
    if rpm is not None:
        updates["requests_per_minute"] = max(1, int(rpm))
        safe_rpm = max(1, updates["requests_per_minute"])
        updates["inter_request_delay"] = round(60.0 / safe_rpm, 2)

    inter_delay = payload.get("inter_request_delay")
    if inter_delay is not None and "inter_request_delay" not in updates:
        updates["inter_request_delay"] = max(0.0, float(inter_delay))

    _save_batch_settings(updates)

    # Reset the semaphore to the new concurrency level
    new_concurrent = _get_batch_settings()["max_concurrent"]
    _reset_job_semaphore(new_concurrent)

    return jsonify({"status": "ok", "batch_settings": _get_batch_settings()})


@bp.route("/help")
def help_center() -> str:
    return render_template("help.html")
