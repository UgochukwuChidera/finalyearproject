"""Config-related routes: list, create, edit, discover, delete, and upload."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from flask import jsonify, redirect, render_template, request, url_for

from ai_extraction.gemini_client import GeminiClient
from ai_extraction.prompt_builder import build_discovery_prompt
from .bp import bp
from .common import (
    _root_dir, _cfg_dir, _safe_config_name, _config_path, _load_config,
    _list_configs, _save_config, _get_api_key, _allowed_ext,
)
from .job_routes import _queue_job


@bp.route("/configs", methods=["GET"])
def configs_page() -> str:
    return render_template("configs.html", configs=_list_configs())


@bp.route("/configs/new", methods=["GET", "POST"])
def config_new() -> Any:
    if request.method == "GET":
        return render_template("config_editor.html", mode="new", config_name="", config_text="{}", config={})

    name = _safe_config_name(request.form.get("config_name", ""))
    payload = json.loads(request.form.get("config_json", "{}") or "{}")
    _save_config(name, payload)
    return redirect(url_for("web.config_edit", name=name))


@bp.route("/configs/<name>/edit", methods=["GET", "POST"])
def config_edit(name: str) -> Any:
    safe_name = _safe_config_name(name)
    cfg = _load_config(safe_name)

    if request.method == "POST":
        payload = json.loads(request.form.get("config_json", "{}") or "{}")
        _save_config(safe_name, payload)
        cfg = payload

    return render_template(
        "config_editor.html",
        mode="edit",
        config_name=safe_name,
        config=cfg,
        config_text=json.dumps(cfg, indent=2),
        template_path=cfg.get("template_path", ""),
    )


@bp.route("/api/config/discover", methods=["POST"])
def config_discover() -> dict[str, Any] | tuple[dict[str, str], int]:
    file = request.files.get("template_file")
    if not file:
        return jsonify({"error": "template_file is required"}), 400

    templates_dir = _root_dir() / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    template_filename = f"template_{uuid.uuid4()}.png"
    template_path = templates_dir / template_filename
    file.save(template_path)

    try:
        with open(template_path, "rb") as f:
            image_bytes = f.read()

        client = GeminiClient(api_key=_get_api_key())
        prompt = build_discovery_prompt()
        result = client.extract_from_images([image_bytes], [prompt])

        if isinstance(result, dict):
            # Use forward slashes for cross-platform compatibility
            result["template_path"] = f"templates/{template_filename}"

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/configs/<name>/delete", methods=["POST"])
def config_delete(name: str) -> str:
    safe_name = _safe_config_name(name)
    path = _config_path(safe_name)
    if path.exists():
        path.unlink()
    return redirect(url_for("web.configs_page"))


@bp.route("/upload", methods=["POST"])
def upload() -> str | tuple[dict[str, Any], int]:
    cfg = request.form.get("config_name", "").strip()
    files = request.files.getlist("form_files")
    if not files:
        single = request.files.get("form_file")
        files = [single] if single else []

    files = [f for f in files if f and f.filename]
    if not cfg or not files:
        return jsonify({"error": "config_name and at least one form file are required"}), 400
    try:
        cfg = _safe_config_name(cfg)
    except ValueError:
        return jsonify({"error": "Invalid config name"}), 400

    invalid = [f.filename for f in files if not _allowed_ext(f.filename or "")]
    if invalid:
        return jsonify({"error": "Only TIFF/PNG/JPG files are supported", "invalid_files": invalid}), 400

    batch_id = str(uuid.uuid4()) if len(files) > 1 else None
    job_ids = [_queue_job(cfg, f, batch_id=batch_id) for f in files]
    if len(job_ids) == 1:
        return redirect(url_for("web.job_detail", id=job_ids[0]))
    return redirect(url_for("web.jobs", batch_id=batch_id))
