import json
import io
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
from flask import Blueprint, current_app, jsonify, redirect, render_template, request, send_file, send_from_directory, url_for
from werkzeug.utils import safe_join, secure_filename

from main import process_form
from ai_extraction.gemini_client import GeminiClient
from ai_extraction.prompt_builder import build_discovery_prompt

bp = Blueprint("web", __name__)

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _cfg_dir() -> Path:
    return Path(current_app.config["CONFIGS_DIR"])


def _dict_dir() -> Path:
    return Path(current_app.config["DICTIONARIES_DIR"])


def _uploads_dir() -> Path:
    return Path(current_app.config["UPLOADS_DIR"])


def _outputs_dir() -> Path:
    return Path(current_app.config["OUTPUTS_DIR"])


def _logs_dir() -> Path:
    return Path(current_app.config["LOGS_DIR"])


def _root_dir() -> Path:
    return Path(current_app.config["ROOT_DIR"])


def _list_configs() -> list[str]:
    return sorted([p.stem for p in _cfg_dir().glob("*.json")])


def _safe_config_name(name: str) -> str:
    import re

    cleaned = (name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cleaned):
        raise ValueError("Invalid config name")
    return cleaned


def _config_path(name: str) -> Path:
    safe = _safe_config_name(name)
    joined = safe_join(str(_cfg_dir()), f"{safe}.json")
    if not joined:
        raise ValueError("Invalid config path")
    return Path(joined)


def _load_config(name: str) -> dict:
    path = _config_path(name)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_config(name: str, payload: dict) -> Path:
    path = _config_path(name)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


def _allowed_ext(filename: str) -> bool:
    ext = Path(filename or "").suffix.lower()
    return ext in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _queue_job(cfg: str, file, batch_id: str | None = None) -> str:
    ext = Path(file.filename or "").suffix.lower()
    job_id = str(uuid.uuid4())
    save_name = f"{job_id}{ext}"
    path = _uploads_dir() / save_name
    file.save(path)

    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "batch_id": batch_id,
            "status": "queued",
            "config_name": cfg,
            "image_path": str(path),
            "original_filename": file.filename,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    threading.Thread(target=_job_runner, args=(job_id, str(path), cfg, file.filename), daemon=True).start()
    return job_id


def _read_audit_entries(limit: int = 200) -> list[dict]:
    audit_path = _outputs_dir() / "audit.jsonl"
    if not audit_path.exists():
        return []
    items: list[dict] = []
    with audit_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(reversed(items[-limit:]))




def _append_review_event(job: dict, reviewer: str, corrections: dict):
    audit_path = _outputs_dir() / "audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "job_id": job.get("job_id"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "review_action",
        "reviewer": reviewer,
        "actions": corrections,
    }
    with audit_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def _job_runner(job_id: str, image_path: str, config_name: str, original_filename: str):
    try:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "running"
        result = process_form(
            image_path=image_path,
            config_name=config_name,
            output_dir=str(_outputs_dir()),
            log_dir=str(_logs_dir()),
            dictionaries_dir=str(_dict_dir()),
            original_filename=original_filename,
            job_id=job_id,
        )
        with JOBS_LOCK:
            JOBS[job_id].update(result)
            JOBS[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(exc)
            JOBS[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()


@bp.route("/")
def index():
    with JOBS_LOCK:
        jobs = sorted(JOBS.values(), key=lambda x: x.get("created_at", ""), reverse=True)
    return render_template("index.html", configs=_list_configs(), jobs=jobs[:10])


@bp.route("/configs", methods=["GET"])
def configs_page():
    return render_template("configs.html", configs=_list_configs())


@bp.route("/configs/new", methods=["GET", "POST"])
def config_new():
    if request.method == "GET":
        return render_template("config_editor.html", mode="new", config_name="", config_text="{}", config={})

    name = _safe_config_name(request.form.get("config_name", ""))
    payload = json.loads(request.form.get("config_json", "{}") or "{}")
    _save_config(name, payload)
    return redirect(url_for("web.config_edit", name=name))


@bp.route("/configs/<name>/edit", methods=["GET", "POST"])
def config_edit(name: str):
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
def config_discover():
    file = request.files.get("template_file")
    if not file:
        return jsonify({"error": "template_file is required"}), 400

    # Save template temporarily for processing
    tmp_path = Path(current_app.config["UPLOADS_DIR"]) / f"tmp_discover_{uuid.uuid4()}.png"
    file.save(tmp_path)

    try:
        with open(tmp_path, "rb") as f:
            image_bytes = f.read()

        client = GeminiClient()
        prompt = build_discovery_prompt()
        result = client.extract_from_images([image_bytes], [prompt])

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@bp.route("/upload", methods=["POST"])
def upload():
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


@bp.route("/jobs", methods=["GET"])
def jobs():
    with JOBS_LOCK:
        items = sorted(JOBS.values(), key=lambda x: x.get("created_at", ""), reverse=True)
    batch_id = (request.args.get("batch_id") or "").strip()
    if batch_id:
        items = [j for j in items if j.get("batch_id") == batch_id]
    return render_template("jobs.html", jobs=items, selected_batch=batch_id)


@bp.route("/jobs/<id>", methods=["GET"])
def job_detail(id: str):
    with JOBS_LOCK:
        job = JOBS.get(id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    audits = _read_audit_entries(limit=500)
    job_audit = next((a for a in audits if str(a.get("job_id")) == id), None)
    return render_template("job_detail.html", job=job, job_audit=job_audit)


@bp.route("/jobs/<id>/review", methods=["GET", "POST"])
def review(id: str):
    with JOBS_LOCK:
        job = JOBS.get(id)
    if not job:
        return jsonify({"error": "job not found"}), 404

    if request.method == "GET":
        return render_template("review.html", job=job)

    payload = request.get_json(silent=True) or {}
    corrections = payload.get("corrections", {})
    reviewer = payload.get("reviewer", "web_user")

    with JOBS_LOCK:
        fields = job.get("fields", [])
        for f in fields:
            fid = f.get("field_id")
            if fid not in corrections:
                continue
            val = corrections[fid]
            if val == "__ILLEGIBLE__":
                f["final_value"] = ""
                f["validation_status"] = "illegible"
                f["correction"] = "illegible"
            else:
                f["final_value"] = val
                f["validation_status"] = "accepted"
                f["correction"] = val
            f["reviewer"] = reviewer
            f["corrected"] = True
            f["needs_review"] = f["validation_status"] == "pending_review"

        job["pending_fields"] = [f for f in fields if f.get("needs_review")]
        job["status"] = "completed" if not job["pending_fields"] else "pending_review"
        job["updated_at"] = datetime.now(timezone.utc).isoformat()

    _append_review_event(job, reviewer, corrections)
    return jsonify({"status": "ok", "job_id": id})


@bp.route("/api/dictionaries/upload", methods=["POST"])
def api_upload_dictionary():
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


@bp.route("/api/jobs", methods=["GET"])
def api_jobs():
    with JOBS_LOCK:
        items = sorted(JOBS.values(), key=lambda x: x.get("created_at", ""), reverse=True)
    return jsonify({"jobs": items})


@bp.route("/audits", methods=["GET"])
def audits():
    entries = _read_audit_entries(limit=500)
    return render_template("audits.html", audits=entries)


@bp.route("/api/template-preview", methods=["GET"])
def api_template_preview():
    template_path = (request.args.get("template_path") or "").strip()
    if not template_path:
        return jsonify({"error": "template_path is required"}), 400
    root = _root_dir().resolve()
    target = (root / template_path).resolve()
    if root not in target.parents and target != root:
        return jsonify({"error": "Invalid template path"}), 400
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
def templates_static(filename: str):
    return send_from_directory(_root_dir() / "templates", filename)


@bp.route("/uploads/<path:filename>")
def uploads_static(filename: str):
    return send_from_directory(_uploads_dir(), filename)


@bp.route("/outputs/<path:filename>")
def outputs_static(filename: str):
    return send_from_directory(_outputs_dir(), filename)
