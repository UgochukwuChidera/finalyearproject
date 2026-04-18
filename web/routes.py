import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, send_from_directory, url_for

from main import process_form

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


def _list_configs() -> list[str]:
    return sorted([p.stem for p in _cfg_dir().glob("*.json")])


def _load_config(name: str) -> dict:
    path = _cfg_dir() / f"{name}.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_config(name: str, payload: dict) -> Path:
    path = _cfg_dir() / f"{name}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


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
    return render_template("index.html", configs=_list_configs())


@bp.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("index.html", configs=_list_configs())

    cfg = request.form.get("config_name", "").strip()
    file = request.files.get("form_file")
    if not cfg or not file:
        return jsonify({"error": "config_name and form_file are required"}), 400

    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        return jsonify({"error": "Only TIFF/PNG/JPG files are supported"}), 400

    job_id = str(uuid.uuid4())
    save_name = f"{job_id}{ext}"
    path = _uploads_dir() / save_name
    file.save(path)

    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "config_name": cfg,
            "image_path": str(path),
            "original_filename": file.filename,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    t = threading.Thread(target=_job_runner, args=(job_id, str(path), cfg, file.filename), daemon=True)
    t.start()

    return redirect(url_for("web.jobs", highlight=job_id))


@bp.route("/jobs")
def jobs():
    with JOBS_LOCK:
        items = sorted(JOBS.values(), key=lambda x: x.get("created_at", ""), reverse=True)
    return render_template("jobs.html", jobs=items, highlight=request.args.get("highlight"))


@bp.route("/jobs/<job_id>/review")
def review(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return render_template("review.html", job=job)


@bp.route("/audit")
def audit():
    entries = []
    p = _outputs_dir() / "audit.jsonl"
    if p.exists():
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    entries.reverse()
    return render_template("index.html", configs=_list_configs(), audit_entries=entries)


@bp.route("/configs")
def configs_page():
    selected = request.args.get("name")
    selected_payload = None
    if selected:
        try:
            selected_payload = _load_config(selected)
        except Exception:
            selected_payload = None
    return render_template(
        "config_editor.html",
        configs=_list_configs(),
        selected=selected,
        selected_payload=json.dumps(selected_payload, indent=2) if selected_payload else "",
    )


@bp.route("/dictionaries", methods=["POST"])
def upload_dictionary():
    file = request.files.get("dictionary_file")
    if not file:
        return redirect(url_for("web.configs_page"))
    dest = _dict_dir() / Path(file.filename).name
    file.save(dest)
    store = current_app.config["DICTIONARY_STORE"]
    current_app.config["DICTIONARIES_CACHE"] = store.load()
    return redirect(url_for("web.configs_page"))


@bp.route("/api/configs", methods=["GET", "POST"])
@bp.route("/api/configs/", methods=["GET", "POST"])
def api_configs():
    if request.method == "GET":
        return jsonify({"configs": _list_configs()})

    payload = request.get_json(silent=True) or {}
    form_type = (payload.get("form_type") or "").strip()
    if not form_type:
        return jsonify({"error": "form_type is required"}), 400
    _save_config(form_type, payload)
    return jsonify({"status": "ok", "form_type": form_type})


@bp.route("/api/configs/<name>", methods=["GET", "PUT", "DELETE"])
def api_config_one(name: str):
    path = _cfg_dir() / f"{name}.json"

    if request.method == "GET":
        if not path.exists():
            return jsonify({"error": "not found"}), 404
        with path.open("r", encoding="utf-8") as fh:
            return jsonify(json.load(fh))

    if request.method == "DELETE":
        if path.exists():
            path.unlink()
        return jsonify({"status": "deleted", "name": name})

    payload = request.get_json(silent=True) or {}
    _save_config(name, payload)
    return jsonify({"status": "updated", "name": name})


@bp.route("/api/jobs", methods=["GET"])
def api_jobs():
    with JOBS_LOCK:
        items = sorted(JOBS.values(), key=lambda x: x.get("created_at", ""), reverse=True)
    return jsonify({"jobs": items})


@bp.route("/api/jobs/<job_id>/review", methods=["POST"])
def api_job_review(job_id: str):
    payload = request.get_json(silent=True) or {}
    corrections = payload.get("corrections", {})
    reviewer = payload.get("reviewer", "web_user")

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "job not found"}), 404
        fields = job.get("fields", [])
        for f in fields:
            fid = f.get("field_id")
            if fid in corrections:
                val = corrections[fid]
                if val == "__ILLEGIBLE__":
                    f["final_value"] = ""
                    f["validation_status"] = "illegible"
                    f["correction"] = "illegible"
                else:
                    f["final_value"] = val
                    f["validation_status"] = "corrected"
                    f["correction"] = val
                f["corrected"] = True
                f["needs_review"] = False
                f["reviewer"] = reviewer

        job["pending_fields"] = [f for f in fields if f.get("needs_review")]
        job["status"] = "completed" if not job["pending_fields"] else "pending_review"

    return jsonify({"status": "ok", "job_id": job_id})


@bp.route("/uploads/<path:filename>")
def uploads_static(filename: str):
    return send_from_directory(_uploads_dir(), filename)


@bp.route("/outputs/<path:filename>")
def outputs_static(filename: str):
    return send_from_directory(_outputs_dir(), filename)
