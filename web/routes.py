import json
import io
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
from flask import Blueprint, current_app, jsonify, redirect, render_template, request, send_file, send_from_directory, url_for
from werkzeug.utils import safe_join, secure_filename

from main import process_form
from ai_extraction.gemini_client import GeminiClient, get_active_model
from ai_extraction.prompt_builder import build_discovery_prompt

bp = Blueprint("web", __name__)

# ---------------------------------------------------------------------------
# Job Persistence & Management
# ---------------------------------------------------------------------------

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _root_dir() -> Path:
    return Path(current_app.config["ROOT_DIR"])


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


def _jobs_db_path() -> Path:
    return _logs_dir() / "jobs_db.json"


def _load_jobs_db() -> dict:
    path = _jobs_db_path()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_jobs_db(jobs_data: dict) -> None:
    path = _jobs_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(jobs_data, fh, indent=2, ensure_ascii=False)


def _init_jobs_internal():
    global JOBS
    if not JOBS:
        loaded = _load_jobs_db()
        # Mark any jobs that were still running/queued when the server stopped
        changed = False
        for job in loaded.values():
            if job.get("status") in ("running", "queued"):
                job["status"] = "interrupted"
                job["error"] = "Server was restarted while this job was in progress."
                job["updated_at"] = datetime.now(timezone.utc).isoformat()
                changed = True
        JOBS.update(loaded)
        if changed:
            _save_jobs_db(JOBS)


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


def _list_configs() -> list[str]:
    return sorted([p.stem for p in _cfg_dir().glob("*.json")])


def _save_config(name: str, payload: dict) -> Path:
    path = _config_path(name)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


def _allowed_ext(filename: str) -> bool:
    ext = Path(filename or "").suffix.lower()
    return ext in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


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


def _norm_text(value) -> str:
    return str(value or "").strip().lower()


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

    batch_cfg = _get_batch_settings()
    sem = _get_job_semaphore(batch_cfg["max_concurrent"])
    inter_delay = float(batch_cfg.get("inter_request_delay", 0.5))

    app = current_app._get_current_object()
    threading.Thread(
        target=_job_runner_gated,
        args=(app, job_id, str(path), cfg, file.filename, sem, inter_delay),
        daemon=True,
    ).start()
    return job_id


def _job_runner_gated(
    app, job_id: str, image_path: str, config_name: str, original_filename: str,
    sem: threading.Semaphore, inter_request_delay: float
):
    """Wrapper that rate-limits job execution via a semaphore + inter-request delay."""
    global _LAST_JOB_START
    # Wait for a concurrency slot
    sem.acquire()
    try:
        # Enforce minimum gap between consecutive job starts
        if inter_request_delay > 0:
            with _RATE_LOCK:
                now = time.monotonic()
                elapsed = now - _LAST_JOB_START
                if elapsed < inter_request_delay:
                    time.sleep(inter_request_delay - elapsed)
                _LAST_JOB_START = time.monotonic()
        _job_runner(app, job_id, image_path, config_name, original_filename)
    finally:
        sem.release()


def _job_runner(app, job_id: str, image_path: str, config_name: str, original_filename: str):
    def progress_cb(stage, message):
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["current_stage"] = stage
                JOBS[job_id]["current_message"] = message
                if "logs" not in JOBS[job_id]:
                    JOBS[job_id]["logs"] = []
                JOBS[job_id]["logs"].append({
                    "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                    "stage": stage,
                    "message": message
                })

    with app.app_context():
        try:
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "running"
                JOBS[job_id]["logs"] = []
            
            result = process_form(
                image_path=image_path,
                config_name=config_name,
                output_dir=str(_outputs_dir()),
                log_dir=str(_logs_dir()),
                dictionaries_dir=str(_dict_dir()),
                original_filename=original_filename,
                job_id=job_id,
                progress_callback=progress_cb,
                api_key=_get_api_key()
            )
            with JOBS_LOCK:
                JOBS[job_id].update(result)
                JOBS[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
                JOBS[job_id]["current_message"] = "Processing complete."
                _save_jobs_db(JOBS)
        except Exception as exc:
            import traceback
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["error"] = str(exc)
                JOBS[job_id]["traceback"] = traceback.format_exc()
                JOBS[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
                _save_jobs_db(JOBS)


# ---------------------------------------------------------------------------
# Model configuration helpers
# ---------------------------------------------------------------------------

_BUILTIN_MODELS = [
    {
        "id": "openai/gpt-4o-mini",
        "label": "GPT-4o Mini",
        "description": "Best default: fast, cheap, image-capable, reliable logprobs. Recommended for OCR pipeline use.",
        "vision": True,
        "logprobs": True,
        "free_tier": False
    },
    {
        "id": "google/gemini-2.0-flash-001",
        "label": "Gemini 2.0 Flash",
        "description": "Fast multimodal model from Google. Supports image input. Logprobs available via OpenRouter.",
        "vision": True,
        "logprobs": True,
        "free_tier": False
    },
    {
        "id": "google/gemini-2.5-flash-preview:free",
        "label": "Gemini 2.5 Flash (Free)",
        "description": "Free-tier Gemini 2.5 Flash with vision support. Good for bulk OCR within free trial limits. Logprobs may be limited.",
        "vision": True,
        "logprobs": False,
        "free_tier": True
    },
    {
        "id": "google/gemini-2.5-flash-preview",
        "label": "Gemini 2.5 Flash",
        "description": "Paid tier Gemini 2.5 Flash. Faster rate limits, image support, logprobs available.",
        "vision": True,
        "logprobs": True,
        "free_tier": False
    },
    {
        "id": "meta-llama/llama-4-maverick:free",
        "label": "Llama 4 Maverick (Free)",
        "description": "Meta's Llama 4 Maverick, free tier. Multimodal with image support. Good OCR alternative within free quota.",
        "vision": True,
        "logprobs": False,
        "free_tier": True
    },
    {
        "id": "meta-llama/llama-4-maverick",
        "label": "Llama 4 Maverick",
        "description": "Paid tier Llama 4 Maverick. Image support, logprobs returned by OpenRouter.",
        "vision": True,
        "logprobs": True,
        "free_tier": False
    },
    {
        "id": "qwen/qwen2.5-vl-72b-instruct:free",
        "label": "Qwen 2.5 VL 72B (Free)",
        "description": "Alibaba's top vision-language model, free tier. Excellent at reading documents, forms, handwriting. High OCR accuracy.",
        "vision": True,
        "logprobs": False,
        "free_tier": True
    },
    {
        "id": "qwen/qwen2.5-vl-72b-instruct",
        "label": "Qwen 2.5 VL 72B",
        "description": "Paid tier. Best VLM for document OCR tasks. Supports logprobs via OpenRouter.",
        "vision": True,
        "logprobs": True,
        "free_tier": False
    },
    {
        "id": "qwen/qwen2.5-vl-7b-instruct:free",
        "label": "Qwen 2.5 VL 7B (Free)",
        "description": "Smaller, faster Qwen vision model on free tier. Good for high-volume TIFF processing within token quota.",
        "vision": True,
        "logprobs": False,
        "free_tier": True
    },
    {
        "id": "mistralai/mistral-small-3.1-24b-instruct:free",
        "label": "Mistral Small 3.1 24B (Free)",
        "description": "Mistral's multimodal model, free tier. Supports image input. Useful as a free fallback for OCR.",
        "vision": True,
        "logprobs": False,
        "free_tier": True
    }
]

_DEFAULT_MODELS_CONFIG: dict = {
    "active_model": "openai/gpt-4o-mini",
    "api_key": "",
    "models": _BUILTIN_MODELS,
}


def _models_config_path() -> Path:
    return _root_dir() / "models.json"


def _load_models_config() -> dict:
    path = _models_config_path()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return dict(_DEFAULT_MODELS_CONFIG)


def _save_models_config(data: dict) -> None:
    path = _models_config_path()
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Batch / Rate-limit settings
# ---------------------------------------------------------------------------

_BATCH_DEFAULTS: dict = {
    "max_concurrent": 3,
    "requests_per_minute": 10,
    "inter_request_delay": 0.5,
}


def _get_batch_settings() -> dict:
    cfg = _load_models_config()
    stored = cfg.get("batch_settings") or {}
    merged = dict(_BATCH_DEFAULTS)
    merged.update({k: v for k, v in stored.items() if v is not None})
    return merged


def _save_batch_settings(data: dict) -> None:
    cfg = _load_models_config()
    current = cfg.get("batch_settings") or {}
    current.update(data)
    cfg["batch_settings"] = current
    _save_models_config(cfg)


# ---------------------------------------------------------------------------
# Semaphore-based concurrency + rate-limiter
# ---------------------------------------------------------------------------

_JOB_SEM: threading.Semaphore | None = None
_JOB_SEM_LOCK = threading.Lock()

# Tracks the last time any job thread actually started work (wall clock).
_LAST_JOB_START: float = 0.0
_RATE_LOCK = threading.Lock()


def _make_semaphore(n: int) -> threading.Semaphore:
    return threading.Semaphore(max(1, n))


def _get_job_semaphore(max_concurrent: int) -> threading.Semaphore:
    """Return (or lazily create) the global job semaphore."""
    global _JOB_SEM
    with _JOB_SEM_LOCK:
        if _JOB_SEM is None:
            _JOB_SEM = _make_semaphore(max_concurrent)
    return _JOB_SEM


def _reset_job_semaphore(max_concurrent: int) -> None:
    """Replace the global semaphore (called when settings change)."""
    global _JOB_SEM
    with _JOB_SEM_LOCK:
        _JOB_SEM = _make_semaphore(max_concurrent)


def _get_api_key() -> str:
    """Resolve API key: models.json -> OPENROUTER_API_KEY env -> empty string."""
    cfg = _load_models_config()
    key = cfg.get("api_key", "").strip()
    if key:
        return key
    return os.getenv("OPENROUTER_API_KEY", "").strip()

@bp.route("/")
def index():
    _init_jobs_internal()
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
def config_delete(name: str):
    safe_name = _safe_config_name(name)
    path = _config_path(safe_name)
    if path.exists():
        path.unlink()
    return redirect(url_for("web.configs_page"))


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
    _init_jobs_internal()
    with JOBS_LOCK:
        items = sorted(JOBS.values(), key=lambda x: x.get("created_at", ""), reverse=True)
    q = (request.args.get("q") or "").strip()
    sort_key = (request.args.get("sort") or "created_at").strip()
    sort_dir = (request.args.get("dir") or "desc").strip().lower()
    batch_id = (request.args.get("batch_id") or "").strip()
    if batch_id:
        items = [j for j in items if j.get("batch_id") == batch_id]
    if q:
        qn = _norm_text(q)
        def _matches(job: dict) -> bool:
            pending_count = len(job.get("pending_fields", []) or [])
            haystack = [
                job.get("job_id"),
                job.get("config_name"),
                job.get("status"),
                job.get("batch_id"),
                job.get("created_at"),
                job.get("updated_at"),
                job.get("original_filename"),
                job.get("review_finalized_by"),
                str(pending_count),
            ]
            return any(qn in _norm_text(v) for v in haystack)
        items = [j for j in items if _matches(j)]

    key_funcs = {
        "job_id": lambda j: _norm_text(j.get("job_id")),
        "config_name": lambda j: _norm_text(j.get("config_name")),
        "status": lambda j: _norm_text(j.get("status")),
        "batch_id": lambda j: _norm_text(j.get("batch_id")),
        "created_at": lambda j: _norm_text(j.get("created_at")),
        "updated_at": lambda j: _norm_text(j.get("updated_at")),
        "pending": lambda j: len(j.get("pending_fields", []) or []),
    }
    if sort_key not in key_funcs:
        sort_key = "created_at"
    if sort_dir not in {"asc", "desc"}:
        sort_dir = "desc"
    items = sorted(items, key=key_funcs[sort_key], reverse=(sort_dir == "desc"))

    return render_template(
        "jobs.html",
        jobs=items,
        selected_batch=batch_id,
        q=q,
        sort=sort_key,
        direction=sort_dir,
    )


@bp.route("/jobs/<id>", methods=["GET"])
def job_detail(id: str):
    _init_jobs_internal()
    with JOBS_LOCK:
        job = JOBS.get(id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    audits = _read_audit_entries(limit=500)
    job_audit = next((a for a in audits if str(a.get("job_id")) == id), None)
    return render_template("job_detail.html", job=job, job_audit=job_audit)


@bp.route("/jobs/<id>/review", methods=["GET", "POST"])
def review(id: str):
    _init_jobs_internal()
    with JOBS_LOCK:
        job = JOBS.get(id)
    if not job:
        return jsonify({"error": "job not found"}), 404

    if request.method == "GET":
        return render_template("review.html", job=job)

    payload = request.get_json(silent=True) or {}
    corrections = payload.get("corrections", {})
    reviewer = (payload.get("reviewer") or "web_user").strip() or "web_user"

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
        if not job["pending_fields"]:
            job["status"] = "finalized"
            job["review_finalized"] = True
            job["review_finalized_at"] = datetime.now(timezone.utc).isoformat()
            job["review_finalized_by"] = reviewer
        else:
            job["status"] = "pending_review"
            job["review_finalized"] = False
            job["review_finalized_at"] = None
            job["review_finalized_by"] = None
        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_jobs_db(JOBS)

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
    _init_jobs_internal()
    with JOBS_LOCK:
        items = sorted(JOBS.values(), key=lambda x: x.get("created_at", ""), reverse=True)
    return jsonify({"jobs": items})


@bp.route("/audits", methods=["GET"])
def audits():
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


@bp.route("/jobs/<id>/exports/<fmt>", methods=["GET"])
def job_export_download(id: str, fmt: str):
    _init_jobs_internal()
    fmt = (fmt or "").strip().lower()
    if fmt not in {"json", "csv", "xlsx"}:
        return jsonify({"error": "unsupported export format"}), 400

    with JOBS_LOCK:
        job = JOBS.get(id)
    if not job:
        return jsonify({"error": "job not found"}), 404

    export_path = (job.get("export_paths") or {}).get(fmt)
    if not export_path:
        return jsonify({"error": f"{fmt.upper()} export not available for this job"}), 404

    target = Path(str(export_path))
    if not target.is_absolute():
        target = (_root_dir() / target).resolve()
    else:
        target = target.resolve()

    try:
        target.relative_to(_outputs_dir().resolve())
    except ValueError:
        return jsonify({"error": "invalid export path"}), 400

    if not target.exists():
        return jsonify({"error": f"{fmt.upper()} export file is missing"}), 404

    return send_file(target, as_attachment=True, download_name=f"{id}.{fmt}")


@bp.route("/api/template-preview", methods=["GET"])
def api_template_preview():
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
def templates_static(filename: str):
    return send_from_directory(_root_dir() / "templates", filename)


@bp.route("/uploads/<path:filename>")
def uploads_static(filename: str):
    return send_from_directory(_uploads_dir(), filename)


@bp.route("/outputs/<path:filename>")
def outputs_static(filename: str):
    return send_from_directory(_outputs_dir(), filename)


@bp.route("/api/utils/convert-to-png", methods=["POST"])
def api_utils_convert_to_png():
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
def evaluation():
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
def settings():
    cfg = _load_models_config()
    return render_template("settings.html", models_config=cfg, active_model=get_active_model())


@bp.route("/api/models", methods=["GET"])
def api_models_get():
    cfg = _load_models_config()
    cfg["resolved_active"] = get_active_model()
    return jsonify(cfg)


@bp.route("/api/models", methods=["POST"])
def api_models_update():
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
def api_batch_settings_get():
    return jsonify(_get_batch_settings())


@bp.route("/api/settings/batch", methods=["POST"])
def api_batch_settings_update():
    payload = request.get_json(silent=True) or {}
    updates: dict = {}

    max_concurrent = payload.get("max_concurrent")
    if max_concurrent is not None:
        updates["max_concurrent"] = max(1, min(int(max_concurrent), 20))

    rpm = payload.get("requests_per_minute")
    if rpm is not None:
        updates["requests_per_minute"] = max(1, int(rpm))
        # Auto-derive inter-request delay from RPM; guard denominator even though max(1,...) already ensures >= 1
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


@bp.route("/api/jobs/<id>", methods=["DELETE"])
def api_job_delete(id: str):
    with JOBS_LOCK:
        if id in JOBS:
            del JOBS[id]
            _save_jobs_db(JOBS)
            return jsonify({"status": "ok"})
    return jsonify({"error": "job not found"}), 404


@bp.route("/api/jobs/batch-delete", methods=["POST"])
def api_jobs_batch_delete():
    payload = request.get_json(silent=True) or {}
    ids = payload.get("job_ids", [])
    if not isinstance(ids, list):
        return jsonify({"error": "job_ids must be a list"}), 400
    
    with JOBS_LOCK:
        count = 0
        for jid in ids:
            if jid in JOBS:
                del JOBS[jid]
                count += 1
        if count > 0:
            _save_jobs_db(JOBS)
    return jsonify({"status": "ok", "deleted_count": count})


@bp.route("/help")
def help_center():
    return render_template("help.html")
