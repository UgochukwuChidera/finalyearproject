"""Job-related routes: queue, run, list, delete, export."""
from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from flask import current_app, jsonify, redirect, render_template, request, send_file
from werkzeug.datastructures import FileStorage

from main import process_form
from .bp import bp
from .common import (
    JOBS, JOBS_LOCK, DEFAULT_REVIEWER,
    _root_dir, _cfg_dir, _dict_dir, _uploads_dir, _outputs_dir, _logs_dir,
    _norm_text, _safe_config_name, _allowed_ext, _load_jobs_db, _save_jobs_db,
    _init_jobs_internal, _get_batch_settings, _get_job_semaphore, _get_api_key,
    _LAST_JOB_START, _RATE_LOCK, _read_audit_entries, _list_configs,
)


def _queue_job(cfg: str, file: FileStorage, batch_id: str | None = None) -> str:
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
    app: Any, job_id: str, image_path: str, config_name: str, original_filename: str,
    sem: threading.Semaphore, inter_request_delay: float
) -> None:
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


def _job_runner(
    app: Any, job_id: str, image_path: str, config_name: str, original_filename: str
) -> None:
    def progress_cb(stage: str, message: str) -> None:  # type: ignore[no-untyped-def]
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


@bp.route("/")
def index() -> str:
    _init_jobs_internal()
    with JOBS_LOCK:
        jobs = sorted(JOBS.values(), key=lambda x: x.get("created_at", ""), reverse=True)
    return render_template("index.html", configs=_list_configs(), jobs=jobs[:10])


@bp.route("/jobs", methods=["GET"])
def jobs() -> str:
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
def job_detail(id: str) -> str:
    _init_jobs_internal()
    with JOBS_LOCK:
        job = JOBS.get(id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    audits = _read_audit_entries(limit=500)
    job_audit = next((a for a in audits if str(a.get("job_id")) == id), None)
    return render_template("job_detail.html", job=job, job_audit=job_audit)


@bp.route("/api/jobs", methods=["GET"])
def api_jobs() -> Any:
    _init_jobs_internal()
    with JOBS_LOCK:
        items = sorted(JOBS.values(), key=lambda x: x.get("created_at", ""), reverse=True)
    return jsonify({"jobs": items})


@bp.route("/api/jobs/<id>", methods=["DELETE"])
def api_job_delete(id: str) -> Any:
    with JOBS_LOCK:
        if id in JOBS:
            del JOBS[id]
            _save_jobs_db(JOBS)
            return jsonify({"status": "ok"})
    return jsonify({"error": "job not found"}), 404


@bp.route("/api/jobs/batch-delete", methods=["POST"])
def api_jobs_batch_delete() -> Any:
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


@bp.route("/jobs/<id>/exports/<fmt>", methods=["GET"])
def job_export_download(id: str, fmt: str) -> Any:
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
        current_app.logger.warning("Rejected export path outside outputs for job %s: %s", id, str(target))
        return jsonify({"error": "invalid export path"}), 400

    if not target.exists():
        return jsonify({"error": f"{fmt.upper()} export file is missing"}), 404

    return send_file(target, as_attachment=True, download_name=f"{id}.{fmt}")
