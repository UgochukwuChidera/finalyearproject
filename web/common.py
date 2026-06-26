"""Shared helper functions, state, and configuration for web routes."""
import json
import io
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
from flask import current_app
from werkzeug.utils import safe_join, secure_filename

# ---------------------------------------------------------------------------
# Job Persistence & Management
# ---------------------------------------------------------------------------

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
DEFAULT_REVIEWER = "web_user"


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
    "max_concurrent": 5,
    "requests_per_minute": 30,
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
