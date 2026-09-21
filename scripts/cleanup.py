#!/usr/bin/env python3
"""Retention: prune outputs/crops older than 30d, logs/jobs_db.json via MAX_JOBS.
Uploads are never auto-deleted (AGENTS.md). Dry-run by default."""

import time
from pathlib import Path

RETENTION_DAYS = 30
for p in Path("outputs/crops").rglob("*"):
    if p.is_file() and time.time() - p.stat().st_mtime > RETENTION_DAYS * 86400:
        print(f"rm {p}")  # replace with p.unlink() to enable
