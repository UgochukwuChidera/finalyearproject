"""Review-related routes: accept/reject field corrections on job results."""
from datetime import datetime, timezone

from flask import jsonify, render_template, request

from .bp import bp
from .common import (
    JOBS, JOBS_LOCK, DEFAULT_REVIEWER,
    _init_jobs_internal, _read_audit_entries, _append_review_event, _save_jobs_db,
)


@bp.route("/jobs/<id>/review", methods=["GET", "POST"])
def review(id: str):
    _init_jobs_internal()
    with JOBS_LOCK:
        job = JOBS.get(id)
    if not job:
        return jsonify({"error": "job not found"}), 404

    if request.method == "GET":
        return render_template("review.html", job=job, default_reviewer=DEFAULT_REVIEWER)

    payload = request.get_json(silent=True) or {}
    corrections = payload.get("corrections", {})
    reviewer = (payload.get("reviewer") or "").strip() or DEFAULT_REVIEWER

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
