import json
from datetime import datetime, timezone
from pathlib import Path


class AuditLogger:
    def __init__(self, log_dir="logs", audit_jsonl_path="outputs/audit.jsonl"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.audit_jsonl_path = Path(audit_jsonl_path)
        self.audit_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def _base_entry(self, form_id, template_id, processing_stats, validated_fields, export_paths):
        ts = datetime.now(timezone.utc)
        total = len(validated_fields)
        flagged = sum(1 for f in validated_fields if f.get("needs_review", False))
        corrected = sum(1 for f in validated_fields if f.get("corrected", False))
        accepted = sum(1 for f in validated_fields if f.get("validation_status") in {"accepted", "accepted_spot_check"})

        return {
            "form_id": form_id,
            "template_id": template_id,
            "timestamp": ts.isoformat(),
            "processing_stats": {
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in processing_stats.items()
                if not isinstance(v, (bytes, type(None)))
            },
            "escalation_summary": {
                "total_fields": total,
                "accepted": accepted,
                "flagged": flagged,
                "corrected": corrected,
                "escalation_rate": round(flagged / total, 4) if total else 0.0,
                "auto_accept_rate": round(accepted / total, 4) if total else 0.0,
            },
            "export_paths": export_paths or {},
        }

    def log(
        self,
        form_id,
        template_id,
        processing_stats,
        validated_fields,
        export_paths=None,
        original_filename=None,
        extra=None,
    ):
        entry = self._base_entry(form_id, template_id, processing_stats, validated_fields, export_paths)

        ts = datetime.now(timezone.utc)
        label = ts.strftime("%Y%m%d_%H%M%S")
        p = self.log_dir / f"{form_id}_{label}.json"
        with p.open("w", encoding="utf-8") as fh:
            json.dump(entry, fh, indent=2, default=str, ensure_ascii=False)

        ai_extractions = []
        for f in validated_fields:
            ai_extractions.append(
                {
                    "field_name": f.get("field_id"),
                    "raw_value": f.get("value"),
                    "C_lp": float(f.get("C_lp", f.get("confidence", 0.0))),
                    "C_dict": float(f.get("C_dict", 0.0)),
                    "C_final": float(f.get("confidence", 0.0)),
                    "status": f.get("validation_status"),
                    "final_value": f.get("final_value"),
                    "reviewer": f.get("reviewer"),
                    "correction": f.get("correction"),
                }
            )

        jsonl_entry = {
            "job_id": form_id,
            "timestamp": ts.isoformat(),
            "form_type": template_id,
            "original_filename": original_filename,
            "extractions": ai_extractions,
            "review_summary": entry.get("escalation_summary", {}),
        }
        if extra and isinstance(extra, dict):
            jsonl_entry.update(extra)

        with self.audit_jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(jsonl_entry, ensure_ascii=False, default=str) + "\n")

        return str(p)
