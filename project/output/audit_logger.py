import json
from datetime import datetime,timezone
from pathlib import Path

class AuditLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir=Path(log_dir); self.log_dir.mkdir(parents=True,exist_ok=True)

    def log(self, form_id, template_id, processing_stats, validated_fields, export_paths=None):
        ts=datetime.now(timezone.utc); label=ts.strftime("%Y%m%d_%H%M%S")
        total=len(validated_fields)
        flagged=sum(1 for f in validated_fields if f.get("needs_review",False))
        corrected=sum(1 for f in validated_fields if f.get("corrected",False))
        accepted=sum(1 for f in validated_fields if f.get("validation_status")=="accepted")
        entry={"form_id":form_id,"template_id":template_id,"timestamp":ts.isoformat(),
               "processing_stats":{k:(round(v,6) if isinstance(v,float) else v)
                                   for k,v in processing_stats.items()
                                   if not isinstance(v,(bytes,type(None)))},
               "escalation_summary":{"total_fields":total,"accepted":accepted,
                                     "flagged":flagged,"corrected":corrected,
                                     "escalation_rate":round(flagged/total,4) if total else 0.0,
                                     "auto_accept_rate":round(accepted/total,4) if total else 0.0},
               "export_paths":export_paths or {}}
        p=self.log_dir/f"{form_id}_{label}.json"
        with p.open("w",encoding="utf-8") as fh:
            json.dump(entry,fh,indent=2,default=str,ensure_ascii=False)
        return str(p)
