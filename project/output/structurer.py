from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class OutputStructurer:
    def __init__(self, output_schema: dict[str, str]) -> None:
        self.schema = output_schema

    def structure(
        self,
        validated_fields: list[dict[str, Any]],
        form_id: str,
        template_id: str,
        processing_stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data: dict[str, object] = {}
        for field in validated_fields:
            fid = field["field_id"]
            if fid == "student_signature":
                continue
            sk = self.schema.get(fid, fid)
            data[sk] = field.get("final_value", "")
        return {"form_id": form_id, "template_id": template_id,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "data": data, "fields": validated_fields,
                "processing_stats": processing_stats or {}}
