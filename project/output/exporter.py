from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class DataExporter:
    def export_json(self, structured: dict[str, Any], output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(structured, fh, indent=2, default=str, ensure_ascii=False)
        return str(path)

    def export_csv(self, structured: dict[str, Any], output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["field", "value"])
            writer.writerow(["_form_id", structured.get("form_id", "")])
            writer.writerow(["_template_id", structured.get("template_id", "")])
            writer.writerow(["_processed_at", structured.get("processed_at", "")])
            for k, v in structured.get("data", {}).items():
                writer.writerow([k, v])
        return str(path)

    def export_all(self, structured: dict[str, Any], base_path: str) -> dict[str, str]:
        base = str(Path(base_path).with_suffix(""))
        return {"json": self.export_json(structured, base + ".json"),
                "csv": self.export_csv(structured, base + ".csv")}
