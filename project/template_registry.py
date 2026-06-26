from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class TemplateRegistry:
    def __init__(self, registry_path: str = "templates/registry.json") -> None:
        self._path = Path(registry_path)
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as fh:
                self._data = json.load(fh)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, ensure_ascii=False)

    def list_templates(self) -> list[str]:
        return list(self._data.keys())

    def get_entry(self, template_id: str) -> dict:
        if template_id not in self._data:
            raise KeyError(f"Template '{template_id}' not found. Available: {self.list_templates()}")
        return self._data[template_id]

    def get_template_image(self, template_id: str) -> np.ndarray:
        entry = self.get_entry(template_id)
        img = cv2.imread(entry["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Template image not found: {entry['image_path']}")
        return img

    def get_field_definitions(self, template_id: str) -> list[dict]:
        return self.get_entry(template_id).get("fields", [])

    def get_output_schema(self, template_id: str) -> dict:
        return self.get_entry(template_id).get("output_schema", {})

    def register_template(
        self,
        template_id: str,
        image_path: str,
        fields: list[dict],
        output_schema: dict | None = None,
    ) -> None:
        self._data[template_id] = {
            "image_path":    image_path,
            "fields":        fields,
            "output_schema": output_schema or {f["id"]: f["id"] for f in fields},
        }
        self._save()
