"""
DeepSeek Vision Pipeline  (Condition 2 — Document AI)
======================================================
Uses the DeepSeek Vision API (OpenAI-compatible endpoint) to extract fields
from scanned form images.

Unlike Tesseract or PaddleOCR, DeepSeek receives the **full page image** and
returns a structured JSON mapping of field_id → value.  Field IDs are taken
from the template registry so the model knows exactly what to look for.

Installation
------------
    pip install openai

Configuration
-------------
Set the API key via environment variable before running:

    export DEEPSEEK_API_KEY=sk-...

or pass ``--deepseek-key sk-...`` to ``run_evaluation.py``.
"""

import time

from .base_pipeline import BasePipeline


class DeepSeekPipeline(BasePipeline):
    """
    Condition 2 — DeepSeek Vision Document AI pipeline.

    Parameters
    ----------
    registry_path  : path to templates/registry.json
    api_key        : DeepSeek API key (overrides DEEPSEEK_API_KEY env var)
    model          : model name (default: deepseek-chat)
    """

    def __init__(
        self,
        registry_path: str        = "templates/registry.json",
        api_key:       str | None = None,
        model:         str | None = None,
    ):
        from project.template_registry import TemplateRegistry
        from project.extraction.deepseek_extractor import DeepSeekExtractor

        self._registry  = TemplateRegistry(registry_path)
        self._extractor = DeepSeekExtractor(
            api_key       = api_key,
            model         = model,
            registry_path = registry_path,
        )

    @property
    def name(self) -> str:
        return "DeepSeek Vision (Document AI)"

    def extract(
        self,
        image_path:  str,
        template_id: str,
    ) -> tuple[list[dict], dict]:
        t0 = time.perf_counter()

        ext_result = self._extractor.extract(image_path, template_id=template_id)

        field_defs = self._registry.get_field_definitions(template_id)
        fields     = []

        for fdef in field_defs:
            fid   = fdef["id"]
            ftype = fdef["type"]
            x, y, w, h = int(fdef["x"]), int(fdef["y"]), int(fdef["w"]), int(fdef["h"])

            raw_value = ext_result.fields.get(fid, False if ftype == "checkbox" else "")

            # Normalise checkbox value
            if ftype == "checkbox":
                if isinstance(raw_value, str):
                    value = raw_value.strip().lower() in ("true", "yes", "1", "checked", "x")
                else:
                    value = bool(raw_value)
            else:
                value = str(raw_value) if raw_value not in (None, False) else ""

            fields.append({
                "field_id":   fid,
                "field_type": ftype,
                "x": x, "y": y, "w": w, "h": h,
                "value":      value,
                "confidence": 0.90,   # treat Document AI output as high-confidence
            })

        elapsed = time.perf_counter() - t0
        return fields, {
            "processing_time_s": round(elapsed, 4),
            "pipeline":          self.name,
        }
