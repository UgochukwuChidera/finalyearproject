"""
Tesseract Baseline Pipeline  (Condition 1)
==========================================
Raw Tesseract OCR on field ROIs defined by the template registry.

What this pipeline deliberately OMITS (to isolate the value of DAPE):
  - Image preprocessing (grayscale only)
  - Template alignment
  - Differential analysis / interaction masking
  - Any postprocessing of OCR output

Field coordinates still come from the template registry — the comparison
is between extraction strategies, not field localization strategies.

For checkbox fields: pixel density on the raw ROI (no differential mask).
For text fields   : Tesseract on the raw cropped ROI.
"""

import time

import cv2
import numpy as np
import pytesseract

from .base_pipeline import BasePipeline


class TesseractPipeline(BasePipeline):
    """
    Condition 1 — Tesseract only, no preprocessing or differential analysis.
    """

    _CFG = "--oem 3 --psm 7"

    def __init__(
        self,
        registry_path: str = "templates/registry.json",
        tesseract_cmd: str | None = None,
        checkbox_threshold: float = 0.10,
    ):
        from project.template_registry import TemplateRegistry
        self._registry = TemplateRegistry(registry_path)
        self._checkbox_threshold = checkbox_threshold
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    @property
    def name(self) -> str:
        return "Tesseract (Baseline)"

    def extract(
        self,
        image_path: str,
        template_id: str,
    ) -> tuple[list[dict], dict]:
        t0 = time.perf_counter()

        # Minimal load — grayscale only, no enhancement
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        field_defs = self._registry.get_field_definitions(template_id)
        fields = []

        for fdef in field_defs:
            fid   = fdef["id"]
            ftype = fdef["type"]
            x, y, w, h = int(fdef["x"]), int(fdef["y"]), int(fdef["w"]), int(fdef["h"])

            ih, iw = image.shape[:2]
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w, iw), min(y + h, ih)

            if x2 <= x1 or y2 <= y1:
                fields.append(self._empty(fid, ftype, x, y, w, h))
                continue

            roi = image[y1:y2, x1:x2]

            if ftype == "checkbox":
                result = self._checkbox(roi)
            else:
                result = self._ocr(roi)

            fields.append({
                "field_id":   fid,
                "field_type": ftype,
                "x": x, "y": y, "w": w, "h": h,
                **result,
            })

        elapsed = time.perf_counter() - t0
        return fields, {"processing_time_s": round(elapsed, 4), "pipeline": self.name}

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _ocr(self, roi: np.ndarray) -> dict:
        try:
            data = pytesseract.image_to_data(
                roi, config=self._CFG,
                output_type=pytesseract.Output.DICT
            )
        except Exception as exc:
            return {"value": "", "confidence": 0.0, "error": str(exc)}

        words, confs = [], []
        for text, conf in zip(data["text"], data["conf"]):
            text = str(text).strip()
            conf = int(conf)
            if text and conf >= 0:
                words.append(text)
                confs.append(conf)

        value   = " ".join(words).strip()
        avg_conf = float(np.mean(confs) / 100.0) if confs else 0.0
        return {"value": value, "confidence": float(np.clip(avg_conf, 0, 1))}

    def _checkbox(self, roi: np.ndarray) -> dict:
        _, binary = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
        density  = float(np.count_nonzero(binary) / max(binary.size, 1))
        marked   = density >= self._checkbox_threshold

        # Confidence: distance from threshold boundary
        gap = abs(density - self._checkbox_threshold)
        conf = float(np.clip(gap / self._checkbox_threshold, 0, 1))
        if density > self._checkbox_threshold * 0.5 and density < self._checkbox_threshold * 1.5:
            conf = 0.25   # ambiguous zone

        return {"value": marked, "confidence": conf, "pixel_density": density}

    @staticmethod
    def _empty(fid, ftype, x, y, w, h) -> dict:
        return {
            "field_id": fid, "field_type": ftype,
            "x": x, "y": y, "w": w, "h": h,
            "value": "" if ftype != "checkbox" else False,
            "confidence": 0.0,
        }
