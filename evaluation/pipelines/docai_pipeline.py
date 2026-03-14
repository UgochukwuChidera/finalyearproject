"""
PaddleOCR Pipeline  (Condition 2 — LSTM/ANN Model)
====================================================
Uses PaddleOCR v3, a production-grade OCR engine backed by a
DB (text detection) + CRNN-LSTM (text recognition) architecture.
Unlike Tesseract's classical heuristics, PaddleOCR produces
genuine per-word posterior confidence scores from the LSTM decoder,
making it a proper LSTM/ANN representative for Condition 2.

Why PaddleOCR for Condition 2:
  - Real confidence scores: LSTM decoder outputs a softmax probability
    per character; PaddleOCR exposes the mean as a float in [0, 1]
  - Designed for mixed printed + handwritten document content
  - CPU-viable: ~5–15 seconds per form at 600 DPI
  - ~300 MB download, cached after first run
  - Well-cited in OCR literature (2021–2024), citable as LSTM/ANN baseline
  - No API key, no internet after first download

Architecture note for Chapter 4:
  PaddleOCR uses a three-stage pipeline internally:
    1. DBNet (detection)      — finds text bounding boxes
    2. CRNN-LSTM (recognition)— decodes text from each box
    3. CTC decoder            — outputs character sequences + confidence
  The confidence score exposed by PaddleOCR.ocr() is the mean CTC
  posterior across all characters in a recognised word — this is a
  genuine probabilistic measure, not a heuristic.

Installation:
    pip install paddlepaddle paddleocr
    (CPU-only paddle is ~200 MB; paddleocr models ~100 MB additional)
"""

import time

import cv2
import numpy as np

from .base_pipeline import BasePipeline

_CB_MARKED   = 0.15
_CB_UNMARKED = 0.04


class PaddleOCRPipeline(BasePipeline):
    """
    Condition 2 — PaddleOCR v3 (LSTM/ANN) pipeline.

    Parameters
    ----------
    registry_path : path to templates/registry.json
    lang          : PaddleOCR language model ('en' for English)
    use_gpu       : False for CPU-only (default)
    """

    # Class-level cache — model loads once per process
    _ocr_engine = None
    _loaded     = False

    def __init__(
        self,
        registry_path: str  = "templates/registry.json",
        lang:          str  = "en",
        use_gpu:       bool = False,
    ):
        from project.template_registry import TemplateRegistry
        self._registry = TemplateRegistry(registry_path)
        self._lang     = lang
        self._use_gpu  = use_gpu

    @property
    def name(self) -> str:
        return "PaddleOCR v3 (LSTM/ANN)"

    # ── Lazy model loader ──────────────────────────────────────────────────────

    @classmethod
    def _load_engine(cls, lang: str, use_gpu: bool) -> None:
        if cls._loaded:
            return
        print(f"\n[PaddleOCR] Loading engine (lang={lang}, gpu={use_gpu})…")
        print( "            First run downloads ~300 MB — subsequent runs use cache.\n")
        try:
            from paddleocr import PaddleOCR
            cls._ocr_engine = PaddleOCR(
                use_angle_cls = True,
                lang          = lang,
                use_gpu       = use_gpu,
                show_log      = False,
            )
            cls._loaded = True
            print("[PaddleOCR] Engine ready.\n")
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR requires paddlepaddle and paddleocr.\n"
                "Install with:  pip install paddlepaddle paddleocr\n"
                f"Original error: {exc}"
            ) from exc

    # ── Main extraction ────────────────────────────────────────────────────────

    def extract(
        self,
        image_path: str,
        template_id: str,
    ) -> tuple[list[dict], dict]:

        self._load_engine(self._lang, self._use_gpu)

        t0         = time.perf_counter()
        field_defs = self._registry.get_field_definitions(template_id)
        image      = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        fields = []

        for fdef in field_defs:
            fid   = fdef["id"]
            ftype = fdef["type"]
            x, y, w, h = int(fdef["x"]), int(fdef["y"]), int(fdef["w"]), int(fdef["h"])

            ih, iw = image.shape[:2]
            x1, y1 = max(x, 0),      max(y, 0)
            x2, y2 = min(x + w, iw), min(y + h, ih)

            if x2 <= x1 or y2 <= y1:
                fields.append(self._empty(fid, ftype, x, y, w, h))
                continue

            roi = image[y1:y2, x1:x2]

            if ftype == "checkbox":
                result = self._checkbox(roi)
            else:
                result = self._ocr_roi(roi)

            fields.append({
                "field_id":   fid,
                "field_type": ftype,
                "x": x, "y": y, "w": w, "h": h,
                **result,
            })

        elapsed = time.perf_counter() - t0

        return fields, {
            "processing_time_s": round(elapsed, 4),
            "pipeline":          self.name,
            "lang":              self._lang,
        }

    # ── OCR via PaddleOCR (genuine LSTM confidence) ────────────────────────────

    def _ocr_roi(self, roi: np.ndarray) -> dict:
        """
        Run PaddleOCR on a single field ROI.

        PaddleOCR returns results as:
            [ [ [box_coords], (text, confidence) ], ... ]

        confidence is the mean CTC posterior across recognised characters
        — a genuine probabilistic score from the LSTM decoder, not a heuristic.
        """
        # PaddleOCR works best on BGR images
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        try:
            result = self.__class__._ocr_engine.ocr(roi_bgr, cls=True)
        except Exception as exc:
            return {"value": "", "confidence": 0.0, "error": str(exc)}

        if not result or result[0] is None:
            return {"value": "", "confidence": 0.0}

        words  = []
        confs  = []
        for line in result[0]:
            if line is None:
                continue
            text_conf = line[1]    # (text, confidence)
            text      = str(text_conf[0]).strip()
            conf      = float(text_conf[1])
            if text:
                words.append(text)
                confs.append(conf)

        value   = " ".join(words).strip()
        avg_conf = float(np.mean(confs)) if confs else 0.0

        return {
            "value":      value,
            "confidence": float(np.clip(avg_conf, 0.0, 1.0)),
        }

    # ── Checkbox detection ─────────────────────────────────────────────────────

    @staticmethod
    def _checkbox(roi: np.ndarray) -> dict:
        _, binary = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
        density   = float(np.count_nonzero(binary) / max(binary.size, 1))
        mid       = (_CB_MARKED + _CB_UNMARKED) / 2.0

        if density >= _CB_MARKED:
            marked = True
            conf   = float(np.clip(
                (density - _CB_MARKED) / max(_CB_MARKED, 1e-6), 0.0, 1.0
            ))
        elif density <= _CB_UNMARKED:
            marked = False
            conf   = float(np.clip(
                1.0 - density / max(_CB_UNMARKED, 1e-6), 0.0, 1.0
            ))
        else:
            marked = density > mid
            conf   = 0.25

        return {
            "value":         marked,
            "confidence":    float(np.clip(conf, 0.0, 1.0)),
            "pixel_density": density,
        }

    @staticmethod
    def _empty(fid, ftype, x, y, w, h) -> dict:
        return {
            "field_id":   fid,  "field_type": ftype,
            "x": x, "y": y, "w": w, "h": h,
            "value":      "" if ftype != "checkbox" else False,
            "confidence": 0.0,
        }
