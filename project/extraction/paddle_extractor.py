"""
PaddleOCR Extractor
===================
Implements :class:`BaseExtractor` using PaddleOCR v3, a CPU-efficient
deep-learning OCR engine backed by DBNet (text detection) + CRNN-LSTM
(text recognition) with genuine per-word CTC confidence scores.

Why PaddleOCR
-------------
- Fully free, open-source (Apache 2.0)
- CPU-viable: ~5–15 s per form at 600 DPI
- Real probabilistic confidence (LSTM CTC posteriors), not heuristics
- ~300 MB one-time model download; cached after first run
- No API key or internet connection required after download

Installation
------------
    pip install paddlepaddle paddleocr

Environment variables
---------------------
None required; GPU can be enabled via *use_gpu=True*.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from .base import BaseExtractor, ExtractionResult
from .checkbox_extractor import CheckboxExtractor

logger = logging.getLogger(__name__)


class PaddleOCRExtractor(BaseExtractor):
    """
    Template-aware PaddleOCR v3 extractor.

    Uses PaddleOCR to OCR each field ROI defined in the template registry.
    The PaddleOCR model is loaded lazily and cached at the class level so
    it is only initialised once per process.

    Parameters
    ----------
    registry_path : path to ``templates/registry.json``
    lang          : PaddleOCR language code (``'en'`` for English)
    use_gpu       : set ``True`` to use GPU (default: CPU-only)
    """

    # Class-level model cache — loaded once per process
    _ocr_engine = None
    _loaded     = False

    def __init__(
        self,
        registry_path: str  = "templates/registry.json",
        lang:          str  = "en",
        use_gpu:       bool = False,
    ) -> None:
        from project.template_registry import TemplateRegistry

        self._registry = TemplateRegistry(registry_path)
        self._lang     = lang
        self._use_gpu  = use_gpu
        self._checkbox = CheckboxExtractor()
        logger.info(
            "PaddleOCRExtractor initialised (lang=%s, gpu=%s, registry=%s)",
            lang, use_gpu, registry_path,
        )

    # ── Lazy engine loader ─────────────────────────────────────────────────────

    @classmethod
    def _load_engine(cls, lang: str, use_gpu: bool) -> None:
        if cls._loaded:
            return
        logger.info("Loading PaddleOCR engine (lang=%s, gpu=%s)…", lang, use_gpu)
        print(
            f"\n[PaddleOCR] Loading engine (lang={lang}, gpu={use_gpu})…\n"
            "            First run downloads ~300 MB — subsequent runs use cache.\n"
        )
        try:
            from paddleocr import PaddleOCR  # type: ignore[import]

            cls._ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=use_gpu,
                show_log=False,
            )
            cls._loaded = True
            print("[PaddleOCR] Engine ready.\n")
            logger.info("PaddleOCR engine ready.")
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is not installed.\n"
                "Install with:  pip install paddlepaddle paddleocr\n"
                f"Original error: {exc}"
            ) from exc

    # ── Public interface ───────────────────────────────────────────────────────

    def extract(
        self,
        document_path: str,
        template_id: str | None = None,
    ) -> ExtractionResult:
        """
        Extract all fields from *document_path* using PaddleOCR.

        Parameters
        ----------
        document_path : path to the scanned form image
        template_id   : required — key in registry.json

        Returns
        -------
        ExtractionResult
        """
        if not template_id:
            raise ValueError(
                "PaddleOCRExtractor requires a template_id.  "
                "Pass template_id='<id>' or use --template-id on the CLI."
            )

        self._load_engine(self._lang, self._use_gpu)

        image = cv2.imread(document_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot load image: {document_path}")

        field_defs  = self._registry.get_field_definitions(template_id)
        fields_out: dict[str, Any] = {}
        text_parts: list[str]      = []
        raw_list:   list[dict]     = []

        ih, iw = image.shape[:2]

        for fdef in field_defs:
            fid   = fdef["id"]
            ftype = fdef["type"]
            x, y, w, h = int(fdef["x"]), int(fdef["y"]), int(fdef["w"]), int(fdef["h"])

            x1, y1 = max(x, 0),      max(y, 0)
            x2, y2 = min(x + w, iw), min(y + h, ih)

            if x2 <= x1 or y2 <= y1:
                fields_out[fid] = False if ftype == "checkbox" else ""
                continue

            roi = image[y1:y2, x1:x2]

            if ftype == "checkbox":
                result          = self._checkbox.extract(roi, None)
                fields_out[fid] = result.get("value", False)
            else:
                result          = self._ocr_roi(roi)
                value           = result.get("value", "")
                fields_out[fid] = value
                if value:
                    text_parts.append(value)

            raw_list.append({"field_id": fid, "field_type": ftype, **result})

        full_text = "\n".join(text_parts)
        logger.debug(
            "PaddleOCRExtractor: extracted %d fields from %s",
            len(fields_out), document_path,
        )
        return ExtractionResult(
            text=full_text,
            fields=fields_out,
            layout=None,
            raw_ocr=raw_list,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _ocr_roi(self, roi: np.ndarray) -> dict:
        """Run PaddleOCR on a single field ROI and return a value/confidence dict."""
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        try:
            result = self.__class__._ocr_engine.ocr(roi_bgr, cls=True)
        except Exception as exc:
            logger.warning("PaddleOCR inference error: %s", exc)
            return {"value": "", "confidence": 0.0, "error": str(exc)}

        if not result or result[0] is None:
            return {"value": "", "confidence": 0.0}

        words: list[str]   = []
        confs: list[float] = []

        for line in result[0]:
            if line is None:
                continue
            text_conf = line[1]          # (text, confidence)
            text = str(text_conf[0]).strip()
            conf = float(text_conf[1])
            if text:
                words.append(text)
                confs.append(conf)

        value    = " ".join(words).strip()
        avg_conf = float(np.mean(confs)) if confs else 0.0

        return {
            "value":      value,
            "confidence": float(np.clip(avg_conf, 0.0, 1.0)),
        }
