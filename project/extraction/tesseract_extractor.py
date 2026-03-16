"""
Tesseract Form Extractor
========================
Implements :class:`BaseExtractor` using Tesseract OCR, configured for
form-processing use cases (``--oem 3 --psm 6``).

This backend wraps the existing :class:`~project.extraction.ocr_extractor.OCRExtractor`
and :class:`~project.extraction.checkbox_extractor.CheckboxExtractor` so
field-level logic is not duplicated.  Field bounding boxes are read from the
template registry.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2

from .base import BaseExtractor, ExtractionResult
from .checkbox_extractor import CheckboxExtractor
from .ocr_extractor import OCRExtractor

logger = logging.getLogger(__name__)


class TesseractFormExtractor(BaseExtractor):
    """
    Template-aware Tesseract OCR extractor.

    Applies Tesseract with ``--oem 3 --psm 6`` (full-page block mode, best
    available engine) to each field ROI defined in the template registry.
    Checkbox fields use pixel-density detection rather than OCR.

    Parameters
    ----------
    registry_path : path to ``templates/registry.json``
    tesseract_cmd : optional explicit path to the Tesseract binary
    """

    def __init__(
        self,
        registry_path: str = "templates/registry.json",
        tesseract_cmd: str | None = None,
    ) -> None:
        from project.template_registry import TemplateRegistry

        self._registry = TemplateRegistry(registry_path)
        self._ocr      = OCRExtractor(tesseract_cmd=tesseract_cmd)
        self._checkbox = CheckboxExtractor()
        logger.info("TesseractFormExtractor initialised (registry=%s)", registry_path)

    def extract(
        self,
        document_path: str,
        template_id: str | None = None,
    ) -> ExtractionResult:
        """
        Extract all fields from *document_path* using Tesseract.

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
                "TesseractFormExtractor requires a template_id.  "
                "Pass template_id='<id>' or use --template-id on the CLI."
            )

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
                result         = self._checkbox.extract(roi, None)
                fields_out[fid] = result.get("value", False)
            else:
                result         = self._ocr.extract(roi, ftype)
                value          = result.get("value", "")
                fields_out[fid] = value
                if value:
                    text_parts.append(value)

            raw_list.append({"field_id": fid, "field_type": ftype, **result})

        full_text = "\n".join(text_parts)
        logger.debug(
            "TesseractFormExtractor: extracted %d fields from %s",
            len(fields_out), document_path,
        )
        return ExtractionResult(
            text=full_text,
            fields=fields_out,
            layout=None,
            raw_ocr=raw_list,
        )
