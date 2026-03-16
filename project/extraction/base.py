"""
Extractor Abstraction
=====================
Defines the shared data container and abstract interface used by all
OCR / extraction backends (Tesseract, DeepSeek, PaddleOCR, …).

Placing these in ``project/extraction/base.py`` keeps them close to the
existing field-extraction code and avoids circular imports with the
evaluation package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractionResult:
    """
    Normalised output returned by every extraction backend.

    Attributes
    ----------
    text    : full-page text; blank-separated tokens from all recognised fields
    fields  : ``{field_id: value}`` mapping aligned with the template registry
    layout  : engine-specific bounding-box / layout data (optional)
    raw_ocr : verbatim engine response — useful for HITL display and debugging
    """

    text:    str
    fields:  dict[str, Any]
    layout:  Any = field(default=None, repr=False)
    raw_ocr: Any = field(default=None, repr=False)


class BaseExtractor(ABC):
    """
    Abstract base class for all document extraction backends.

    Every concrete extractor must implement :meth:`extract`, accepting a
    scanned form image path and (for template-aware backends) the template ID.
    The returned :class:`ExtractionResult` is the only structure that
    downstream pipeline stages (HITL, validation, evaluation) depend on.
    """

    @abstractmethod
    def extract(
        self,
        document_path: str,
        template_id: str | None = None,
    ) -> ExtractionResult:
        """
        Run OCR / document AI on *document_path* and return structured output.

        Parameters
        ----------
        document_path : absolute or relative path to a scanned form image
        template_id   : key in ``templates/registry.json``; required by
                        template-aware backends (Tesseract, PaddleOCR)

        Returns
        -------
        ExtractionResult
        """
        raise NotImplementedError
