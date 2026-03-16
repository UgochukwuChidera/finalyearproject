"""
Extractor Factory / Registry
=============================
Central lookup table for all extraction backends.  Use :func:`get_extractor`
to obtain a configured :class:`~project.extraction.base.BaseExtractor`
instance by name.

Supported names
---------------
- ``tesseract``   — :class:`~project.extraction.tesseract_extractor.TesseractFormExtractor`
- ``paddle``      — :class:`~project.extraction.paddle_extractor.PaddleOCRExtractor`
- ``deepseek``    — :class:`~project.extraction.deepseek_extractor.DeepSeekExtractor`

The default extractor is ``tesseract``, preserving existing behaviour.

Extending the registry
----------------------
Add a new entry to ``_REGISTRY`` mapping a name string to a zero-argument
factory callable (or a callable that accepts **kwargs)::

    _REGISTRY["mybackend"] = lambda **kw: MyExtractor(**kw)

Environment variable
--------------------
``OCR_EXTRACTOR`` — set the default extractor name when *name* is ``None``::

    export OCR_EXTRACTOR=paddle

Example
-------
    from project.extraction.factory import get_extractor

    extractor = get_extractor("tesseract", registry_path="templates/registry.json")
    result    = extractor.extract("form.tif", template_id="student_academic_record")
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .base import BaseExtractor

logger = logging.getLogger(__name__)

# ── Registry ───────────────────────────────────────────────────────────────────

def _make_tesseract(**kwargs: Any) -> BaseExtractor:
    from .tesseract_extractor import TesseractFormExtractor
    return TesseractFormExtractor(**kwargs)


def _make_paddle(**kwargs: Any) -> BaseExtractor:
    from .paddle_extractor import PaddleOCRExtractor
    return PaddleOCRExtractor(**kwargs)


def _make_deepseek(**kwargs: Any) -> BaseExtractor:
    from .deepseek_extractor import DeepSeekExtractor
    return DeepSeekExtractor(**kwargs)


#: Maps lower-cased extractor names to factory callables.
#: Add new entries here to register additional backends.
_REGISTRY: dict[str, Any] = {
    "tesseract": _make_tesseract,
    "paddle":    _make_paddle,
    "paddleocr": _make_paddle,   # alias
    "deepseek":  _make_deepseek,
    "docai":     _make_deepseek, # alias (legacy name used in evaluation)
}

_DEFAULT_EXTRACTOR = "tesseract"


# ── Public API ─────────────────────────────────────────────────────────────────

def list_extractors() -> list[str]:
    """Return the canonical extractor names (excluding aliases)."""
    return ["tesseract", "paddle", "deepseek"]


def get_extractor(
    name: str | None = None,
    **kwargs: Any,
) -> BaseExtractor:
    """
    Return a configured :class:`BaseExtractor` for the given backend *name*.

    Parameters
    ----------
    name   : extractor identifier — ``'tesseract'``, ``'paddle'``, or
             ``'deepseek'``; falls back to the ``OCR_EXTRACTOR`` environment
             variable, then to ``'tesseract'``
    **kwargs
           : forwarded directly to the extractor constructor
             (e.g. ``registry_path``, ``tesseract_cmd``, ``api_key``)

    Returns
    -------
    BaseExtractor

    Raises
    ------
    ValueError
        If *name* is not found in the registry.

    Examples
    --------
    >>> ext = get_extractor("tesseract", registry_path="templates/registry.json")
    >>> ext = get_extractor("paddle", lang="en", use_gpu=False)
    >>> ext = get_extractor("deepseek", api_key="sk-…")
    """
    resolved = (name or os.environ.get("OCR_EXTRACTOR", _DEFAULT_EXTRACTOR)).lower().strip()

    factory = _REGISTRY.get(resolved)
    if factory is None:
        known = ", ".join(f"'{k}'" for k in sorted(_REGISTRY))
        raise ValueError(
            f"Unknown extractor '{resolved}'.  "
            f"Available extractors: {known}.  "
            f"Set OCR_EXTRACTOR environment variable or pass --extractor on the CLI."
        )

    logger.info("Creating extractor '%s' with kwargs=%s", resolved, list(kwargs))
    return factory(**kwargs)
