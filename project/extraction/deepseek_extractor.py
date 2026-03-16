"""
DeepSeek Document AI Extractor
===============================
Implements :class:`BaseExtractor` using the DeepSeek Vision API (or any
compatible OpenAI-style vision endpoint) to extract structured fields from
scanned form images.

The extractor encodes the form image as a base-64 PNG, sends it to the
DeepSeek chat-completions endpoint with a structured prompt, and parses the
JSON response into an :class:`ExtractionResult`.

Configuration
-------------
API credentials and endpoint are read from environment variables so that
secrets are never hard-coded:

    DEEPSEEK_API_KEY    — required; your DeepSeek secret key
    DEEPSEEK_API_BASE   — optional; defaults to https://api.deepseek.com/v1
    DEEPSEEK_MODEL      — optional; defaults to deepseek-chat

These can also be passed directly to the constructor (constructor values
take precedence over environment variables).

Installation
------------
    pip install openai          # DeepSeek uses the OpenAI-compatible SDK

Usage example
-------------
    from project.extraction.deepseek_extractor import DeepSeekExtractor
    extractor = DeepSeekExtractor()          # reads key from env
    result = extractor.extract("form.tif", template_id="student_academic_record")
    print(result.fields)
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────────

_DEFAULT_BASE   = "https://api.deepseek.com/v1"
_DEFAULT_MODEL  = "deepseek-chat"
_SYSTEM_PROMPT  = (
    "You are a precise document data extraction assistant.  "
    "The user will provide an image of a filled paper form and a list of "
    "field IDs.  Extract the value for each field and return ONLY a valid "
    "JSON object mapping field_id → value (string for text fields, "
    "true/false for checkboxes).  If a field is blank or unreadable, "
    "use an empty string.  Do not add extra keys or commentary."
)


# ── Extractor class ────────────────────────────────────────────────────────────

from .base import BaseExtractor, ExtractionResult


class DeepSeekExtractor(BaseExtractor):
    """
    DeepSeek Vision / Document AI extractor.

    Sends the full form image to the DeepSeek vision endpoint and asks the
    model to return a JSON mapping of field_id → value.  Field IDs are
    retrieved from the template registry when *template_id* is provided;
    otherwise the model is asked to extract all visible key-value pairs.

    Parameters
    ----------
    api_key       : DeepSeek API key (overrides ``DEEPSEEK_API_KEY`` env var)
    api_base      : API base URL (overrides ``DEEPSEEK_API_BASE`` env var)
    model         : model identifier (overrides ``DEEPSEEK_MODEL`` env var)
    registry_path : path to ``templates/registry.json``
    """

    def __init__(
        self,
        api_key:       str | None = None,
        api_base:      str | None = None,
        model:         str | None = None,
        registry_path: str        = "templates/registry.json",
    ) -> None:
        self._api_key  = api_key  or os.environ.get("DEEPSEEK_API_KEY", "")
        self._api_base = api_base or os.environ.get("DEEPSEEK_API_BASE", _DEFAULT_BASE)
        self._model    = model    or os.environ.get("DEEPSEEK_MODEL",    _DEFAULT_MODEL)
        self._registry_path = registry_path

        if not self._api_key:
            logger.warning(
                "DeepSeekExtractor: no API key found.  "
                "Set DEEPSEEK_API_KEY or pass api_key= to the constructor.  "
                "Calls to .extract() will raise until a key is provided."
            )

        logger.info(
            "DeepSeekExtractor initialised (model=%s, base=%s)",
            self._model, self._api_base,
        )

    # ── Public interface ───────────────────────────────────────────────────────

    def extract(
        self,
        document_path: str,
        template_id: str | None = None,
    ) -> ExtractionResult:
        """
        Extract fields from *document_path* using the DeepSeek Vision API.

        Parameters
        ----------
        document_path : path to the scanned form image (any PIL-readable format)
        template_id   : optional; used to retrieve expected field IDs from
                        the template registry so the prompt is specific

        Returns
        -------
        ExtractionResult
        """
        if not self._api_key:
            raise RuntimeError(
                "DeepSeek API key is not configured.\n"
                "  Option 1: export DEEPSEEK_API_KEY=sk-…\n"
                "  Option 2: pass api_key='sk-…' to DeepSeekExtractor()\n"
                "  Option 3: use --deepseek-key sk-… on the CLI"
            )

        # Encode image as base-64 PNG for the vision API
        image_b64 = self._encode_image(document_path)

        # Determine expected field IDs for the prompt
        field_ids = self._get_field_ids(template_id)
        user_prompt = self._build_prompt(field_ids)

        # Call the API
        raw_response = self._call_api(image_b64, user_prompt)

        # Parse JSON response → fields dict
        fields_out = self._parse_response(raw_response, field_ids)

        full_text = " ".join(
            str(v) for v in fields_out.values() if v not in ("", None, False)
        )

        logger.debug(
            "DeepSeekExtractor: extracted %d fields from %s",
            len(fields_out), document_path,
        )
        return ExtractionResult(
            text=full_text,
            fields=fields_out,
            layout=None,
            raw_ocr=raw_response,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _encode_image(path: str) -> str:
        """Read *path* and return a base-64-encoded PNG data URI."""
        # Re-encode via Pillow so any format (TIF, JPG, …) becomes PNG
        try:
            from PIL import Image  # type: ignore[import]
            import io

            img = Image.open(path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64}"
        except ImportError:
            # Fall back to raw bytes if Pillow is not installed
            with open(path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("utf-8")
            return f"data:image/png;base64,{b64}"

    def _get_field_ids(self, template_id: str | None) -> list[str]:
        """Return the list of field IDs for *template_id*, or [] if unknown."""
        if not template_id:
            return []
        try:
            from project.template_registry import TemplateRegistry
            reg   = TemplateRegistry(self._registry_path)
            defs  = reg.get_field_definitions(template_id)
            return [d["id"] for d in defs]
        except Exception as exc:
            logger.warning("Could not load field IDs for template '%s': %s", template_id, exc)
            return []

    @staticmethod
    def _build_prompt(field_ids: list[str]) -> str:
        if field_ids:
            ids_json = json.dumps(field_ids)
            return (
                f"Extract the values for these fields: {ids_json}\n"
                "Return ONLY a valid JSON object mapping each field_id to its value.  "
                "Use true/false for checkbox fields."
            )
        return (
            "Extract all visible key-value pairs from this form.  "
            "Return ONLY a valid JSON object mapping field names to their values."
        )

    def _call_api(self, image_b64: str, user_prompt: str) -> Any:
        """Send the image + prompt to the DeepSeek API and return the raw response."""
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for DeepSeekExtractor.\n"
                "Install with:  pip install openai"
            ) from exc

        client = OpenAI(api_key=self._api_key, base_url=self._api_base)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_b64}},
                    {"type": "text",      "text": user_prompt},
                ],
            },
        ]

        logger.debug("DeepSeekExtractor: calling %s with model %s", self._api_base, self._model)
        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return response

    @staticmethod
    def _parse_response(response: Any, field_ids: list[str]) -> dict[str, Any]:
        """
        Extract a ``{field_id: value}`` dict from the API response.

        Tries to parse the first code-block or raw JSON in the response text.
        Falls back to an empty dict with the expected field IDs set to ``""``
        if parsing fails.
        """
        try:
            content = response.choices[0].message.content.strip()
        except (AttributeError, IndexError):
            content = str(response)

        # Strip markdown code fences if present
        if "```" in content:
            parts   = content.split("```")
            content = parts[1] if len(parts) > 1 else parts[0]
            if content.startswith("json"):
                content = content[4:]

        try:
            parsed = json.loads(content.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            logger.warning(
                "DeepSeekExtractor: could not parse JSON from API response:\n%s",
                content[:500],
            )

        # Return safe defaults
        return {fid: "" for fid in field_ids}
