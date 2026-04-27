import base64
import json
import logging
import os
import time
from typing import List

import httpx
from openai import APIConnectionError, APITimeoutError, OpenAI

from ai_extraction.confidence import compute_C_lp

logger = logging.getLogger(__name__)

_RETRYABLE = (APITimeoutError, APIConnectionError, httpx.TimeoutException, httpx.ConnectError)

# Default model: Qwen2.5-VL-72B — free on OpenRouter, strong multimodal document
# understanding, and reliable logprobs support via the standard OpenAI-compatible API.
# Override with the OPENROUTER_MODEL environment variable if desired.
_DEFAULT_MODEL = "qwen/qwen2.5-vl-72b-instruct:free"


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: int = 180):
        self.model = model or os.getenv("OPENROUTER_MODEL", _DEFAULT_MODEL)
        self.api_key = (api_key or os.getenv("OPENROUTER_API_KEY", "")).strip()
        self.client = (
            OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
                timeout=httpx.Timeout(timeout, connect=30.0),
            )
            if self.api_key
            else None
        )

    @staticmethod
    def _data_url(image_bytes: bytes) -> str:
        return "data:image/png;base64," + base64.b64encode(image_bytes).decode("utf-8")

    @staticmethod
    def _safe_json_extract(raw: str) -> dict:
        payload = (raw or "").strip()
        if not payload:
            return {}
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            start = payload.find("{")
            end = payload.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(payload[start : end + 1])
                except json.JSONDecodeError:
                    return {}
            return {}

    def extract_from_images(self, images: List[bytes], prompts: List[str]) -> dict:
        if not self.client:
            return {"error": "GeminiClient not initialized. Check your OPENROUTER_API_KEY environment variable."}
        if not images:
            return {"error": "No images provided to GeminiClient."}

        content = []
        for idx, image_bytes in enumerate(images):
            text = prompts[idx] if idx < len(prompts) else "Extract requested fields from this image."
            content.append({"type": "text", "text": text})
            content.append({"type": "image_url", "image_url": {"url": self._data_url(image_bytes)}})

        response = None
        last_err = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0,
                    max_tokens=4096,
                    response_format={"type": "json_object"},
                    logprobs=True,
                    top_logprobs=5,
                )
                break
            except _RETRYABLE as exc:
                last_err = exc
                if attempt < 2:
                    wait = 2 ** attempt
                    logger.warning("API call attempt %d failed (%s). Retrying in %ds…", attempt + 1, exc, wait)
                    time.sleep(wait)
                else:
                    logger.warning("API call attempt %d failed (%s). No more retries.", attempt + 1, exc)
            except Exception as exc:
                return {"error": str(exc)}

        if response is None:
            return {"error": f"API call failed after 3 attempts: {last_err}"}

        raw = (response.choices[0].message.content if response.choices else "") or ""
        payload = self._safe_json_extract(raw)

        if "meta" not in payload:
            payload["meta"] = {}

        logger.debug("Model: %s", self.model)
        logger.debug("Raw response (first 500 chars): %s", raw[:500])

        has_logprobs = False
        logprobs_debug = None
        if response.choices and hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs:
            lp = response.choices[0].logprobs
            logprobs_debug = f"logprobs type: {type(lp)}, keys/attrs: {dir(lp)[:10]}"
            has_logprobs = bool(
                getattr(lp, "content", None)
                or getattr(lp, "tokens", None)
                or (isinstance(lp, dict) and lp)
            )

        payload["meta"]["has_logprobs"] = has_logprobs
        payload["meta"]["logprobs_debug"] = logprobs_debug
        if not has_logprobs:
            logger.warning("API response did not contain logprobs. Confidence scores will use fallback values.")
            logger.debug("Response choice attributes: %s", [a for a in dir(response.choices[0]) if not a.startswith("_")])

        if "C_lp" not in payload["meta"]:
            payload["meta"]["C_lp"] = {}

        if response.choices and hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs:
            lp = response.choices[0].logprobs
            logger.debug("Logprobs full debug: %s", str(lp)[:500])
            tokens = getattr(lp, "content", None) or getattr(lp, "tokens", None) or []
            if not tokens and isinstance(lp, dict):
                tokens = lp.get("tokens", []) or lp.get("content", []) or []
                logger.debug("Extracted tokens from dict: %d", len(tokens))
            logger.debug("Processing %d logprob tokens", len(tokens))
            if tokens:
                char_offset = 0
                token_offsets = []
                for t in tokens:
                    start = char_offset
                    tok_str = getattr(t, "token", "")
                    char_offset += len(tok_str)
                    token_offsets.append((start, char_offset, getattr(t, "logprob", 0.0)))

                all_logprobs = [lp for _, _, lp in token_offsets]
                avg_logprob = sum(all_logprobs) / len(all_logprobs) if all_logprobs else 0.0
                payload["meta"]["overall_confidence"] = compute_C_lp([avg_logprob])
                logger.debug("Overall average logprob: %.4f -> confidence: %.4f", avg_logprob, payload["meta"]["overall_confidence"])

                fields = payload.get("fields", {})
                logger.debug("Fields to map: %s", list(fields.keys()))
                for k, v in fields.items():
                    k_idx = raw.find(f'"{k}"')
                    logger.debug("Field '%s' key pos: %d", k, k_idx)

                    if isinstance(v, str):
                        v_str = json.dumps(v)
                    elif isinstance(v, bool):
                        v_str = "true" if v else "false"
                    elif v is None:
                        v_str = "null"
                    else:
                        v_str = str(v)

                    v_idx = raw.find(v_str, k_idx)
                    if v_idx != -1:
                        v_start = v_idx
                        v_end = v_idx + len(v_str)

                        field_logprobs = []
                        for t_start, t_end, lp in token_offsets:
                            if t_end > v_start and t_start < v_end:
                                field_logprobs.append(lp)

                        if field_logprobs:
                            payload["meta"]["C_lp"][k] = compute_C_lp(field_logprobs)

                if not payload["meta"]["C_lp"]:
                    logger.warning("Logprobs were received but no fields matched. Using overall confidence as fallback for all fields.")
                    for k in fields:
                        payload["meta"]["C_lp"][k] = payload["meta"]["overall_confidence"]
                else:
                    logger.debug("Computed C_lp for %d fields: %s", len(payload["meta"]["C_lp"]), list(payload["meta"]["C_lp"].keys()))

        # For every field that still has no C_lp entry, fill it using:
        #   1. The model's self-reported per-field confidence (returned in the "confidence" key), or
        #   2. The overall response confidence derived from logprobs, if available.
        # This handles the common case where logprob token mapping succeeds for only some
        # fields — previously those remaining fields silently fell back to the 0.65 hard-coded
        # default in the pipeline, making almost all scores look identical.
        self_conf = payload.get("confidence")
        self_conf = self_conf if isinstance(self_conf, dict) else {}
        overall_conf = payload["meta"].get("overall_confidence")
        all_fields = payload.get("fields", {})

        missing_fields = [k for k in all_fields if k not in payload["meta"]["C_lp"]]
        if missing_fields:
            filled_self, filled_overall = 0, 0
            for k in missing_fields:
                if k in self_conf:
                    try:
                        payload["meta"]["C_lp"][k] = compute_C_lp(float(self_conf[k]))
                        filled_self += 1
                        continue
                    except (TypeError, ValueError):
                        pass
                if overall_conf is not None:
                    payload["meta"]["C_lp"][k] = overall_conf
                    filled_overall += 1
            if filled_self:
                logger.info("Used self-reported confidence for %d fields.", filled_self)
            if filled_overall:
                logger.info("Used overall logprob confidence as fallback for %d fields.", filled_overall)

        # Legacy path: if C_lp is still completely empty (no logprobs at all and
        # no per-field self-reported values), bulk-fill from self-reported confidence.
        if not payload["meta"]["C_lp"]:
            if self_conf:
                for k, v in self_conf.items():
                    try:
                        payload["meta"]["C_lp"][k] = compute_C_lp(float(v))
                    except (TypeError, ValueError):
                        pass
                if payload["meta"]["C_lp"]:
                    logger.info("Using model self-reported confidence for %d fields.", len(payload["meta"]["C_lp"]))
                else:
                    logger.warning("Model self-reported confidence map was empty or malformed.")
            else:
                logger.warning("No self-reported confidence in response; per-field C_lp will use default fallback.")

        return payload


# Backwards-compatible alias; deprecated and kept temporarily for migration.
# New code should instantiate GeminiClient directly.
OpenRouterGeminiClient = GeminiClient
