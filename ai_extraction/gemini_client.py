import base64
import json
import os
from dataclasses import dataclass

from openai import OpenAI

from .confidence import logprob_to_confidence


@dataclass
class AIFieldResult:
    value: str
    c_lp: float


class OpenRouterGeminiClient:
    def __init__(self, model: str = "google/gemini-2.5-flash-lite"):
        self.model = model
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.client = (
            OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)
            if self.api_key
            else None
        )

    @staticmethod
    def _encode_png(image_bytes: bytes) -> str:
        return "data:image/png;base64," + base64.b64encode(image_bytes).decode("utf-8")

    @staticmethod
    def _safe_json_extract(raw: str) -> dict:
        raw = (raw or "").strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    return {}
            return {}

    @staticmethod
    def _build_prompt(form_type: str, fields: list[dict]) -> str:
        lines = [
            "Extract the requested fields from this form image.",
            "Return ONLY valid JSON with top-level key 'fields'.",
            "Format: {\"fields\": {\"field_name\": \"value\"}}",
            f"Form type: {form_type}",
            "Field instructions:",
        ]
        for f in fields:
            bb = f.get("bounding_box", {})
            lines.append(
                f"- {f.get('name')}: label_hint={f.get('label_hint','')}, type={f.get('expected_type','string')}, bbox=({bb.get('x',0)},{bb.get('y',0)},{bb.get('w',0)},{bb.get('h',0)})"
            )
        return "\n".join(lines)

    @staticmethod
    def _extract_avg_logprob(resp) -> float | None:
        try:
            choice = resp.choices[0]
            logprobs = getattr(choice, "logprobs", None)
            if not logprobs:
                return None
            content = getattr(logprobs, "content", None) or []
            vals = [float(t.logprob) for t in content if getattr(t, "logprob", None) is not None]
            if not vals:
                return None
            return sum(vals) / len(vals)
        except Exception:
            return None

    def extract_fields_from_image(self, image_bytes: bytes, form_type: str, fields: list[dict]) -> dict[str, AIFieldResult]:
        if not self.client:
            return {f["name"]: AIFieldResult(value="", c_lp=0.5) for f in fields}

        prompt = self._build_prompt(form_type, fields)
        image_url = self._encode_png(image_bytes)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            temperature=0,
            response_format={"type": "json_object"},
            logprobs=True,
        )
        content = (resp.choices[0].message.content if resp.choices else "") or ""
        payload = self._safe_json_extract(content)
        values = payload.get("fields", {}) if isinstance(payload, dict) else {}
        c_lp = logprob_to_confidence(self._extract_avg_logprob(resp))

        out: dict[str, AIFieldResult] = {}
        for f in fields:
            name = f["name"]
            out[name] = AIFieldResult(value=str(values.get(name, "") or ""), c_lp=c_lp)
        return out

    def extract_single_field_from_crop(self, image_bytes: bytes, field: dict, form_type: str) -> AIFieldResult:
        if not self.client:
            return AIFieldResult(value="", c_lp=0.5)

        prompt = (
            "Read the value for a single form field from this crop and return ONLY JSON.\n"
            f"Form type: {form_type}\n"
            f"Field name: {field.get('name')}\n"
            f"Label hint: {field.get('label_hint','')}\n"
            "JSON format: {\"value\": \"...\"}"
        )
        image_url = self._encode_png(image_bytes)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            temperature=0,
            response_format={"type": "json_object"},
            logprobs=True,
        )
        content = (resp.choices[0].message.content if resp.choices else "") or ""
        payload = self._safe_json_extract(content)
        value = str(payload.get("value", "") if isinstance(payload, dict) else "")
        c_lp = logprob_to_confidence(self._extract_avg_logprob(resp))
        return AIFieldResult(value=value, c_lp=c_lp)
