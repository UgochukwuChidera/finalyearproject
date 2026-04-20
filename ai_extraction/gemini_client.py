import base64
import json
import os
from typing import List

from openai import OpenAI


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str = "google/gemini-2.5-flash-lite"):
        self.model = model
        self.api_key = (api_key or os.getenv("OPENROUTER_API_KEY", "")).strip()
        self.client = (
            OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)
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
        if not self.client or not images:
            return {}

        content = []
        for idx, image_bytes in enumerate(images):
            text = prompts[idx] if idx < len(prompts) else "Extract requested fields from this image."
            content.append({"type": "text", "text": text})
            content.append({"type": "image_url", "image_url": {"url": self._data_url(image_bytes)}})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        raw = (response.choices[0].message.content if response.choices else "") or ""
        return self._safe_json_extract(raw)


# Backwards-compatible alias; deprecated and kept temporarily for migration.
# New code should instantiate GeminiClient directly.
OpenRouterGeminiClient = GeminiClient
