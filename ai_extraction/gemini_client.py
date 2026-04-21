import base64
import io
import json
import os
from typing import List

from openai import OpenAI
from PIL import Image


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
    def _process_image(image_bytes: bytes, max_dim: int = 1200) -> bytes:
        """Resizes image to a max dimension and converts to JPEG to save tokens."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode in ("RGBA", "P", "CMYK"):
                img = img.convert("RGB")
            
            if max(img.width, img.height) > max_dim:
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=85)
            return output.getvalue()
        except Exception:
            return image_bytes

    @staticmethod
    def _data_url(image_bytes: bytes) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")

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
            processed_bytes = self._process_image(image_bytes)
            text = prompts[idx] if idx < len(prompts) else "Extract requested fields from this image."
            content.append({"type": "text", "text": text})
            content.append({"type": "image_url", "image_url": {"url": self._data_url(processed_bytes)}})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )

        raw = (response.choices[0].message.content if response.choices else "") or ""
        return self._safe_json_extract(raw)


# Backwards-compatible alias; deprecated and kept temporarily for migration.
# New code should instantiate GeminiClient directly.
OpenRouterGeminiClient = GeminiClient
