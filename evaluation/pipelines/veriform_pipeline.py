import time

from .base_pipeline import BasePipeline


class VeriFormPipeline(BasePipeline):
    def __init__(self, config_name: str = "medical_screening_v1"):
        self._config_name = config_name

    @property
    def name(self) -> str:
        return "VeriForm ICR"

    def extract(self, image_path: str, template_id: str) -> tuple[list[dict], dict]:
        from main import process_form

        t0 = time.perf_counter()
        result = process_form(image_path=image_path, config_name=self._config_name)
        elapsed = time.perf_counter() - t0

        fields = []
        for f in result.get("fields", []):
            fields.append(
                {
                    "field_id": f.get("field_id"),
                    "field_type": f.get("field_type", "string"),
                    "x": f.get("x", 0),
                    "y": f.get("y", 0),
                    "w": f.get("w", 0),
                    "h": f.get("h", 0),
                    "value": f.get("final_value", f.get("value")),
                    "confidence": float(f.get("confidence", 0.0)),
                }
            )

        return fields, {"processing_time_s": round(elapsed, 4), "pipeline": self.name}
