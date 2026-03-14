"""
DAPE Pipeline Wrapper  (Condition 3)
=====================================
Thin adapter around the existing DAPEOrchestrator that:

  1. Disables internal HITL (HITL is managed externally by the Evaluator
     so all three pipelines go through the exact same interface)
  2. Returns extracted fields in the unified format expected by the Evaluator
  3. Passes through all pipeline stats for logging

The full DAPE pipeline (preprocessing → alignment → differential analysis →
extraction → confidence validation) runs exactly as documented in Chapter 3.
No modifications to the core architecture.
"""

import time

from .base_pipeline import BasePipeline


class DAPEPipeline(BasePipeline):
    """
    Condition 3 — Full DAPE differential analysis pipeline.
    HITL is disabled internally; the Evaluator applies it externally.
    """

    def __init__(
        self,
        registry_path:        str        = "templates/registry.json",
        confidence_threshold: float      = 0.60,
        tesseract_cmd:        str | None = None,
    ):
        from project.orchestrator import DAPEOrchestrator

        # Disable HITL here — the Evaluator manages it for all pipelines
        self._orchestrator = DAPEOrchestrator(
            registry_path        = registry_path,
            output_dir           = "outputs/_dape_eval",
            log_dir              = "logs/_dape_eval",
            confidence_threshold = confidence_threshold,
            enable_hitl          = False,   # ← critical: evaluator handles HITL
            tesseract_cmd        = tesseract_cmd,
        )
        self._threshold = confidence_threshold

    @property
    def name(self) -> str:
        return "DAPE (Full Pipeline)"

    def extract(
        self,
        image_path: str,
        template_id: str,
    ) -> tuple[list[dict], dict]:
        t0 = time.perf_counter()

        result = self._orchestrator.process(
            image_path  = image_path,
            template_id = template_id,
        )

        elapsed = time.perf_counter() - t0

        # The orchestrator already ran validation internally.
        # Pull the validated fields so the Evaluator sees confidence + status.
        validated = result["structured_output"].get("fields", [])

        # Normalise to the unified field format (strip validation metadata
        # so the external validator re-runs fresh for fair comparison).
        fields = []
        for f in validated:
            fields.append({
                "field_id":   f["field_id"],
                "field_type": f["field_type"],
                "x":          f.get("x", 0),
                "y":          f.get("y", 0),
                "w":          f.get("w", 0),
                "h":          f.get("h", 0),
                "value":      f.get("value", ""),
                "confidence": float(f.get("confidence", 0.0)),
            })

        stats = {
            "processing_time_s": round(elapsed, 4),
            "pipeline":          self.name,
            **{
                k: v for k, v in result["stats"].items()
                if isinstance(v, (int, float, str, bool))
            },
        }
        return fields, stats
