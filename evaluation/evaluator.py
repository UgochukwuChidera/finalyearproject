import json
from pathlib import Path

from .metrics import aggregate_metrics
from .pipelines.base_pipeline import BasePipeline
from .pipelines.dape_pipeline import DAPEPipeline


class Evaluator:
    def __init__(
        self,
        results_dir: str = "evaluation/results",
    ):
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._pipelines: list[BasePipeline] = [DAPEPipeline()]

    def run(self, forms: list[dict]) -> dict:
        all_results = {}

        for pipeline in self._pipelines:
            per_form_records = []
            metrics = []
            for form in forms:
                fields, stats = pipeline.extract(form["image_path"], form["template_id"])
                m = {"field_accuracy": 0.0, "overall_field_accuracy": 0.0, "n_fields": len(fields)}
                metrics.append(m)
                per_form_records.append(
                    {
                        "form_id": form["form_id"],
                        "template_id": form["template_id"],
                        "pipeline": pipeline.name,
                        "fields": fields,
                        "pipeline_stats": stats,
                        "pre_hitl_metrics": m,
                        "post_hitl_metrics": m,
                    }
                )

            agg = aggregate_metrics(metrics) if metrics else {}
            all_results[pipeline.name] = {
                "pre_hitl_forms": per_form_records,
                "post_hitl_forms": per_form_records,
                "pre_hitl_aggregate": agg,
                "post_hitl_aggregate": agg,
                "hitl_impact": {
                    "pre_hitl_accuracy": agg.get("overall_field_accuracy", 0),
                    "post_hitl_accuracy": agg.get("overall_field_accuracy", 0),
                    "hitl_accuracy_gain": 0,
                    "hitl_gain_pct": 0,
                },
            }

        self._write_results(all_results)
        return all_results

    def _write_results(self, all_results: dict) -> None:
        json_path = self._results_dir / "full_results.json"
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2, default=str, ensure_ascii=False)
