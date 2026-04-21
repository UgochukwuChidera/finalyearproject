import json
from pathlib import Path

from .ground_truth import GroundTruth, compare_field
from .metrics import aggregate_metrics, compute_form_metrics, hitl_impact
from .pipelines.base_pipeline import BasePipeline
from .pipelines.dape_pipeline import DAPEPipeline


class Evaluator:
    def __init__(
        self,
        results_dir: str = "evaluation/results",
        ground_truth_dir: str = "ground_truth",
        confidence_threshold: float = 0.60,
    ):
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._gt = GroundTruth(ground_truth_dir)
        self._threshold = confidence_threshold
        self._pipelines: list[BasePipeline] = [DAPEPipeline()]

    def run(self, forms: list[dict]) -> dict:
        all_results = {}

        for pipeline in self._pipelines:
            pre_form_records: list[dict] = []
            post_form_records: list[dict] = []
            pre_metrics_list: list[dict] = []
            post_metrics_list: list[dict] = []

            for form in forms:
                fields, pipeline_stats = pipeline.extract(form["image_path"], form["template_id"])
                processing_time = float(pipeline_stats.get("processing_time_s", 0.0))

                gt_record = self._gt.get(form["form_id"])
                gt_fields: dict = (gt_record.get("fields", {}) if gt_record else {}) or {}

                # ── Pre-HITL: use raw extracted values ────────────────────────
                pre_comparisons = self._compare_fields(fields, gt_fields)
                pre_hitl_stats = self._hitl_stats(fields, corrected=False)
                pre_m = compute_form_metrics(pre_comparisons, pre_hitl_stats, processing_time)
                pre_metrics_list.append(pre_m)

                # ── Post-HITL: apply simulated corrections for needs_review ───
                corrected_fields = self._simulate_corrections(fields, gt_fields)
                post_comparisons = self._compare_fields(corrected_fields, gt_fields)
                post_hitl_stats = self._hitl_stats(corrected_fields, corrected=True)
                post_m = compute_form_metrics(post_comparisons, post_hitl_stats, processing_time)
                post_metrics_list.append(post_m)

                pre_form_records.append(
                    {
                        "form_id": form["form_id"],
                        "template_id": form["template_id"],
                        "pipeline": pipeline.name,
                        "fields": fields,
                        "comparisons": pre_comparisons,
                        "pipeline_stats": pipeline_stats,
                        "has_ground_truth": gt_record is not None,
                        "pre_hitl_metrics": pre_m,
                        "post_hitl_metrics": post_m,
                    }
                )
                post_form_records.append(
                    {
                        **pre_form_records[-1],
                        "fields": corrected_fields,
                        "comparisons": post_comparisons,
                    }
                )

            pre_agg = aggregate_metrics(pre_metrics_list) if pre_metrics_list else {}
            post_agg = aggregate_metrics(post_metrics_list) if post_metrics_list else {}
            impact = hitl_impact(pre_agg, post_agg)

            all_results[pipeline.name] = {
                "pre_hitl_forms": pre_form_records,
                "post_hitl_forms": post_form_records,
                "pre_hitl_aggregate": pre_agg,
                "post_hitl_aggregate": post_agg,
                "hitl_impact": impact,
            }

        self._write_results(all_results)
        return all_results

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _compare_fields(self, fields: list[dict], gt_fields: dict) -> list[dict]:
        """Compare each extracted field against its ground truth value."""
        comparisons: list[dict] = []
        for f in fields:
            fid = f.get("field_id", "")
            field_type = f.get("field_type", "string")
            extracted = f.get("final_value", f.get("value"))
            gt_value = gt_fields.get(fid)
            if gt_value is None:
                continue
            result = compare_field(extracted, gt_value, field_type)
            comparisons.append(
                {
                    **result,
                    "field_id": fid,
                    "field_type": field_type,
                    "extracted_value": extracted,
                    "ground_truth_value": gt_value,
                    "extracted_bool": bool(extracted) if field_type == "checkbox" else None,
                    "needs_review": f.get("needs_review", False),
                    "corrected": f.get("corrected", False),
                }
            )
        return comparisons

    def _hitl_stats(self, fields: list[dict], corrected: bool) -> dict:
        total = len(fields)
        flagged = sum(1 for f in fields if f.get("needs_review", False))
        corrected_count = sum(1 for f in fields if f.get("corrected", False)) if corrected else 0
        return {
            "total_fields": total,
            "flagged_count": flagged,
            "corrected_count": corrected_count,
            "escalation_rate": round(flagged / total, 4) if total else 0.0,
        }

    def _simulate_corrections(self, fields: list[dict], gt_fields: dict) -> list[dict]:
        """
        Simulate post-HITL state: for fields flagged for review that have a
        known ground truth, assume a human corrected them to the GT value.
        """
        result: list[dict] = []
        for f in fields:
            fid = f.get("field_id", "")
            if f.get("needs_review") and fid in gt_fields:
                f = {
                    **f,
                    "final_value": gt_fields[fid],
                    "corrected": True,
                    "needs_review": False,
                }
            result.append(f)
        return result

    def _write_results(self, all_results: dict) -> None:
        json_path = self._results_dir / "full_results.json"
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2, default=str, ensure_ascii=False)
