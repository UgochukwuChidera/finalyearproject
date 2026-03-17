"""
Evaluator
=========
Runs all six evaluation conditions against the 45-form dataset and
produces results tables for Chapter 5.

Six conditions:
  ┌─────────────────────────────┬──────────┬───────────┐
  │ Pipeline                    │ No HITL  │ With HITL │
  ├─────────────────────────────┼──────────┼───────────┤
  │ 1. Tesseract (Baseline)     │ Cond 1A  │ Cond 1B   │
  │ 2. PaddleOCR (LSTM/ANN)     │ Cond 2A  │ Cond 2B   │
  │ 3. DAPE (Full Pipeline)     │ Cond 3A  │ Cond 3B   │
  └─────────────────────────────┴──────────┴───────────┘

When ``deepseek_api_key`` is provided the PaddleOCR condition is replaced
by DeepSeek Vision for a Document AI comparison.

For each form:
  1. Run pipeline → raw extracted fields
  2. Run unified validator → validated fields (pre-HITL)
  3. Record pre-HITL metrics
  4. Run HITL interface → apply corrections
  5. Record post-HITL metrics

All results are written to evaluation/results/.
"""

import json
import time
import traceback
from pathlib import Path

from .pipelines.base_pipeline      import BasePipeline
from .pipelines.tesseract_pipeline import TesseractPipeline
from .pipelines.docai_pipeline     import PaddleOCRPipeline
from .pipelines.dape_pipeline      import DAPEPipeline
from .unified_hitl                import UnifiedHITL
from .ground_truth                import GroundTruth, compare_field
from .metrics                     import compute_form_metrics, aggregate_metrics, hitl_impact


class Evaluator:
    """
    Orchestrates all six evaluation conditions.

    Parameters
    ----------
    registry_path        : path to templates/registry.json
    ground_truth_dir     : directory containing per-form ground truth JSON files
    results_dir          : output directory for results tables
    confidence_threshold : shared threshold for all three pipelines' HITL
    deepseek_api_key     : optional; when provided, replaces PaddleOCR with
                           DeepSeek Vision pipeline for Condition 2
    deepseek_model       : optional DeepSeek model name (default: deepseek-chat)
    hitl_host, hitl_port : Flask review UI address
    tesseract_cmd        : optional Tesseract binary path
    """

    def __init__(
        self,
        registry_path:        str        = "templates/registry.json",
        ground_truth_dir:     str        = "ground_truth",
        results_dir:          str        = "evaluation/results",
        confidence_threshold: float      = 0.60,
        deepseek_api_key:     str | None = None,
        deepseek_model:       str | None = None,
        hitl_host:            str        = "127.0.0.1",
        hitl_port:            int        = 5050,
        tesseract_cmd:        str | None = None,
    ):
        self._registry_path = registry_path
        self._results_dir   = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._threshold     = confidence_threshold

        # Ground truth
        self._gt = GroundTruth(ground_truth_dir)

        # Condition 2: DeepSeek when key is provided, PaddleOCR otherwise
        if deepseek_api_key:
            from .pipelines.deepseek_pipeline import DeepSeekPipeline
            condition2: BasePipeline = DeepSeekPipeline(
                registry_path = registry_path,
                api_key       = deepseek_api_key,
                model         = deepseek_model,
            )
        else:
            condition2 = PaddleOCRPipeline(
                registry_path = registry_path,
            )

        # Three pipelines
        self._pipelines: list[BasePipeline] = [
            TesseractPipeline(
                registry_path = registry_path,
                tesseract_cmd = tesseract_cmd,
            ),
            condition2,
            DAPEPipeline(
                registry_path        = registry_path,
                confidence_threshold = confidence_threshold,
                tesseract_cmd        = tesseract_cmd,
            ),
        ]

        # Unified HITL — shared across all pipelines
        self._hitl = UnifiedHITL(
            confidence_threshold = confidence_threshold,
            enable_hitl          = True,
            hitl_host            = hitl_host,
            hitl_port            = hitl_port,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        forms: list[dict],
    ) -> dict:
        """
        Run all pipelines on all forms and collect results.

        Parameters
        ----------
        forms : list of dicts, each with keys:
            image_path  : str
            template_id : str
            form_id     : str  (must match a ground truth file)

        Returns
        -------
        dict — full results keyed by pipeline name, with sub-keys:
            pre_hitl_forms, post_hitl_forms,
            pre_hitl_aggregate, post_hitl_aggregate,
            hitl_impact
        """
        all_results = {}

        for pipeline in self._pipelines:
            print(f"\n{'='*60}")
            print(f"  Evaluating: {pipeline.name}")
            print(f"{'='*60}")

            pre_metrics_list  = []
            post_metrics_list = []
            per_form_records  = []

            for form in forms:
                record = self._evaluate_form(pipeline, form)
                if record is None:
                    continue
                pre_metrics_list.append(record["pre_hitl_metrics"])
                post_metrics_list.append(record["post_hitl_metrics"])
                per_form_records.append(record)

            pre_agg  = aggregate_metrics(pre_metrics_list)
            post_agg = aggregate_metrics(post_metrics_list)
            impact   = hitl_impact(pre_agg, post_agg)

            all_results[pipeline.name] = {
                "pre_hitl_forms":      per_form_records,
                "post_hitl_forms":     per_form_records,
                "pre_hitl_aggregate":  pre_agg,
                "post_hitl_aggregate": post_agg,
                "hitl_impact":         impact,
            }

            print(f"\n  ✓ {pipeline.name}")
            print(f"    Pre-HITL  accuracy : {pre_agg.get('overall_field_accuracy', 0):.1%}")
            print(f"    Post-HITL accuracy : {post_agg.get('overall_field_accuracy', 0):.1%}")
            print(f"    HITL gain          : +{impact.get('hitl_gain_pct', 0):.1f}pp")

        # Write all results to disk
        self._write_results(all_results)
        return all_results

    # ── Per-form evaluation ────────────────────────────────────────────────────

    def _evaluate_form(
        self,
        pipeline: BasePipeline,
        form:     dict,
    ) -> dict | None:
        form_id     = form["form_id"]
        template_id = form["template_id"]
        image_path  = form["image_path"]

        gt = self._gt.get(form_id)
        if gt is None:
            print(f"  [WARN] No ground truth for {form_id} — skipping.")
            return None

        gt_fields = gt.get("fields", {})

        # Load field definitions for this template
        from project.template_registry import TemplateRegistry
        registry   = TemplateRegistry(self._registry_path)
        field_defs = registry.get_field_definitions(template_id)

        # ── Extract ────────────────────────────────────────────────────────────
        try:
            t0 = time.perf_counter()
            extracted_fields, pipeline_stats = pipeline.extract(image_path, template_id)
            extraction_time = time.perf_counter() - t0
        except Exception as exc:
            print(f"  [ERROR] {pipeline.name} failed on {form_id}: {exc}")
            traceback.print_exc()
            return None

        # ── Pre-HITL validation + metrics ──────────────────────────────────────
        pre_validated, pre_hitl_stats = self._hitl.validate_only(
            extracted_fields, field_defs
        )
        pre_comparisons = self._build_comparisons(pre_validated, gt_fields)
        pre_metrics     = compute_form_metrics(
            pre_comparisons, pre_hitl_stats, extraction_time
        )

        # ── Post-HITL validation + metrics ─────────────────────────────────────
        post_validated, post_hitl_stats = self._hitl.run(
            extracted_fields, field_defs
        )
        post_comparisons = self._build_comparisons(post_validated, gt_fields)
        post_metrics     = compute_form_metrics(
            post_comparisons, post_hitl_stats, extraction_time
        )

        print(
            f"  {form_id}: pre={pre_metrics['field_accuracy']:.0%} "
            f"post={post_metrics['field_accuracy']:.0%} "
            f"flagged={pre_hitl_stats.get('flagged_count', 0)}"
        )

        return {
            "form_id":           form_id,
            "template_id":       template_id,
            "pipeline":          pipeline.name,
            "pre_hitl_metrics":  pre_metrics,
            "post_hitl_metrics": post_metrics,
            "pipeline_stats":    pipeline_stats,
        }

    # ── Comparison builder ─────────────────────────────────────────────────────

    @staticmethod
    def _build_comparisons(
        validated_fields: list[dict],
        gt_fields:        dict,
    ) -> list[dict]:
        comparisons = []
        for f in validated_fields:
            fid    = f["field_id"]
            ftype  = f["field_type"]
            value  = f.get("final_value", f.get("value", ""))
            gt_val = gt_fields.get(fid)

            if gt_val is None:
                continue   # Field not in ground truth — skip

            result = compare_field(value, gt_val, ftype)
            result["field_id"]       = fid
            result["field_type"]     = ftype
            result["extracted_value"] = value
            result["ground_truth"]   = gt_val
            result["extracted_bool"] = bool(value) if ftype == "checkbox" else None
            result["needs_review"]   = f.get("needs_review", False)
            result["corrected"]      = f.get("corrected", False)
            comparisons.append(result)

        return comparisons

    # ── Results writer ─────────────────────────────────────────────────────────

    def _write_results(self, all_results: dict) -> None:
        import csv

        # Full JSON dump
        json_path = self._results_dir / "full_results.json"
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2, default=str, ensure_ascii=False)

        # Summary CSV — one row per pipeline × condition
        csv_path = self._results_dir / "summary_table.csv"
        rows = []
        for pipeline_name, data in all_results.items():
            for condition, agg_key in [("No HITL", "pre_hitl_aggregate"),
                                        ("With HITL", "post_hitl_aggregate")]:
                agg = data.get(agg_key, {})
                rows.append({
                    "pipeline":          pipeline_name,
                    "condition":         condition,
                    "n_forms":           agg.get("n_forms", 0),
                    "field_accuracy":    agg.get("overall_field_accuracy", 0),
                    "text_accuracy":     agg.get("mean_text_field_accuracy", 0),
                    "checkbox_f1":       agg.get("mean_checkbox_f1", 0),
                    "escalation_rate":   agg.get("mean_escalation_rate", 0),
                    "avg_time_s":        agg.get("mean_processing_time_s", 0),
                })

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            if rows:
                writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        # HITL impact CSV
        impact_path = self._results_dir / "hitl_impact.csv"
        with impact_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "pipeline", "pre_hitl_accuracy", "post_hitl_accuracy",
                "hitl_accuracy_gain", "hitl_gain_pct"
            ])
            writer.writeheader()
            for pipeline_name, data in all_results.items():
                row = {"pipeline": pipeline_name, **data.get("hitl_impact", {})}
                writer.writerow(row)

        print(f"\n  Results written to: {self._results_dir}")
        print(f"    {json_path.name}")
        print(f"    {csv_path.name}")
        print(f"    {impact_path.name}")
