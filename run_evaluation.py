import argparse
import glob
from pathlib import Path

from evaluation.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="DAPE evaluation runner")
    parser.add_argument("--forms-dir", default="form")
    parser.add_argument("--results-dir", default="evaluation/results")
    parser.add_argument("--ground-truth-dir", default="ground_truth")
    parser.add_argument("--template-id", default="medical_screening_v1")
    parser.add_argument("--config-name", default=None,
                        help="Config name used by DAPEPipeline (defaults to --template-id)")
    parser.add_argument("--extensions", default="tif,tiff,png,jpg,jpeg")
    args = parser.parse_args()

    config_name = args.config_name or args.template_id

    exts = [e.strip().lstrip(".") for e in args.extensions.split(",")]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(f"{args.forms_dir}/*.{ext}"))
        paths.extend(glob.glob(f"{args.forms_dir}/*.{ext.upper()}"))
    paths = sorted(set(paths))

    if not paths:
        print(f"[ERROR] No form images found in '{args.forms_dir}'")
        return

    forms = [{"form_id": Path(p).stem, "template_id": args.template_id, "image_path": p} for p in paths]

    evaluator = Evaluator(
        results_dir=args.results_dir,
        ground_truth_dir=args.ground_truth_dir,
        config_name=config_name,
    )

    results = evaluator.run(forms)
    for pipeline_name, data in results.items():
        impact = data.get("hitl_impact", {})
        agg = data.get("pre_hitl_aggregate", {})
        print(
            f"[{pipeline_name}]  "
            f"pre-HITL accuracy={impact.get('pre_hitl_accuracy', 0):.3f}  "
            f"post-HITL accuracy={impact.get('post_hitl_accuracy', 0):.3f}  "
            f"gain={impact.get('hitl_gain_pct', 0):.1f}%  "
            f"forms={agg.get('n_forms', 0)}"
        )
    print(f"Finished evaluation for {len(results)} pipeline(s). Results: {args.results_dir}/full_results.json")


if __name__ == "__main__":
    main()
