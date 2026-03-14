"""
run_evaluation.py
==================
Entry point for the full six-condition comparative evaluation.

Usage
-----
    # With DeepSeek (all 3 pipelines):
    python run_evaluation.py --deepseek-key sk-xxxx

    # Without DeepSeek (Tesseract + DAPE only):
    python run_evaluation.py

    # Custom paths:
    python run_evaluation.py \\
        --forms-dir    form/ \\
        --gt-dir       ground_truth/ \\
        --template-id  template_001 \\
        --threshold    0.60

Ground Truth Setup
------------------
For each scanned form (e.g. form/678324.tif), create a file:
    ground_truth/678324.json
with the following structure:

    {
      "form_id":     "678324",
      "template_id": "template_001",
      "fields": {
        "name":       "Jane Smith",
        "dob":        "12/03/1995",
        "consent":    true
      }
    }

Field IDs must match those defined in templates/registry.json.
"""

import argparse
import glob
from pathlib import Path

from evaluation.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(
        description="DAPE Comparative Evaluation — 6-condition pipeline comparison"
    )
    parser.add_argument(
        "--forms-dir",    default="form",
        help="Directory containing scanned form images (default: form/)"
    )
    parser.add_argument(
        "--gt-dir",       default="ground_truth",
        help="Directory containing ground truth JSON files (default: ground_truth/)"
    )
    parser.add_argument(
        "--results-dir",  default="evaluation/results",
        help="Output directory for result tables (default: evaluation/results/)"
    )
    parser.add_argument(
        "--registry",     default="templates/registry.json",
        help="Template registry path (default: templates/registry.json)"
    )
    parser.add_argument(
        "--template-id",  default="template_001",
        help="Template ID to evaluate against (default: template_001)"
    )
    parser.add_argument(
        "--threshold",    type=float, default=0.60,
        help="Confidence threshold for HITL escalation (default: 0.60)"
    )
    parser.add_argument(
        "--got-cache-dir", default="models/got_ocr2",
        help="Directory to cache GOT-OCR 2.0 model weights (default: models/got_ocr2)"
    )
    parser.add_argument(
        "--got-device",    default="cpu",
        help="Device for GOT-OCR inference: cpu or cuda (default: cpu)"
    )
    parser.add_argument(
        "--hitl-port",    type=int, default=5050,
        help="Port for the HITL review UI (default: 5050)"
    )
    parser.add_argument(
        "--tesseract-cmd", default=None,
        help="Explicit path to Tesseract binary (optional)"
    )
    parser.add_argument(
        "--extensions",   default="tif,tiff,png,jpg,jpeg",
        help="Comma-separated image extensions to scan (default: tif,tiff,png,jpg,jpeg)"
    )
    args = parser.parse_args()

    # Collect form image paths
    exts  = [e.strip().lstrip(".") for e in args.extensions.split(",")]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(f"{args.forms_dir}/*.{ext}"))
        paths.extend(glob.glob(f"{args.forms_dir}/*.{ext.upper()}"))
    paths = sorted(set(paths))

    if not paths:
        print(f"[ERROR] No form images found in '{args.forms_dir}'")
        print(f"        Looked for extensions: {exts}")
        return

    print(f"\n{'='*60}")
    print(f"  DAPE Comparative Evaluation")
    print(f"{'='*60}")
    print(f"  Forms found  : {len(paths)}")
    print(f"  Template     : {args.template_id}")
    print(f"  Threshold    : {args.threshold}")
    print(f"  GOT-OCR dir  : {args.got_cache_dir}")
    print(f"  GOT device   : {args.got_device}")
    print(f"  Results dir  : {args.results_dir}")
    print(f"{'='*60}\n")

    # Build form list (form_id inferred from filename stem)
    forms = [
        {
            "form_id":     Path(p).stem,
            "template_id": args.template_id,
            "image_path":  p,
        }
        for p in paths
    ]

    # Run evaluation
    evaluator = Evaluator(
        registry_path        = args.registry,
        ground_truth_dir     = args.gt_dir,
        results_dir          = args.results_dir,
        confidence_threshold = args.threshold,
        got_ocr_cache_dir    = args.got_cache_dir,
        got_ocr_device       = args.got_device,
        hitl_port            = args.hitl_port,
        tesseract_cmd        = args.tesseract_cmd,
    )

    results = evaluator.run(forms)

    # Print final summary table
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Pipeline':<28} {'No HITL':>9} {'With HITL':>10} {'Gain':>8}")
    print(f"  {'-'*56}")
    for name, data in results.items():
        pre  = data["pre_hitl_aggregate"].get("overall_field_accuracy", 0)
        post = data["post_hitl_aggregate"].get("overall_field_accuracy", 0)
        gain = post - pre
        print(f"  {name:<28} {pre:>8.1%} {post:>10.1%} {gain:>+7.1%}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
