import argparse
import glob
from pathlib import Path

from evaluation.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="DAPE evaluation runner")
    parser.add_argument("--forms-dir", default="form")
    parser.add_argument("--results-dir", default="evaluation/results")
    parser.add_argument("--template-id", default="medical_screening_v1")
    parser.add_argument("--extensions", default="tif,tiff,png,jpg,jpeg")
    args = parser.parse_args()

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

    evaluator = Evaluator(results_dir=args.results_dir)
    results = evaluator.run(forms)
    print(f"Finished evaluation for {len(results)} pipeline(s)")


if __name__ == "__main__":
    main()
