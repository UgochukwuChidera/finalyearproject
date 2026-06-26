"""Command-line interface for DAPE form processing."""
import argparse
import json

from pipeline import process_form


def parse_args():
    p = argparse.ArgumentParser(description="DAPE form processing")
    p.add_argument("--image", required=True, help="Path to filled form image")
    p.add_argument("--config-name", default=None, help="Config name in configs/<name>.json")
    p.add_argument("--config-path", default=None, help="Absolute/relative config path")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--dictionaries-dir", default="dictionaries")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    result = process_form(
        image_path=args.image,
        config_name=args.config_name,
        config_path=args.config_path,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        dictionaries_dir=args.dictionaries_dir,
        dpi=args.dpi,
    )
    print(json.dumps({"job_id": result["job_id"], "status": result["status"], "audit": result["audit_log_path"]}, indent=2))


if __name__ == "__main__":
    main()
