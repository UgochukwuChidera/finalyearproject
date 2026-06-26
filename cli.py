"""Command-line interface for DAPE form processing."""
from __future__ import annotations

import argparse
import json
from typing import Any

from pipeline import process_form


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAPE form processing")
    parser.add_argument("--image", required=True, help="Path to filled form image")
    parser.add_argument("--config-name", default=None, help="Config name in configs/<name>.json")
    parser.add_argument("--config-path", default=None, help="Absolute/relative config path")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--dictionaries-dir", default="dictionaries")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
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
