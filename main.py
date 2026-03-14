"""
DAPE Pipeline — Entry Point
============================
Usage
-----
  # Register a template (run once per form type):
  python main.py --register --template-id student_academic_record \
                 --template-image templates/student_academic_record_blank.tif

  # Process one form:
  python main.py --image form/student_academic_record_01.tif \
                 --template-id student_academic_record --dpi 600

  # Batch process all forms of a type:
  python main.py --batch --forms-dir form/ \
                 --template-id student_academic_record --dpi 600

  # Skip HITL (evaluate only):
  python main.py --image form/student_academic_record_01.tif \
                 --template-id student_academic_record --dpi 600 --no-hitl
"""

import argparse
import glob
from pathlib import Path

from project.orchestrator import DAPEOrchestrator
from project.template_analyzer.registry_builder import TemplateRegistryBuilder


def parse_args():
    p = argparse.ArgumentParser(description="DAPE Form Processing Pipeline")

    # Mode
    p.add_argument("--register", action="store_true",
                   help="Register a blank template before processing")
    p.add_argument("--batch",    action="store_true",
                   help="Process all forms in --forms-dir")

    # Input
    p.add_argument("--image",         default=None,
                   help="Path to scanned form image (single-form mode)")
    p.add_argument("--forms-dir",     default="form",
                   help="Directory of forms for batch mode (default: form/)")
    p.add_argument("--template-id",   default="student_academic_record",
                   help="Template ID to use (must exist in registry.json)")
    p.add_argument("--template-image",default=None,
                   help="Blank template image path (--register mode only)")

    # Pipeline settings
    p.add_argument("--dpi",           type=int,   default=300,
                   help="Actual scan DPI — scales all kernels (default: 300)")
    p.add_argument("--threshold",     type=float, default=0.60,
                   help="Confidence threshold for HITL escalation (default: 0.60)")
    p.add_argument("--no-hitl",       action="store_true",
                   help="Disable HITL review (auto-accept all extractions)")
    p.add_argument("--hitl-port",     type=int,   default=5050,
                   help="Port for HITL Flask review UI (default: 5050)")

    # Paths
    p.add_argument("--registry",      default="templates/registry.json")
    p.add_argument("--output-dir",    default="outputs")
    p.add_argument("--log-dir",       default="logs")
    p.add_argument("--tesseract-cmd", default=None,
                   help="Explicit path to Tesseract binary")

    return p.parse_args()


def main():
    args = parse_args()

    # ── Register mode ──────────────────────────────────────────────────────────
    if args.register:
        if not args.template_image:
            print("[ERROR] --template-image required in --register mode")
            return
        builder = TemplateRegistryBuilder()
        builder.build(
            template_id   = args.template_id,
            image_path    = args.template_image,
            registry_path = args.registry,
        )
        print(f"\nTemplate '{args.template_id}' registered.")
        print("Review templates/registry.json to adjust field coordinates if needed.")
        print(f"Then run:  python main.py --image <form.tif> "
              f"--template-id {args.template_id} --dpi {args.dpi}\n")
        return

    # ── Build orchestrator ─────────────────────────────────────────────────────
    orchestrator = DAPEOrchestrator(
        registry_path        = args.registry,
        output_dir           = args.output_dir,
        log_dir              = args.log_dir,
        confidence_threshold = args.threshold,
        enable_hitl          = not args.no_hitl,
        hitl_host            = "127.0.0.1",
        hitl_port            = args.hitl_port,
        tesseract_cmd        = args.tesseract_cmd,
        dpi                  = args.dpi,
    )

    # ── Batch mode ─────────────────────────────────────────────────────────────
    if args.batch:
        exts  = ["tif", "tiff", "jpg", "jpeg", "png"]
        paths = []
        for ext in exts:
            paths += glob.glob(f"{args.forms_dir}/*.{ext}")
            paths += glob.glob(f"{args.forms_dir}/*.{ext.upper()}")
        paths = sorted(set(paths))

        if not paths:
            print(f"[ERROR] No images found in '{args.forms_dir}'")
            return

        print(f"\nBatch processing {len(paths)} forms "
              f"(template: {args.template_id}, DPI: {args.dpi})\n")
        results = orchestrator.process_batch(paths, args.template_id)

        print(f"\n{'='*55}")
        print(f"  BATCH COMPLETE")
        print(f"{'='*55}")
        ok  = [r for r in results if "error" not in r]
        err = [r for r in results if "error" in r]
        print(f"  Processed : {len(ok)}")
        print(f"  Errors    : {len(err)}")
        for r in err:
            print(f"  ERROR {r.get('form_id','?')}: {r['error']}")
        return

    # ── Single-form mode ───────────────────────────────────────────────────────
    if not args.image:
        print("[ERROR] Provide --image <path> or use --batch / --register")
        return

    print(f"\nProcessing : {args.image}")
    print(f"Template   : {args.template_id}")
    print(f"DPI        : {args.dpi}")
    print(f"HITL       : {'disabled' if args.no_hitl else f'enabled (port {args.hitl_port})'}\n")

    result = orchestrator.process(
        image_path  = args.image,
        template_id = args.template_id,
    )

    print(f"\n{'='*55}")
    print(f"  RESULT")
    print(f"{'='*55}")
    data = result["structured_output"]["data"]
    for k, v in data.items():
        print(f"  {k:<28} {v}")
    print(f"\n  Exports   : {result['export_paths']}")
    print(f"  Audit log : {result['audit_log_path']}")
    esc = result["stats"]
    print(f"  Flagged   : {esc.get('esc_flagged_count',0)}")
    print(f"  Corrected : {esc.get('esc_corrected_count',0)}")
    print()


if __name__ == "__main__":
    main()
