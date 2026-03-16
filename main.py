"""
DAPE Pipeline — Entry Point
============================
Usage
-----
  # Register a template (run once per form type):
  python main.py --register --template-id student_academic_record \
                 --template-image templates/student_academic_record_blank.tif

  # Process one form (default: Tesseract extractor):
  python main.py --image form/student_academic_record_01.tif \
                 --template-id student_academic_record --dpi 600

  # Use PaddleOCR for field extraction:
  python main.py --image form/student_academic_record_01.tif \
                 --template-id student_academic_record --extractor paddle

  # Use DeepSeek Document AI:
  python main.py --image form/student_academic_record_01.tif \
                 --template-id student_academic_record --extractor deepseek \
                 --deepseek-key sk-...

  # Batch process all forms of a type:
  python main.py --batch --forms-dir form/ \
                 --template-id student_academic_record --dpi 600

  # Skip HITL (evaluate only):
  python main.py --image form/student_academic_record_01.tif \
                 --template-id student_academic_record --dpi 600 --no-hitl

Extractor backends
------------------
  tesseract   Classical rule-based OCR via Tesseract (default; no extra deps)
  paddle      Deep-learning OCR via PaddleOCR v3 (requires paddlepaddle + paddleocr)
  deepseek    Cloud Document AI via DeepSeek Vision API (requires API key + openai)

The active extractor can also be set via the OCR_EXTRACTOR environment variable:
  export OCR_EXTRACTOR=paddle
"""

import argparse
import glob
import os
from pathlib import Path

from project.orchestrator import DAPEOrchestrator
from project.template_analyzer.registry_builder import TemplateRegistryBuilder


def parse_args():
    p = argparse.ArgumentParser(
        description="DAPE Form Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    mode = p.add_argument_group("Mode")
    mode.add_argument("--register", action="store_true",
                      help="Register a blank template before processing")
    mode.add_argument("--batch",    action="store_true",
                      help="Process all forms in --forms-dir")

    # ── Input ─────────────────────────────────────────────────────────────────
    inp = p.add_argument_group("Input")
    inp.add_argument("--image",          default=None,
                     help="Path to scanned form image (single-form mode)")
    inp.add_argument("--forms-dir",      default="form",
                     help="Directory of forms for batch mode (default: form/)")
    inp.add_argument("--template-id",    default="student_academic_record",
                     help="Template ID to use (must exist in registry.json)")
    inp.add_argument("--template-image", default=None,
                     help="Blank template image path (--register mode only)")

    # ── Extractor selection ───────────────────────────────────────────────────
    ext = p.add_argument_group(
        "Extractor",
        "Choose the OCR / extraction backend.  The OCR_EXTRACTOR environment\n"
        "variable sets the default when --extractor is not supplied.",
    )
    ext.add_argument(
        "--extractor",
        default=None,
        choices=["tesseract", "paddle", "deepseek"],
        metavar="BACKEND",
        help=(
            "OCR backend to use.  Choices: tesseract (default), paddle, deepseek.  "
            "Falls back to OCR_EXTRACTOR env var, then 'tesseract'."
        ),
    )
    ext.add_argument(
        "--deepseek-key",
        default=None,
        metavar="KEY",
        help=(
            "DeepSeek API key (required when --extractor deepseek).  "
            "Can also be set via DEEPSEEK_API_KEY environment variable."
        ),
    )
    ext.add_argument(
        "--deepseek-model",
        default=None,
        metavar="MODEL",
        help="DeepSeek model name (default: deepseek-chat; override via DEEPSEEK_MODEL env var)",
    )
    ext.add_argument(
        "--paddle-lang",
        default="en",
        metavar="LANG",
        help="PaddleOCR language code (default: en)",
    )
    ext.add_argument(
        "--paddle-gpu",
        action="store_true",
        help="Enable GPU for PaddleOCR inference (default: CPU-only)",
    )

    # ── Pipeline settings ─────────────────────────────────────────────────────
    pipe = p.add_argument_group("Pipeline settings")
    pipe.add_argument("--dpi",       type=int,   default=300,
                      help="Actual scan DPI — scales all kernels (default: 300)")
    pipe.add_argument("--threshold", type=float, default=0.60,
                      help="Confidence threshold for HITL escalation (default: 0.60)")
    pipe.add_argument("--no-hitl",   action="store_true",
                      help="Disable HITL review (auto-accept all extractions)")
    pipe.add_argument("--hitl-port", type=int,   default=5050,
                      help="Port for HITL Flask review UI (default: 5050)")

    # ── Paths ─────────────────────────────────────────────────────────────────
    paths = p.add_argument_group("Paths")
    paths.add_argument("--registry",      default="templates/registry.json")
    paths.add_argument("--output-dir",    default="outputs")
    paths.add_argument("--log-dir",       default="logs")
    paths.add_argument("--tesseract-cmd", default=None,
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

    # ── Resolve extractor and kwargs ───────────────────────────────────────────
    extractor_name = args.extractor or os.environ.get("OCR_EXTRACTOR", "tesseract")

    extractor_kwargs: dict = {}
    if extractor_name == "deepseek":
        # API key: --deepseek-key flag > DEEPSEEK_API_KEY env var
        api_key = args.deepseek_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            print(
                "[ERROR] DeepSeek extractor requires an API key.\n"
                "  Option 1:  --deepseek-key sk-...\n"
                "  Option 2:  export DEEPSEEK_API_KEY=sk-..."
            )
            return
        extractor_kwargs["api_key"] = api_key
        if args.deepseek_model:
            extractor_kwargs["model"] = args.deepseek_model
    elif extractor_name in ("paddle", "paddleocr"):
        extractor_kwargs["lang"]    = args.paddle_lang
        extractor_kwargs["use_gpu"] = args.paddle_gpu

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
        extractor            = extractor_name,
        extractor_kwargs     = extractor_kwargs,
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
              f"(template: {args.template_id}, DPI: {args.dpi}, "
              f"extractor: {extractor_name})\n")
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
    print(f"Extractor  : {extractor_name}")
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
