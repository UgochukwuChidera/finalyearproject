#!/usr/bin/env python3
"""
test_deepseek_ocr.py  —  Standalone DeepSeek OCR test script
=============================================================
Run this script to send a scanned form image to the DeepSeek Vision API
and inspect every layer of the returned data.

REQUIREMENTS
------------
1. A funded DeepSeek API account and key (https://platform.deepseek.com)
2. Python packages:
       pip install openai pillow
3. The API key set in one of these ways (in order of precedence):
   a) CLI flag:           python test_deepseek_ocr.py --key sk-xxxxx
   b) Environment var:   export DEEPSEEK_API_KEY=sk-xxxxx

USAGE
-----
  # Minimal — freeform OCR, no template:
  python test_deepseek_ocr.py --image form/student_academic_record_01.tif

  # Template-aware — tells DeepSeek exactly which fields to find:
  python test_deepseek_ocr.py \\
      --image       form/student_academic_record_01.tif \\
      --template-id student_academic_record

  # Override model (e.g. deepseek-reasoner for higher accuracy):
  python test_deepseek_ocr.py \\
      --image form/student_academic_record_01.tif \\
      --model deepseek-chat

  # Save the full raw API response to a JSON file for later inspection:
  python test_deepseek_ocr.py \\
      --image form/student_academic_record_01.tif \\
      --save-raw output_raw.json

OUTPUT SECTIONS
---------------
The script prints six sections:
  [1] Configuration     — model, endpoint, template
  [2] API call          — encoding progress and success
  [3] Raw API response  — full JSON from the DeepSeek API (usage, finish reason, …)
  [4] Parsed fields     — the {field_id: value} mapping extracted from the response
  [5] Concatenated text — all non-blank values joined as a single string
  [6] Summary stats     — field count, blank / checkbox / text breakdowns, token usage

NOTES
-----
• deepseek-chat            supports vision (images embedded as base-64).
• deepseek-reasoner        supports vision too, at higher cost per token.
• A typical 600 DPI form image uses ~1 500–3 000 input tokens.
• Charges apply per API call. Check https://platform.deepseek.com/pricing.
• The DEEPSEEK_API_BASE env-var lets you point at a compatible local endpoint
  (e.g. Ollama, LM Studio) for offline testing without any charges.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path


# ── Pretty-print helpers ──────────────────────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_RED    = "\033[31m"

def _h(title: str) -> str:
    bar = "─" * (60 - len(title) - 3)
    return f"\n{_BOLD}{_CYAN}── {title} {bar}{_RESET}"

def _ok(msg: str)   -> str: return f"  {_GREEN}✔{_RESET}  {msg}"
def _warn(msg: str) -> str: return f"  {_YELLOW}⚠{_RESET}  {msg}"
def _err(msg: str)  -> str: return f"  {_RED}✘{_RESET}  {msg}"


# ── Argument parsing ─────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="test_deepseek_ocr.py",
        description=(
            "Send a form image to the DeepSeek Vision API and print the full "
            "extraction output so you can see exactly what the model returns."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples
            --------
              python test_deepseek_ocr.py --image form/student_academic_record_01.tif
              python test_deepseek_ocr.py --image form/medical_screening_01.tif \\
                  --template-id medical_screening --key sk-xxxx
        """),
    )

    p.add_argument(
        "--image", "-i", required=True,
        metavar="PATH",
        help="Path to the scanned form image (TIF, PNG, JPG, …).",
    )
    p.add_argument(
        "--template-id", "-t", default=None,
        metavar="ID",
        help=(
            "Template ID defined in templates/registry.json "
            "(e.g. student_academic_record).  "
            "When supplied, the model is told exactly which fields to extract.  "
            "When omitted, the model extracts all visible key-value pairs."
        ),
    )
    p.add_argument(
        "--key", "-k", default=None,
        metavar="SK-…",
        help=(
            "DeepSeek API key.  "
            "Can also be set via the DEEPSEEK_API_KEY environment variable."
        ),
    )
    p.add_argument(
        "--model", "-m",
        default=None,
        metavar="NAME",
        help=(
            "DeepSeek model to use (default: deepseek-chat).  "
            "Can also be set via the DEEPSEEK_MODEL environment variable.  "
            "Other options: deepseek-reasoner"
        ),
    )
    p.add_argument(
        "--api-base",
        default=None,
        metavar="URL",
        help=(
            "Override the API base URL "
            "(default: https://api.deepseek.com/v1).  "
            "Useful for pointing at a local compatible endpoint.  "
            "Can also be set via DEEPSEEK_API_BASE."
        ),
    )
    p.add_argument(
        "--registry", default="templates/registry.json",
        metavar="PATH",
        help="Path to templates/registry.json (default: templates/registry.json).",
    )
    p.add_argument(
        "--save-raw", default=None,
        metavar="FILE",
        help="If provided, save the raw API response JSON to this file.",
    )
    p.add_argument(
        "--no-colour", action="store_true",
        help="Disable ANSI colour codes in output.",
    )
    return p.parse_args()


# ── Main logic ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # Disable colour if requested or piped output
    global _RESET, _BOLD, _GREEN, _YELLOW, _CYAN, _RED
    if args.no_colour or not sys.stdout.isatty():
        _RESET = _BOLD = _GREEN = _YELLOW = _CYAN = _RED = ""

    # ── Resolve configuration ─────────────────────────────────────────────
    api_key  = args.key      or os.environ.get("DEEPSEEK_API_KEY", "")
    api_base = args.api_base or os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    model    = args.model    or os.environ.get("DEEPSEEK_MODEL",    "deepseek-chat")

    image_path = Path(args.image)

    print(_h("1. Configuration"))
    print(f"  Image        : {image_path}")
    print(f"  Template ID  : {args.template_id or '(none — freeform extraction)'}")
    print(f"  Model        : {model}")
    print(f"  API base     : {api_base}")
    print(f"  Registry     : {args.registry}")

    # ── Pre-flight checks ─────────────────────────────────────────────────
    errors = []

    if not image_path.exists():
        errors.append(f"Image file not found: {image_path}")

    if not api_key:
        errors.append(
            "No API key found.\n"
            "    Fix: export DEEPSEEK_API_KEY=sk-…\n"
            "      or: python test_deepseek_ocr.py --key sk-…\n"
            "    Get a key at: https://platform.deepseek.com"
        )

    try:
        import openai  # noqa: F401
    except ImportError:
        errors.append(
            "The 'openai' package is not installed.\n"
            "    Fix: pip install openai"
        )

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        errors.append(
            "The 'Pillow' package is not installed (needed for image encoding).\n"
            "    Fix: pip install pillow"
        )

    if errors:
        for e in errors:
            print(_err(e))
        print()
        sys.exit(1)

    print(_ok(f"Image exists ({image_path.stat().st_size // 1024} KB)"))
    print(_ok("openai SDK found"))
    print(_ok("Pillow found"))
    print(_ok("API key configured"))

    # ── Import extractor (after pre-flight so errors are clean) ───────────
    # Add the repo root to sys.path so the project package is importable
    repo_root = Path(__file__).parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from project.extraction.deepseek_extractor import DeepSeekExtractor

    extractor = DeepSeekExtractor(
        api_key       = api_key,
        api_base      = api_base,
        model         = model,
        registry_path = args.registry,
    )

    # ── Run extraction ────────────────────────────────────────────────────
    print(_h("2. API Call"))
    print(f"  Encoding image and sending to {api_base} …")

    try:
        result = extractor.extract(str(image_path), template_id=args.template_id)
    except Exception as exc:
        print(_err(f"Extraction failed: {exc}"))
        raise SystemExit(1) from exc

    print(_ok("API call succeeded"))

    # ── Raw API response ──────────────────────────────────────────────────
    print(_h("3. Raw API Response"))

    raw = result.raw_ocr
    if raw is not None:
        # Serialise the response object to a plain dict so we can print it nicely
        try:
            raw_dict = raw.model_dump()       # openai pydantic v2
        except AttributeError:
            try:
                raw_dict = dict(raw)          # legacy openai v1
            except TypeError:
                raw_dict = {"raw": str(raw)}  # fallback

        raw_json = json.dumps(raw_dict, indent=2, default=str)
        print(raw_json)

        if args.save_raw:
            out_path = Path(args.save_raw)
            out_path.write_text(raw_json)
            print(_ok(f"Raw response saved to {out_path}"))
    else:
        print("  (no raw response captured)")

    # ── Parsed fields ─────────────────────────────────────────────────────
    print(_h("4. Extracted Fields"))

    if result.fields:
        # Column widths
        max_key = max(len(k) for k in result.fields)
        for fid, value in result.fields.items():
            tag = ""
            if isinstance(value, bool):
                tag  = f"{_YELLOW}[checkbox]{_RESET}"
                disp = "☑ checked" if value else "☐ unchecked"
            elif value == "" or value is None:
                tag  = f"{_RED}[blank]{_RESET}"
                disp = "(empty)"
            else:
                disp = str(value)
            padded = fid.ljust(max_key)
            print(f"  {_BOLD}{padded}{_RESET}  {disp}  {tag}")
    else:
        print(_warn("No fields were extracted."))

    # ── Full text ─────────────────────────────────────────────────────────
    if result.text:
        print(_h("5. Concatenated Text"))
        # Wrap long text at 70 chars
        wrapped = textwrap.fill(result.text, width=70, initial_indent="  ", subsequent_indent="  ")
        print(wrapped)

    # ── Summary stats ─────────────────────────────────────────────────────
    print(_h("6. Summary"))

    total     = len(result.fields)
    checkboxes = sum(1 for v in result.fields.values() if isinstance(v, bool))
    blanks     = sum(1 for v in result.fields.values() if v == "" or v is None)
    filled     = total - blanks

    print(f"  Total fields      : {total}")
    print(f"  Text fields filled: {filled - checkboxes}")
    print(f"  Checkbox fields   : {checkboxes}")
    print(f"  Blank / missing   : {blanks}")
    if total > 0:
        pct = 100 * filled / total
        bar_filled = int(pct / 5)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        print(f"  Fill rate         : {pct:5.1f}%  [{bar}]")

    # Token usage (if available)
    try:
        usage = raw.usage
        if usage:
            print(f"\n  Token usage:")
            print(f"    Prompt tokens     : {usage.prompt_tokens}")
            print(f"    Completion tokens : {usage.completion_tokens}")
            print(f"    Total tokens      : {usage.total_tokens}")
    except AttributeError:
        pass

    print()


if __name__ == "__main__":
    main()
