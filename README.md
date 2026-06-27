# DAPE — Differential + AI Form Extraction Pipeline

**DAPE** (Differential and AI-Powered Extraction) is a hybrid computer-vision and LLM pipeline for extracting handwritten and printed fields from scanned forms. It combines classical image processing (deskewing, binarization, template alignment, differential analysis) with vision-language model inference (via OpenRouter) to achieve high-accuracy field extraction with human-in-the-loop review.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Web App](#web-app)
  - [CLI Pipeline](#cli-pipeline)
  - [Batch Processing](#batch-processing)
- [Configuration](#configuration)
  - [Config Schema](#config-schema)
  - [Adding a New Form Type](#adding-a-new-form-type)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [API Routes](#api-routes)
- [Hybrid CV + AI Strategy](#hybrid-cv--ai-strategy)
- [Dictionary-Backed Validation](#dictionary-backed-validation)
- [Outputs](#outputs)

---

## Features

- **Hybrid extraction strategy** — critical fields use differential ink-only crops for focused AI analysis; non-critical fields use full-form AI extraction with template JSON diffing
- **Multi-image prompting** — single AI request per form sends both the full form and differential crops in one prompt
- **8-stage orchestrated pipeline** — preprocessing → alignment → differential analysis → extraction → confidence validation → HITL escalation → output structuring → export
- **Template alignment** — ORB feature matching + RANSAC to register filled forms against blanks
- **Checkbox detection** — automated tick/cross detection via contour analysis
- **Confidence scoring** — three-tier confidence (logprobs-based `C_lp`, dictionary match `C_dict`, composite `C_final`)
- **Human-in-the-loop (HITL)** — web-based review interface for low-confidence fields
- **Batch processing** — parallel form processing with configurable concurrency and rate limiting
- **Multiple export formats** — JSON, CSV, XLSX (relational), plus audit JSONL
- **Evaluation framework** — ground-truth comparison with accuracy, precision/recall/F1, and HITL impact metrics
- **Configurable AI models** — switch models via web UI (`/settings`) or `models.json`; default is `openai/gpt-4o-mini`

---

## Architecture

The pipeline is orchestrated by `DAPEOrchestrator` in `project/orchestrator.py` and runs 8 stages:

| Stage | Module | Description |
|-------|--------|-------------|
| 1. Preprocessing | `project/preprocessing/` | DPI-aware kernel scaling, load → grayscale → baseline metrics → skew analysis → illumination normalization → binarization → border removal → structure prep → fusion scoring |
| 2. Template Alignment | `project/alignment/aligner.py` | ORB feature detection + RANSAC homography to align filled form to blank template |
| 3. Differential Analysis | `project/differential/analyzer.py` | Pixel-difference mask between aligned filled form and blank template to isolate handwritten ink |
| 4. Field Extraction | `project/extraction/` | Crop fields by bounding box from either differential mask (critical fields) or full aligned image; checkbox contour detection |
| 5. Confidence Validation | `project/validation/` | Three-tier confidence: logprobs-based `C_lp`, dictionary match `C_dict`, composite `C_final`; field-level pass/flag/review |
| 6. HITL Escalation | `project/hitl/` | Flag low-confidence fields for human review via web interface |
| 7. Output Structuring | `project/output/` | Structure extracted fields into the defined output schema |
| 8. Export + Audit | `project/output/` | Export to JSON/CSV/XLSX, append to audit JSONL |

A parallel fast-path exists via `pipeline.process_form()` (used by CLI and web uploads) which skips the full orchestrator and runs a streamlined version with AI-based extraction.

### Hybrid CV + AI Strategy

The system uses **two extraction paths** depending on field criticality:

1. **Critical fields** (`"critical": true`) — The differential analyzer computes an ink-only crop (the pixel difference between the aligned filled form and the blank template, constrained to the field's bounding box). This crop is sent to the AI model, eliminating background noise and pre-printed text.

2. **Non-critical fields** — The full aligned form image is sent for AI extraction, and the result is diffed against a pre-computed blank template extraction. Only values that differ from the template are kept.

Both paths use a **single multi-image AI request** per form, constructed by `ai_extraction/prompt_builder.py`. The AI client (`GeminiClient` — named historically, uses OpenRouter under the hood) sends multiple image crops in one prompt with structured JSON output instructions.

---

## Project Structure

```
.
├── ai_extraction/               # AI/LLM extraction layer
│   ├── gemini_client.py         # OpenRouter API client (retry, rate-limit, logging)
│   ├── prompt_builder.py        # Multi-image prompt construction
│   ├── dictionary_matcher.py    # Dictionary-backed field matching
│   └── confidence.py            # C_lp / C_dict / C_final computation
├── batching.py                  # ProcessPoolExecutor-based parallel processing
├── cli.py                       # Click/argparse CLI entry point (38 lines)
├── configs/
│   └── screen3.json             # Active form configuration (student academic record)
├── dictionaries/                # CSV dictionaries for field validation
├── form/                        # Scanned form images (45 .tif files, 3 types × 15)
├── ground_truth/
│   └── ground_truth_entry.xlsx  # Ground truth data for evaluation (3 sheets)
├── main.py                      # Thin re-export module for backward compat (32 lines)
├── models.json                  # AI model registry and active model selection
├── pipeline.py                  # process_form() orchestrator (431 lines)
├── project/
│   ├── orchestrator.py          # DAPEOrchestrator (8-stage pipeline)
│   ├── template_registry.py     # Template/image registry
│   ├── alignment/
│   │   └── aligner.py           # ORB + RANSAC template alignment
│   ├── differential/
│   │   └── analyzer.py          # Pixel-difference / ink-only mask
│   ├── extraction/
│   │   ├── field_extractor.py   # Field cropping and checkbox detection
│   │   └── checkbox_detector.py
│   ├── validation/
│   │   └── confidence_validator.py
│   ├── hitl/
│   │   ├── escalation.py        # Low-confidence flagging logic
│   │   └── interface.py         # Review interface helper
│   ├── output/
│   │   ├── structurer.py        # Output schema structuring
│   │   ├── exporter.py          # JSON/CSV export
│   │   ├── relational_exporter.py  # XLSX relational export
│   │   └── audit_logger.py      # Audit JSONL writer
│   └── preprocessing/           # 10 modules
│       ├── io.py                # Image loading
│       ├── grayscale.py         # Color → grayscale
│       ├── baseline_metrics.py  # Image statistics
│       ├── skew_analysis.py     # Hough-based skew detection
│       ├── illumination.py      # CLAHE normalization
│       ├── binarization.py      # Adaptive thresholding
│       ├── border_removal.py    # Edge cleanup
│       ├── dpi.py               # DPI-aware kernel sizing
│       ├── fusion.py            # Quality fusion score
│       └── structure_prep.py    # Dimension normalization
├── run.py                       # Flask dev server entry point
├── run_evaluation.py            # Evaluation pipeline runner
├── templates/                   # Blank template images + registry
│   ├── registry.json
│   ├── student_academic_record_blank.tif
│   ├── medical_screening_blank.tif
│   ├── leave_application_blank.tif
│   └── *.png                    # Uploaded/managed templates
├── evaluation/
│   ├── evaluator.py             # Main evaluation harness
│   ├── ground_truth.py          # Ground truth loader
│   ├── metrics.py               # Accuracy, precision/recall/F1
│   └── pipelines/               # Pipeline adapters
├── scripts/
│   └── precompute_template_extraction.py
├── tests/                       # 79 pytest tests
│   ├── test_pipeline.py
│   ├── test_preprocessing.py
│   ├── test_batching.py
│   └── test_confidence.py
├── web/                         # Flask web application
│   ├── __init__.py              # create_app() factory
│   ├── bp.py                    # Shared Blueprint
│   ├── common.py                # Shared state, helpers, model config
│   ├── routes.py                # Route aggregator
│   ├── config_routes.py         # Config CRUD routes
│   ├── job_routes.py            # Job queue and upload routes
│   ├── api_routes.py            # API, settings, utility routes
│   ├── review_routes.py         # HITL review routes
│   ├── templates/               # 11 Jinja2 templates
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── configs.html
│   │   ├── config_editor.html
│   │   ├── jobs.html
│   │   ├── job_detail.html
│   │   ├── review.html
│   │   ├── audits.html
│   │   ├── evaluation.html
│   │   ├── settings.html
│   │   └── help.html
│   └── static/
│       ├── style.css            # B&W minimalist theme
│       └── app.js               # Frontend JS
├── utils/                       # Document/image utilities
├── uploads/                     # Uploaded form images (runtime)
├── outputs/                     # Extraction results, crops, audit logs
└── logs/                        # Processing logs
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) (Windows only — required by `pdf2image`)
- An [OpenRouter](https://openrouter.ai/) API key

### Install

```bash
pip install -r requirements.txt
```

### Configure API Key

Copy `.env.example` to `.env` and add your OpenRouter API key:

```bash
cp .env.example .env
```

Edit `.env`:

```ini
OPENROUTER_API_KEY="sk-or-v1-..."
# Optional: hard-override the active model (bypasses models.json / /settings UI)
# Default: openai/gpt-4o-mini (configurable via /settings UI or OPENROUTER_MODEL env var)
# OPENROUTER_MODEL="openai/gpt-4o-mini"
OPENROUTER_MAX_TOKENS=4096
```

You can also set the key as a system environment variable:

- **Linux/macOS:** `export OPENROUTER_API_KEY="sk-or-v1-..."`
- **Windows (PowerShell):** `$env:OPENROUTER_API_KEY="sk-or-v1-..."`

---

## Usage

### Web App

```bash
python run.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The dashboard provides:
- **Upload** — Select a config and upload form images (TIFF/PNG/JPG)
- **Job queue** — Monitor processing progress with stage-level logging
- **Review** — Correct low-confidence fields via HITL interface
- **Config editor** — Create and edit form configurations
- **Settings** — Change AI models, API keys, batch concurrency, rate limits
- **Audits** — Browse the audit trail (JSONL stream)
- **Evaluation** — View evaluation results

### CLI Pipeline

Process a single form:

```bash
python cli.py --image form/student_academic_record_01.tif --config-name screen3
```

Or using the legacy entry point (same code path):

```bash
python main.py --image form/student_academic_record_01.tif --config-name screen3
```

Additional CLI options:

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | (required) | Path to filled form image |
| `--config-name` | `None` | Config name (looks in `configs/<name>.json`) |
| `--config-path` | `None` | Absolute/relative path to config (alternative to `--config-name`) |
| `--output-dir` | `outputs` | Directory for extraction results |
| `--log-dir` | `logs` | Directory for log files |
| `--dictionaries-dir` | `dictionaries` | Directory for CSV dictionaries |
| `--dpi` | `300` | Scan DPI for kernel scaling |

### Batch Processing

Upload multiple forms in the web UI to process them in parallel. Concurrency is configurable:

- **Max concurrent jobs** — controlled via `/settings` (default: 5)
- **Rate limit** — requests per minute (default: 30), enforced via `threading.Semaphore` + inter-request delay

For programmatic batch processing, use the evaluation pipeline:

```bash
python run_evaluation.py \
  --forms-dir form \
  --config-name screen3 \
  --ground-truth-dir ground_truth
```

### Precompute Blank Template Extraction

If the config references a `template_extraction` file, generate it with:

```bash
python scripts/precompute_template_extraction.py --config configs/screen3.json
```

---

## Configuration

### Config Schema

Form configurations live in `configs/` as JSON files. Example (`configs/screen3.json`):

```json
{
  "template_path": "templates/template_565bd191-1690-40a7-88ee-319ecd1b389a.png",
  "editor_canvas": {
    "width": 800,
    "height": 1100
  },
  "fields": [
    {
      "name": "full_name",
      "type": "text",
      "critical": false,
      "label": "Full Name:",
      "label_hint": "Full Name:",
      "expected_type": "string",
      "bounding_box": { "x": 116, "y": 245, "w": 1004, "h": 28 }
    },
    {
      "name": "matric_number",
      "type": "text",
      "critical": true,
      "label": "Matric Number:",
      "label_hint": "Matric Number:",
      "expected_type": "string",
      "bounding_box": { "x": 116, "y": 285, "w": 1004, "h": 28 }
    },
    {
      "name": "level_100",
      "type": "checkbox",
      "critical": false,
      "label": "Level: [ ] 100",
      "label_hint": "Level: [ ] 100",
      "expected_type": "checkbox",
      "bounding_box": { "x": 66, "y": 416, "w": 132, "h": 28 }
    },
    {
      "name": "course_registration_table",
      "type": "section",
      "critical": false,
      "bounding_box": { "x": 63, "y": 602, "w": 650, "h": 165 }
    }
  ]
}
```

**Field properties:**

| Property | Type | Description |
|----------|------|-------------|
| `name` | string | Unique field identifier |
| `type` | `"text"`, `"checkbox"`, or `"section"` | Field input type |
| `critical` | boolean | If `true`, uses differential ink-only crop for extraction |
| `bounding_box` | `{x, y, w, h}` | Pixel coordinates on the aligned image |
| `label` / `label_hint` | string | Display label in the UI |
| `expected_type` | `"string"` or `"checkbox"` | Expected value type |
| `regex_validation` | string (optional) | Regex pattern for output validation |
| `required` | boolean (optional) | Whether the field must have a value |

### Adding a New Form Type

1. **Place the blank template image** in `templates/` (TIFF or PNG).
2. **Register the template** in `templates/registry.json`:
   ```json
   {
     "my_form": {
       "image_path": "templates/my_form_blank.tif",
       "fields": [
         { "id": "field_1", "type": "handwritten", "x": 100, "y": 200, "w": 500, "h": 100, "required": true }
       ],
       "output_schema": { "field_1": "field_1" }
     }
   }
   ```
3. **Create a config** at `configs/my_form.json` with bounding boxes, critical flags, and field types matching the template layout.
4. **Upload filled forms** via the web UI or process with:
   ```bash
   python cli.py --image path/to/filled_form.tif --config-name my_form
   ```

### AI Model Configuration

Models are configured in `models.json` (managed via `/settings` in the web UI):

```json
{
  "active_model": "openai/gpt-4o-mini",
  "api_key": "",
  "models": [
    { "id": "openai/gpt-4o-mini", "label": "GPT-4o Mini", "vision": true, "logprobs": true },
    { "id": "google/gemini-2.0-flash-001", "label": "Gemini 2.0 Flash", "vision": true, "logprobs": true }
  ]
}
```

The active model can also be overridden via the `OPENROUTER_MODEL` environment variable.

---

## Evaluation

Run the evaluation pipeline against ground truth data:

```bash
python run_evaluation.py \
  --forms-dir form \
  --config-name screen3 \
  --ground-truth-dir ground_truth
```

Results are saved to `evaluation/results/full_results.json` and viewable at `/evaluation` in the web UI.

### Metrics Computed

- **Field-level accuracy** — Normalised and exact-match comparison against ground truth
- **Checkbox metrics** — Precision, recall, and F1 score for checkbox fields
- **Escalation rate** — Percentage of fields flagged for HITL review
- **Pre/post-HITL delta** — Accuracy improvement after reviewer corrections

### Ground Truth Format

Ground truth is stored in `ground_truth/ground_truth_entry.xlsx` with 3 sheets:

- **FORMS** — Per-form field values (form ID, field name, expected value)
- **COURSES** — Course registration table ground truth
- **README** — Documentation of columns and conventions

---

## Testing

79 pytest tests covering pipeline modules, preprocessing, batching, and confidence scoring:

```bash
python -m pytest tests/ -v
```

Test layout:

| File | Scope |
|------|-------|
| `tests/test_pipeline.py` | Config loading, deskewing, cropping, validation, checkbox detection, `process_form` integration |
| `tests/test_preprocessing.py` | Grayscale, binarization, skew, illumination, border removal |
| `tests/test_batching.py` | Parallel worker dispatch, error wrapping |
| `tests/test_confidence.py` | `C_lp`, `C_dict`, `C_final` computation |

---

## API Routes

The Flask web app exposes the following routes (all mounted under the `web` Blueprint):

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Dashboard + upload form |
| `GET` | `/configs` | List all configs |
| `GET` | `/configs/new` | New config form |
| `POST` | `/configs/new` | Create new config |
| `GET` | `/configs/<name>/edit` | Edit config form |
| `POST` | `/configs/<name>/edit` | Update config |
| `POST` | `/configs/<name>/delete` | Delete config |
| `POST` | `/upload` | Upload form images (single or batch) |
| `GET` | `/jobs` | Job queue with search and sort |
| `GET` | `/jobs/<id>` | Job detail with logs and audit data |
| `GET` | `/jobs/<id>/review` | HITL review page |
| `POST` | `/jobs/<id>/review` | Submit field corrections |
| `GET` | `/jobs/<id>/exports/<fmt>` | Download export (json/csv/xlsx) |
| `DELETE` | `/api/jobs/<id>` | Delete a single job |
| `POST` | `/api/jobs/batch-delete` | Batch delete jobs |
| `POST` | `/api/config/discover` | Auto-discover form fields from template image |
| `GET` | `/api/template-preview` | Render template as PNG |
| `GET` | `/templates/<path>` | Serve template static files |
| `GET` | `/uploads/<path>` | Serve uploaded files |
| `GET` | `/outputs/<path>` | Serve output files |
| `POST` | `/api/utils/convert-to-png` | Convert uploaded image to PNG |
| `GET` | `/settings` | AI model and batch settings page |
| `GET` | `/api/models` | List available AI models |
| `POST` | `/api/models` | Update AI model configuration |
| `GET` | `/api/settings/batch` | Get batch processing settings |
| `POST` | `/api/settings/batch` | Update batch processing settings |
| `GET` | `/audits` | Audit log viewer |
| `GET` | `/evaluation` | Evaluation results page |
| `GET` | `/help` | Help / documentation page |

---

## Dictionary-Backed Validation

Place `.csv` files in `dictionaries/` (e.g., `dictionaries/nigerian_names.csv`). Reference them in field configs to improve extraction accuracy for domain-specific terms. The `DictionaryStore` in `ai_extraction/dictionary_matcher.py` loads and matches against these at inference time.

---

## Outputs

All processing outputs are stored under `outputs/`:

```
outputs/
├── <job_id>.json           # Structured extraction result (JSON)
├── <job_id>.csv            # Tabular extraction result
├── <job_id>.xlsx           # Relational XLSX export
├── audit.jsonl             # Append-only audit stream (all jobs)
└── crops/
    └── <job_id>/           # Per-field image crops for review
```

Each audit entry in `audit.jsonl` includes the job ID, extracted fields with confidence scores, validation status, reviewer corrections, and the three-tier confidence vector (`C_lp`, `C_dict`, `C_final`).

---

## License

Academic project — contact maintainer for licensing details.
