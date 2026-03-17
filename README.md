# DAPE — Differential Analysis and Preprocessing for Extraction

**AI and Intelligent Character Recognition (ICR) for Manual Form Data Processing**  
Final Year Project | Python 3.11+

---

## Overview

DAPE is a modular, template-driven pipeline for extracting structured data from
scanned paper forms. It compares each completed form against a blank reference
template to isolate user-provided input, then extracts, validates, and exports
that data — with optional human review for uncertain fields.

---

## Pipeline Flow

```
Scanned Form (TIF/JPG/PNG/PDF)
         │
         ▼
① PREPROCESS
   grayscale → denoise → deskew → illumination normalize → binarize
   All kernels scale automatically to scan DPI (--dpi flag)
         │
         ▼
② ALIGN TO BLANK TEMPLATE
   ORB feature matching → RANSAC homography → pixel-level registration
         │
         ▼
③ DIFFERENTIAL ANALYSIS
   pixel-wise diff → morphological cleanup → binary interaction mask
   (isolates handwriting and markings from template structure)
         │
         ▼
④ FIELD EXTRACTION  (three methods evaluated in parallel)
   ├── Tesseract       legacy rule-based OCR
   ├── PaddleOCR v3    LSTM/ANN with real confidence scores
   └── DeepSeek API    Document AI / LLM-based OCR
         │
         ▼
⑤ STRUCTURE OUTPUT
   field_id → semantic key → JSON schema
   nested table fields → relational sheets in XLSX
         │
         ▼
⑥ VALIDATE
   confidence threshold check + semantic format rules
   flag low-confidence / format-failed fields
         │
         ▼
⑦ HITL  (flagged fields only)
   Flask web UI at http://127.0.0.1:5050
   human corrects → final_value updated → pipeline continues
         │
         ▼
⑧ EXPORT
   JSON (canonical) + relational XLSX (RECORDS / COURSES / VALIDATION_LOG)
   + CSV for simple consumption
         │
         ▼
⑨ AUDIT LOG
   per-form JSON log: accuracy, escalation rate, corrections, timing
```

---

## Installation

### 1. System dependencies

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils

# Windows — install Tesseract from:
# https://github.com/UB-Mannheim/tesseract/wiki
# Then set TESSDATA_PREFIX or pass --tesseract-cmd to the CLI
```

### 2. Python packages

```bash
pip install -r requirements.txt
```

> **PaddlePaddle note:** The command above installs the CPU-only version.
> For GPU support: `pip install paddlepaddle-gpu`

---

## Project Structure

```
DAPE/
├── main.py                        ← single-form processing entry point
├── run_evaluation.py              ← 6-condition comparative evaluation
├── generate_ground_truth_excel.py ← create ground_truth_entry.xlsx
├── excel_to_json.py               ← convert filled Excel → 45 JSON files
├── requirements.txt
├── README.md
│
├── project/                       ← core DAPE pipeline
│   ├── orchestrator.py            ← end-to-end pipeline controller
│   ├── template_registry.py       ← loads templates/registry.json
│   ├── preprocessing/             ← image normalisation modules
│   ├── alignment/                 ← ORB + homography alignment
│   ├── differential/              ← pixel-diff + mask generation
│   ├── extraction/                ← Tesseract OCR + checkbox detection
│   ├── validation/                ← confidence + semantic validation
│   ├── hitl/                      ← Flask HITL review interface
│   ├── output/                    ← JSON / relational XLSX / CSV export
│   └── template_analyzer/         ← auto-detect fields from blank form
│
├── evaluation/                    ← 3-pipeline × 2-condition comparison
│   ├── evaluator.py
│   ├── pipelines/
│   │   ├── tesseract_pipeline.py  ← Condition 1
│   │   ├── docai_pipeline.py      ← Condition 2 (PaddleOCR)
│   │   └── dape_pipeline.py       ← Condition 3
│   ├── unified_hitl.py
│   ├── ground_truth.py
│   └── metrics.py
│
├── utils/                         ← general utilities
│   ├── tiff_operations.py         ← TIFF split/merge/extract/delete
│   ├── document_processor.py      ← multi-format page extraction
│   └── image_preprocessor.py      ← standalone preprocessing helpers
│
├── templates/                     ← blank reference template images
│   ├── registry.json
│   ├── student_academic_record_blank.tif
│   ├── medical_screening_blank.tif
│   └── leave_application_blank.tif
│
├── form/                          ← 45 scanned completed forms
│   ├── student_academic_record_01.tif … _15.tif
│   ├── medical_screening_01.tif       … _15.tif
│   └── leave_application_01.tif       … _15.tif
│
├── ground_truth/
│   ├── ground_truth_entry.xlsx    ← fill this in, then run excel_to_json.py
│   └── *.json                     ← generated after running excel_to_json.py
│
├── outputs/                       ← extracted form data written here
└── logs/                          ← per-form audit logs written here
```

---

## Extractor Architecture

DAPE supports three interchangeable OCR / extraction backends.  All three share
the same `BaseExtractor` interface defined in `project/extraction/base.py` and
produce an `ExtractionResult(text, fields, layout, raw_ocr)` container that is
consumed unchanged by the HITL, validation, and evaluation modules.

### Available backends

| Name | Module | Description | Extra deps |
|---|---|---|---|
| `tesseract` *(default)* | `TesseractFormExtractor` | Rule-based OCR (`--oem 3 --psm 6`) | `pytesseract` + system Tesseract |
| `paddle` | `PaddleOCRExtractor` | Deep-learning OCR (DBNet + CRNN-LSTM). Real CTC confidence scores. CPU-viable. | `pip install paddlepaddle paddleocr` |
| `deepseek` | `DeepSeekExtractor` | Cloud Document AI via DeepSeek Vision API. Sends full page, returns structured JSON. | `pip install openai` + API key |

### Selecting the backend

**Via CLI flag:**
```bash
# Tesseract (default — no extra deps):
python main.py --image form.tif --template-id student_academic_record

# PaddleOCR deep-learning backend:
python main.py --image form.tif --template-id student_academic_record --extractor paddle

# DeepSeek Vision Document AI:
python main.py --image form.tif --template-id student_academic_record \
  --extractor deepseek --deepseek-key sk-xxxx
```

**Via environment variable:**
```bash
export OCR_EXTRACTOR=paddle        # set default for all runs
python main.py --image form.tif --template-id student_academic_record
```

**Via Python API:**
```python
from project.extraction.factory import get_extractor

extractor = get_extractor("paddle")                    # PaddleOCR
extractor = get_extractor("deepseek", api_key="sk-…") # DeepSeek
result    = extractor.extract("form.tif", template_id="student_academic_record")
print(result.fields)   # {"name": "Jane Smith", "dob": "12/03/1995", …}
```

### DeepSeek configuration

The DeepSeek extractor reads credentials from environment variables:

| Variable | Required | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | ✓ | Your DeepSeek secret key |
| `DEEPSEEK_API_BASE` | — | API base URL (default: `https://api.deepseek.com/v1`) |
| `DEEPSEEK_MODEL` | — | Model name (default: `deepseek-chat`) |

Missing `DEEPSEEK_API_KEY` raises a clear `RuntimeError` with setup instructions.

---

## Usage

### Step 1 — Register templates (one-time per template type)

```bash
python main.py --register \
  --template-id student_academic_record \
  --template-image templates/student_academic_record_blank.tif
```

Or edit `templates/registry.json` directly (field coordinates are in pixels).

### Step 2 — Process a single form

```bash
python main.py \
  --image  form/student_academic_record_01.tif \
  --template-id student_academic_record \
  --dpi 600
```

**Pipeline flags:**

| Flag | Default | Description |
|---|---|---|
| `--image` | required | Path to scanned form |
| `--template-id` | required | Key in registry.json |
| `--dpi` | `300` | Actual scan DPI — scales all kernels |
| `--threshold` | `0.60` | Confidence threshold for HITL escalation |
| `--no-hitl` | off | Skip HITL review (auto-accept all) |
| `--hitl-port` | `5050` | Port for HITL Flask UI |
| `--output-dir` | `outputs/` | Where to write exports |
| `--tesseract-cmd` | auto | Explicit path to Tesseract binary |

**Extractor flags:**

| Flag | Default | Description |
|---|---|---|
| `--extractor` | `tesseract` | OCR backend: `tesseract`, `paddle`, or `deepseek` |
| `--deepseek-key` | env | DeepSeek API key (or set `DEEPSEEK_API_KEY`) |
| `--deepseek-model` | `deepseek-chat` | DeepSeek model name |
| `--paddle-lang` | `en` | PaddleOCR language code |
| `--paddle-gpu` | off | Enable GPU for PaddleOCR |

### Step 3 — HITL review

If flagged fields exist, the pipeline pauses and prints:

```
[HITL] Review interface → http://127.0.0.1:5050
       3 field(s) require attention.
       Waiting for submission…
```

Open the URL in a browser, correct any flagged fields, click **Submit corrections**.

### Step 4 — Batch processing

```bash
python main.py --batch \
  --forms-dir form/ \
  --template-id student_academic_record \
  --dpi 600
```

### Step 5 — Comparative evaluation

```bash
# Fill ground truth first
python generate_ground_truth_excel.py   # creates ground_truth_entry.xlsx
# ... fill in the Excel spreadsheet ...
python excel_to_json.py                 # converts to 45 JSON files

# Run evaluation — Tesseract + PaddleOCR + DAPE (all 3 pipelines × before/after HITL):
python run_evaluation.py --template-id student_academic_record

# Run with DeepSeek replacing PaddleOCR for Condition 2:
python run_evaluation.py --deepseek-key sk-xxxx
```

---

## Output Files

For each processed form, the following are written to `outputs/`:

| File | Contents |
|---|---|
| `<form_id>.json` | Canonical extracted data |
| `<form_id>.xlsx` | Relational workbook: RECORDS + COURSES + VALIDATION_LOG |
| `<form_id>.csv` | Flat key-value export |

`logs/<form_id>_<timestamp>.json` — audit record with pipeline metrics.

---

## Three Form Templates

| Template ID | Form Name | Fields |
|---|---|---|
| `student_academic_record` | Student Academic Record Form | 27 fields incl. course table |
| `medical_screening` | Medical Screening and Declaration Form | ~20 fields |
| `leave_application` | Leave Application Form | ~15 fields |

---

## Notes

- Student signatures are **excluded** from extraction by design (not evaluable as text).
- The course registration table is extracted as individual fields (`sn_1`, `course_code_1`, etc.) and rendered as a separate COURSES sheet in the XLSX export.
- All kernel sizes scale automatically with DPI. Always pass `--dpi` matching your scanner setting.
- HITL corrections update `final_value` only. They never retrain models or alter thresholds.
