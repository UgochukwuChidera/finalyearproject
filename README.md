# DAPE

Hybrid differential + AI form extraction pipeline.

## What changed

- Local OCR engines removed from runtime pipeline.
- Extraction now uses **Gemini Flash Lite via OpenRouter**.
- Hybrid strategy:
  - **Critical fields** (`critical: true`) use differential ink-only crops.
  - **Non-critical fields** use full-form AI extraction + template JSON diff.
- Single AI request per form through multi-image prompting.
- Flask UI for config management, uploads, jobs, and review.
- Audit JSONL enhanced with `C_lp`, `C_dict`, `C_final`, status, reviewer/correction.

## Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **System Dependencies (Windows):**
   - This project uses `pdf2image`, which requires **Poppler**. 
   - Download the latest [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases) and add the `bin/` directory to your System PATH.

3. **Configure API Key:**
   - Create a `.env` file in the root directory (you can copy `.env.example`).
   - Add your [OpenRouter](https://openrouter.ai/) API key using the following format:
     ```env
     OPENROUTER_API_KEY="your-api-key-here"
     ```
   - Alternatively, set it as a system environment variable:
     - **Windows (PowerShell):** `$env:OPENROUTER_API_KEY="<your-key>"`
     - **Windows (CMD):** `set OPENROUTER_API_KEY=<your-key>`
     - **Linux/macOS:** `export OPENROUTER_API_KEY="<your-key>"`

## Run Web App

```bash
python run.py
```

Open `http://127.0.0.1:5060`. (Note: `app.py` is also available as an entry point).

## Run pipeline from CLI

```bash
python main.py --image form/medical_screening_01.tif --config-name medical_screening_v1
```

## Config schema (example)

`configs/medical_screening_v1.json` includes:

- `template_path`
- `template_extraction`
- `fields[].critical`
- `fields[].dictionary`
- `confidence_weights`
- `thresholds`

## Precompute blank template extraction

```bash
python scripts/precompute_template_extraction.py --config configs/medical_screening_v1.json
```

This creates/updates the `template_extraction` JSON used for text-based diff.

## Flask routes

- `GET /` dashboard + upload
- `GET /configs`
- `GET|POST /configs/new`
- `GET|POST /configs/<name>/edit`
- `POST /upload`
- `GET /jobs`
- `GET /jobs/<id>`
- `GET|POST /jobs/<id>/review`
- `POST /api/dictionaries/upload`

## Dictionaries

The pipeline supports dictionary-backed validation for specific fields. 
- Place `.csv` files in the `dictionaries/` folder (e.g., `dictionaries/nigerian_names.csv`).
- Reference them in your field config: `"dictionary": "nigerian_names"` (without the extension).
- This improves extraction accuracy for regional names and specific terminology.

## Output

- Structured exports in `outputs/`
- Audit stream in `outputs/audit.jsonl`
- Field crops in `outputs/crops/<job_id>/`
