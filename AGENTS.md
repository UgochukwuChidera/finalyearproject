# Project-Specific Agent Instructions

## Safety
- **NEVER** remove or modify any files/directories inside `uploads/` unless explicitly asked.
- **NEVER** delete user data or processing outputs without confirmation.
- Always use absolute paths for file operations.

## Code Style
- Follow existing patterns: boxy/minimalist CSS (no border-radius), Phosphor icons, Inter font.
- No comments in code unless explaining a non-obvious workaround.
- Python: type hints on function signatures, `ruff`-compatible style.
- HTML/Jinja: inline styles preferred over separate CSS classes for one-off layout.

## Architecture
- `web/routes.py` — Flask routes + job queue (threading-based).
- `main.py` — `process_form()` pipeline orchestration.
- `project/preprocessing/` — image preprocessing stages.
- `ai_extraction/gemini_client.py` — OpenRouter API client with retry logic.
- Batch concurrency is controlled via `threading.Semaphore` + inter-request delay.
- API rate limits are configurable in Settings UI (`/settings`).

## Testing
- No formal test framework detected. Run `python run.py` to start the dev server.
- Check `web/` templates by loading the relevant page in a browser.
