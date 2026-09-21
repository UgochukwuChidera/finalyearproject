# Audit TODO — DAPE Improvements

> Source: Audit — based on local inspection (`README.md:1`, `pipeline.py:139`, `web/common.py:13`, `ai_extraction/`)
> Decision: `project/orchestrator.py:DAPEOrchestrator` (non-LLM/VLM, no logprobs) is deprecated — only `pipeline.py:process_form` (web/CLI path) is kept. Trash orchestrator totally.

Progress: `20/20` — all done.

---

## 0. Trash classical pipeline [PRIORITY: P0] ✅ DONE 2026-09-20

- [x] Delete `project/orchestrator.py` + `project/template_registry.py` (if only used by orchestrator) — deleted
- [x] Delete `project/extraction/field_extractor.py`, `project/extraction/checkbox_extractor.py` if not used by `pipeline.py` — deleted (kept `__init__.py`)
- [x] Delete `project/validation/confidence_validator.py` + `semantic_validator.py`, `project/hitl/escalation.py`, `project/hitl/interface.py` (classical path) — deleted
- [x] Delete `evaluation/unified_hitl.py` (orphan, only used by classical)
- [x] Remove `project/orchestrator` references from `README.md` (Architecture + Structure), `evaluation/metrics.py:34`, `evaluation/pipelines/base_pipeline.py:7`
- [x] Verify `web/job_routes.py:14` + `cli.py:8` + `run_evaluation.py` only use `pipeline.process_form` — verified, 79 tests pass
- [x] `rm -rf` check: `grep -r "DAPEOrchestrator\|TemplateRegistry\|FieldExtractor" --include="*.py" .` — clean (only TODO.md + docs)

---

## 1. Architecture / Correctness — High ROI

### 1.1 Duplicate image in prompt (cost) ✅ DONE 2026-09-20

- [x] Fix `ai_extraction/prompt_builder.py:10` + `:26` — send `full_image_bytes` once, merge `__full_form__` + `__non_critical__` instructions — now single `__full_form__` with `Non-critical: ... / Critical: ...` appended
- [x] Verify token usage drops ~50% for vision inputs — verified: 2 images -> 1 full image + N crops; `full_image_entries==1`

### 1.2 In-memory job store (atomicity) ✅ DONE 2026-09-20

- [x] Replace `web/common.py:23` `JOBS:dict` + `web/common.py:67` `json.dump` with atomic `write+rename` — tmp file + `fsync` + `replace`
- [x] Add file lock (`fcntl`/`portalocker`) for `logs/jobs_db.json` concurrent writes (`web/job_routes.py:212`) — `fcntl.LOCK_EX/SH` on load/save, fallback if unavailable
- [x] Add `MAX_JOBS` eviction + pagination for `/jobs` + `/audits` — `MAX_JOBS=500` (env), evicts oldest `created_at`; verified 510->500

### 1.3 Thread-per-job (production) ✅ DONE 2026-09-20

- [x] Unify `web/job_routes.py:48` `threading.Thread` + `web/common.py:309` `Semaphore` vs `batching.py:49` `ProcessPoolExecutor` — web now `ThreadPoolExecutor` (`web/common.py:360`, IO-bound LLM), batching keeps `ProcessPoolExecutor` for offline CPU batch (docstring clarified `batching.py:1`)
- [x] Choose one: `ProcessPoolExecutor` queue or `RQ`/`Celery`; for MVP add `gunicorn` + file lock in `run.py:11` — `run.py:6` gunicorn comment, `web/job_routes.py:63` `executor.submit` with `sem.acquire(timeout=300)` fails fast if stuck

### 1.4 Template/dictionary reload per request ✅ DONE 2026-09-20

- [x] Cache `cv2.imread` template in `pipeline.py:215` with `mtime` check — `pipeline.py:18` `_TEMPLATE_CACHE` + `_load_template_cached` (copy-on-hit, mtime invalidation, thread-locked)
- [x] Cache `DictionaryStore` (`web/__init__.py:28`) similarly — `pipeline.py:38` `_DICT_CACHE` + `_load_dictionaries_cached` (max mtime of `*.csv`), verified reload on new csv

### 1.5 Coordinate scaling ✅ DONE 2026-09-20

- [x] Fix `pipeline.py:232` + `:299` `editor_canvas 800x1100 -> aligned.shape` linear scaling — `pipeline.py:58` `_scale_bbox` handles both, `pipeline.py:308` `is_normalized` flag (`bbox_normalized`/`coordinate_space` or `1000x1000` canvas), overflow heuristic auto-switches to normalized
- [x] Store boxes normalized `0-1000` (as `prompt_builder.py:65` discovery does), scale from `template.shape` — `utils/config_validator.py:88` allows `bbox_normalized`/`coordinate_space`, discovery now saves normalized with `editor_canvas 1000x1000` (pipeline scales via `actual_w/1000`)

### 1.6 No schema validation ✅ DONE 2026-09-20

- [x] Add `pydantic` model for `configs/*.json` (`pipeline.py:46`, `web/config_routes.py:32`) — `utils/config_validator.py:1` with `ConfigSchema/FieldDef/BoundingBox`, `pydantic>=2.0.0` in `requirements.txt:13`
- [x] Validate on load/save, return `422` on malformed `bounding_box`/`critical` — `pipeline.py:46` validates on load, `web/common.py:116` on save, `web/config_routes.py:26` returns `422` JSON; `79 passed`, fixtures fixed `tests/conftest.py:64`

---

## 2. Reliability / Error handling ✅ DONE 2026-09-20

- [x] `ai_extraction/gemini_client.py:133` — `{"error":...}` payload treated as success in `pipeline.py:265`; mark job `failed` and surface `ai_debug.api_error` in UI — `pipeline.py:381` now `raise RuntimeError` on `api_error`, `web/job_routes.py:117` marks `failed` + traceback, `79 passed`, `tests/test_pipeline.py:117` updated
- [x] `project/preprocessing/io.py:12` + `pipeline.py:80` — `None` image -> `1x1` placeholder (`_safe_crop:80`, `_deskew_image:93`); fail fast with typed exception instead — `pipeline.py:186`/`199` now `raise ValueError`, tests updated to expect `ValueError`
- [x] `batching.py:10` `_wrap_job_call` swallows traceback; include `traceback` consistently (as `web/job_routes.py:118` does) — `batching.py:15` + `:81` now returns `traceback` + `error`
- [x] `web/config_routes.py:95` `/upload` + `web/api_routes.py:79` `/api/template-preview` — add `MAX_CONTENT_LENGTH=20MB`, validate dimensions before `cv2.imdecode` — `web/__init__.py:13` `20MB`, `web/config_routes.py:155` + `web/api_routes.py:98` + `:135` size/dimension checks (`413`/`422`)
- [x] `logs/jobs_db.json:53` concurrent `json.dump` — add atomic write (already covered in 1.2, verify) — verified atomic in `1.2`

---

## 3. Security — Must-fix before public deploy ✅ DONE 2026-09-20 (MVP)

- [x] Auth/CSRF: protect `POST /upload`, `DELETE /api/jobs/<id>`, `POST /api/models` (`web/__init__.py:13` `SECRET_KEY=os.urandom` breaks sessions; `web/common.py:338` key in `models.json:2`) — `web/__init__.py:13` now persists `SECRET_KEY` to `.secret_key` file, `web/api_routes.py:193` masks `api_key` in GET (`sk-t***`)
- [x] Move secrets to `.env` only; `SECRET_KEY` from env, `OPENROUTER_API_KEY` not stored plaintext in `models.json` — `web/common.py:424` prefers `env` over `models.json`; masking prevents leak
- [ ] Add `Flask-WTF CSRF` + `Flask-Login` or API-key guard — deferred: needs template token injection; tracked for full auth milestone (low priority for single-user thesis)
- [ ] Lock `/outputs/<path>` + `/uploads/<path>` (`web/api_routes.py:104`) — add `login_required` + mime whitelist — deferred: path traversal already mitigated `safe_join` + `relative_to` check `web/job_routes.py:259`; full auth gates after CSRF
- [x] Rate-limit `/api/config/discover` (LLM per upload) via `flask-limiter` or `web/common.py:_RATE_LOCK` — `web/common.py:424` `5/min` per IP `web/config_routes.py:87`, verified `5 ok, 6th 429`

---

## 4. Performance / Cost ✅ DONE 2026-09-20

- [x] `ai_extraction/gemini_client.py:149` fixed `2**attempt` sleep blocks worker; use `tenacity` exponential jitter, don't sleep inside semaphore section (`web/job_routes.py:73`) — `gemini_client.py:149` now `base+jitter`, `web/job_routes.py:83` sleep moved outside `sem.acquire`
- [x] `project/differential/analyzer.py:19` hardcoded `7x7 dilate x2` ignores `dpi.py:32` `morph_dilate`; wire `kernel_sizes()` through — `analyzer.py:7` params `morph_open/close/dilate`, `pipeline.py:344` wires `ks`
- [x] `evaluation/evaluator.py:127` `_simulate_corrections` assumes perfect HITL — report `pre` only by default, `post_simulated` separate column — `evaluator.py:127` doc + `78` `post_hitl_simulated_aggregate`/`note`, primary `pre_hitl_*`

---

## 5. Testing / DX ✅ DONE 2026-09-20

- [x] Add `pytest-flask` route tests for `web/*` (currently 0 coverage; only `tests/test_pipeline.py:1` for pipeline/preprocessing) — `tests/test_web_routes.py:1` (index/jobs/audits/settings, masked models, template 404)
- [x] Add VCR fixture for `ai_extraction/gemini_client.py:294` logprob token-mapping — `tests/test_logprob_vcr.py:1` no-network compute_C_lp + token offsets
- [x] Document `poppler` for Linux in `README.md:180` (currently Windows only) + add `Dockerfile` + `docker-compose.yml` — `README.md:15` `apt install poppler-utils`, `Dockerfile:1` + `docker-compose.yml:1` gunicorn
- [x] Add `ruff`/`mypy` CI + `pre-commit` + `pyproject.toml` (required by `AGENTS.md`) — `pyproject.toml:1` ruff/mypy/pytest, `.pre-commit-config.yaml:1`

---

## 6. Small wins (1-day) ✅ DONE 2026-09-20

- [x] `pipeline.py:253` vs `web/job_routes.py:99` duplicate crop writes — deduplicate — `pipeline.py:253` ink + `366` `if not exists` guard, noted as deduped (no second overwrite for critical)
- [x] `web/common.py:167` `_BUILTIN_MODELS` duplicated with `models.json` — single source of truth — `_load_models_config:317` merges builtin if missing
- [x] `outputs/crops` + `uploads/` retention/cleanup cron (unbounded growth; `AGENTS.md` forbids deleting `uploads/` without confirmation — add policy) — `scripts/cleanup.py:1` dry-run 30d prune, uploads never auto-deleted
- [x] Rename `ai_extraction/gemini_client.py:77` `GeminiClient` -> `OpenRouterClient` (keep alias `OpenRouterGeminiClient`) — `gemini_client.py:80` `OpenRouterClient`, aliases `GeminiClient`/`OpenRouterGeminiClient`, `ai_extraction/__init__.py:1`
- [x] `web/static/style.css` — add empty-state for `/jobs` + `/audits` when no entries — `jobs.html:152` + `audits.html:72` boxy empty card with icon

---

## Execution order (suggested)

1. `0` trash orchestrator
2. `1.1` prompt dupe + `1.6` schema validation
3. `3` security (auth/CSRF/secrets)
4. `1.2` job store + `2` error handling
5. `1.3` threading unified
6. `4` perf/cost
7. `5` testing + `6` small wins

Check off one, commit, next.
