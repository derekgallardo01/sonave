# Tests

Fast, offline unit + contract tests for the parts of Sonave where bugs actually bite.
No GPU, no network, no model downloads — every external is mocked.

```bash
.venv\Scripts\python.exe -m pytest          # fast suite (~2s), the default
.venv\Scripts\python.exe -m pytest -m gpu   # + end-to-end model scoring (needs a trained model)
```

## What's covered
| File | Surface |
|---|---|
| `test_railway_quality.py` | Audio-quality math on the capture path — the stdlib rewrite that once broke prod (audioop removed in 3.13): silence/loud/clipping/warm-up verdicts, odd-byte tolerance. |
| `test_railway_api.py` | Capture-service HTTP: page render (no leftover placeholders), `POST /api/verdict`, `/api/quality` verdict-merge + test-speaker filter, `/captures`, path-traversal safety, favicon. |
| `test_railway_scoring.py` | Off-path hosted scoring (`SONAVE_SCORER_URL` → Modal): verdict/rolling bookkeeping, `_av` thresholds, safe no-op without a URL, **errors never propagate** (capture must never break). |
| `test_service_api.py` | Detection microservice contract (`/healthz`, `/version`, `/score`, `/score_clip`, `/score_json`) with the detector mocked. |
| `test_detector_logic.py` | The tri-state real/suspect/fake thresholds and result shaping. |
| `test_model_sls.py` | `fit_length` crop/pad — the transform every clip passes through. |
| `test_tools.py` | `verdict_monitor` (`_verdict`, `_post_clip` multipart) + `play_into_meet` (`_files`, `_find_device`). |
| `test_smoke_gpu.py` | **Opt-in (`-m gpu`).** Loads the real model and scores a real + fake clip end-to-end. Auto-skips when models/data aren't present. |

## What's intentionally NOT unit-tested
- **Training / eval scripts** (`src/train_xlsr.py`, `src/eval_xlsr.py`, …) — validated by the metrics they produce (see `results/`), not unit tests.
- **Live external integrations** (Recall.ai bot lifecycle, real Modal deploy, real Meet capture) — exercised manually; their client code is mocked here.
