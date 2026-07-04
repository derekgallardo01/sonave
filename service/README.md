# Sonave detection service (Phase 2)

Wraps the trained detector (`models/sonave_xlsr_rw/`) as (a) an HTTP API and (b) an
offline "analyze a recording" tool. This is the seam every other product layer plugs
into. See `../docs/product-sketch.md` for the full architecture.

## Files
- `detector.py` — detection core: loads the model once, turns audio → calibrated
  verdict (`real` / `suspect` / `fake`). Shared by everything below.
- `app.py` — FastAPI microservice (`/score`, `/score_json`, `/healthz`, `/version`).
- `analyze_meeting.py` — offline analyzer: recording → windowed rolling scores →
  flagged stretches → JSON report. Supports **per-speaker** mode via a turns CSV.
- `orchestrator.py` — the **live per-speaker engine** (Phase 4/5 core): ingests
  `(speaker_id, chunk, ts)` events, keeps a rolling per-speaker confidence, fires
  ALERT + wire-HOLD hooks on a sustained red. `simulate_meeting()` replays a recording
  through the exact live path for offline testing.
- `dashboard.py` — **live web monitor** (Phase 5 surface): per-speaker RAG cards,
  rolling P(fake) meters, alert/wire-hold feed, and a live GPU-cost meter. Self-
  contained (inline HTML/CSS/JS, no external assets). Built-in "Run demo" replays a
  stitched meeting (real → real → ElevenLabs fake → real) so you can watch Bob turn
  red mid-call and the wire-hold fire.
- `Dockerfile`, `requirements.txt` — containerized deploy.

## Dashboard (see it live)
```bash
.venv/Scripts/python.exe -m pip install -r service/requirements.txt
.venv/Scripts/python.exe -m uvicorn service.dashboard:app --port 8000
# open http://localhost:8000  → click "Run demo meeting"
```
The board polls `/api/status` every second. In production the Recall.ai adapter POSTs
`/api/ingest {speaker_id, audio_b64, ts}` per chunk and the same board reflects it live.

## Verdict policy (tunable)
`P(fake)` → verdict via two thresholds (env-overridable):
`SONAVE_TAU_REAL` (default 0.40) and `SONAVE_TAU_FAKE` (default 0.70).
Defaults sit at the calibrated operating point from
`../results/detector_v2_progress.md` (~64% catch / ~92% real-voice accuracy on
real-world audio). Raise `TAU_REAL` to cry wolf less; lower it to catch more.

## Run the API (in the repo's `.venv`)
```bash
.venv/Scripts/python.exe -m pip install -r service/requirements.txt
.venv/Scripts/python.exe -m uvicorn service.app:app --host 0.0.0.0 --port 8000
# health:
curl http://localhost:8000/healthz
# score a clip:
curl -F "file=@data/real/itw_real_10145.wav" http://localhost:8000/score
```
Response:
```json
{ "p_fake": 0.03, "verdict": "real", "confidence": 0.9,
  "model_version": "sonave_xlsr_rw", "latency_ms": 120 }
```

## Analyze a recording (the demo)
```bash
.venv/Scripts/python.exe service/analyze_meeting.py path/to/meeting.wav --hop 2
```
Prints an overall verdict + flagged time-stretches and writes `<file>.sonave.json`
(a per-window timeline you can chart). Quick self-test: stitch a fake stretch between
real clips and confirm it flags the middle —
```bash
.venv/Scripts/python.exe scratchpad/make_demo.py    # builds data/_demo_meeting.wav
.venv/Scripts/python.exe service/analyze_meeting.py data/_demo_meeting.wav
```

## Per-speaker + live engine
```bash
# per-speaker offline report (turns.csv = start,end,speaker)
.venv/Scripts/python.exe service/analyze_meeting.py meeting.wav --segments turns.csv

# replay a meeting through the LIVE engine (fires alerts / wire-hold on red)
.venv/Scripts/python.exe service/orchestrator.py meeting.wav turns.csv \
    --alert-webhook https://hooks.slack.com/... \
    --hold-webhook  https://payments.internal/hold
```
In production the capture layer supplies `(speaker_id, chunk, ts)` events and speaker
turns; `Orchestrator.ingest()` is called per chunk and `Orchestrator.status()` feeds
the live dashboard.

## Docker
```bash
docker build -t sonave-detect -f service/Dockerfile .
docker run --gpus all -p 8000:8000 sonave-detect
```

## Not yet (next steps)
- **Recall.ai adapter (Phase 3):** translate Recall's audio/speaker events into
  `Orchestrator.ingest()` calls. The only piece needing an external account + keys.
- **Speaker turns source:** live from Recall; offline via pyannote/speechbrain to
  generate the `turns.csv`.
- **Dashboard UI (Phase 5):** render `Orchestrator.status()` as a live per-speaker RAG
  board; the data shape is already there.
- **Cost metering (Phase 4):** count inferences → cost-per-meeting-hour on the board.
