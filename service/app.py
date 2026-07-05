"""
app.py — Sonave detection microservice (Phase 2).

Wraps the trained detector behind a tiny HTTP API. The orchestration layer POSTs
~4 s audio chunks and gets back a calibrated verdict. Stateless per request.

Run:
    uvicorn service.app:app --host 0.0.0.0 --port 8000
Or via the Dockerfile in this directory.

Endpoints:
    POST /score        multipart file OR raw body: audio (wav/flac/ogg) -> verdict
    POST /score_json   { "audio_b64": "...", "speaker_id": "...", "ts": 0.0 }
    GET  /healthz      model loaded + device
    GET  /version      model + threshold policy
"""
from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "src", _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import detector

app = FastAPI(title="Sonave Detection", version="0.1")


@app.on_event("startup")
def _warm():
    detector.load()   # load the model at boot, not on first request


@app.get("/healthz")
def healthz():
    model, device = detector.load()
    return {"status": "ok", "device": device, "model": detector.MODEL_VERSION}


@app.get("/version")
def version():
    return {"model_version": detector.MODEL_VERSION,
            "tau_real": detector.TAU_REAL, "tau_fake": detector.TAU_FAKE}


class ScoreJSON(BaseModel):
    audio_b64: str
    speaker_id: str | None = None
    ts: float | None = None


@app.post("/score")
async def score(file: UploadFile = File(...), speaker_id: str | None = None):
    t0 = time.perf_counter()
    data = await file.read()
    res = detector.score_bytes(data)
    res["speaker_id"] = speaker_id
    res["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    return res


@app.post("/score_clip")
async def score_clip(file: UploadFile = File(...), speaker_id: str | None = None):
    """Score a WHOLE clip (windowed mean) — the live-monitor endpoint. Replaces the
    local GPU scorer: POST a capture chunk, get back the rolling-style verdict."""
    t0 = time.perf_counter()
    data = await file.read()
    res = detector.score_clip(data)
    res["speaker_id"] = speaker_id
    res["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    return res


@app.post("/score_json")
def score_json(body: ScoreJSON):
    t0 = time.perf_counter()
    res = detector.score_bytes(base64.b64decode(body.audio_b64))
    res["speaker_id"] = body.speaker_id
    res["ts"] = body.ts
    res["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    return res
