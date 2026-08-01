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
import os
import secrets
import sys
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "src", _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import detector

app = FastAPI(title="Sonave Detection", version="0.1")

# Auth is OPT-IN (unset SONAVE_API_TOKEN => open, as before). When set, /score* and
# /version require a bearer/X-Sonave-Token; /healthz stays open for platform probes.
API_TOKEN = os.environ.get("SONAVE_API_TOKEN", "")
MAX_UPLOAD_MB = float(os.environ.get("SONAVE_MAX_UPLOAD_MB", "25"))


def require_auth(request: Request):
    if not API_TOKEN:
        return
    auth = request.headers.get("authorization", "")
    bearer = auth[7:] if auth.lower().startswith("bearer ") else ""
    tok = (request.headers.get("x-sonave-token") or bearer
           or request.query_params.get("token"))
    if not (tok and secrets.compare_digest(tok, API_TOKEN)):
        raise HTTPException(status_code=401, detail="unauthorized")


def check_size(request: Request):
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_UPLOAD_MB * 1_000_000:
        raise HTTPException(status_code=413, detail=f"upload exceeds {MAX_UPLOAD_MB:g} MB")


@app.on_event("startup")
def _warm():
    detector.load()   # load the model at boot, not on first request


@app.get("/healthz")
def healthz():
    model, device = detector.load()
    return {"status": "ok", "device": device, "model": detector.MODEL_VERSION}


@app.get("/version", dependencies=[Depends(require_auth)])
def version():
    return {"model_version": detector.MODEL_VERSION,
            "tau_real": detector.TAU_REAL, "tau_fake": detector.TAU_FAKE}


class ScoreJSON(BaseModel):
    audio_b64: str
    speaker_id: str | None = None
    ts: float | None = None


@app.post("/score", dependencies=[Depends(require_auth), Depends(check_size)])
async def score(file: UploadFile = File(...), speaker_id: str | None = None):
    t0 = time.perf_counter()
    data = await file.read()
    res = detector.score_bytes(data)
    res["speaker_id"] = speaker_id
    res["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    return res


@app.post("/score_clip", dependencies=[Depends(require_auth), Depends(check_size)])
async def score_clip(file: UploadFile = File(...), speaker_id: str | None = None):
    """Score a WHOLE clip (windowed mean) — the live-monitor endpoint. Replaces the
    local GPU scorer: POST a capture chunk, get back the rolling-style verdict."""
    t0 = time.perf_counter()
    data = await file.read()
    res = detector.score_clip(data)
    res["speaker_id"] = speaker_id
    res["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    return res


@app.post("/score_json", dependencies=[Depends(require_auth), Depends(check_size)])
def score_json(body: ScoreJSON):
    t0 = time.perf_counter()
    res = detector.score_bytes(base64.b64decode(body.audio_b64))
    res["speaker_id"] = body.speaker_id
    res["ts"] = body.ts
    res["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    return res
