"""
app.py — Sonave detection microservice (production-ready).

Wraps the trained detector behind a tiny HTTP API. The orchestration layer POSTs
~4 s audio chunks and gets back a calibrated verdict. Stateless per request.

Run:
    uvicorn service.app:app --host 0.0.0.0 --port 8000
Or via the Dockerfile in this directory.

Endpoints:
    POST /score        multipart file OR raw body: audio (wav/flac/ogg) -> verdict
    POST /score_clip   {file} -> windowed mean verdict (live-monitor endpoint)
    POST /score_json   { "audio_b64": "...", "speaker_id": "...", "ts": 0.0,
                         "voiceprint_b64": "..." }
    GET  /healthz      model loaded + device
    GET  /version      model + threshold policy
    GET  /ready        deep readiness probe (model loaded + scoring works)
"""
from __future__ import annotations

import base64
import io
import logging
import os
import secrets
import sys
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "src", _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import detector
import enroll

# --- logging -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("sonave.api")

app = FastAPI(title="Sonave Detection", version="0.3")

# Auth is OPT-IN (unset SONAVE_API_TOKEN => open, as before). When set, /score* and
# /version require a bearer/X-Sonave-Token; /healthz and /ready stay open for probes.
API_TOKEN = os.environ.get("SONAVE_API_TOKEN", "")
MAX_UPLOAD_MB = float(os.environ.get("SONAVE_MAX_UPLOAD_MB", "25"))

# Simple in-memory rate-limit bucket per client IP (good enough for single-instance).
# For scale, swap to Redis or a WAF rule.
_RATE_LIMIT_RPS = float(os.environ.get("SONAVE_RATE_LIMIT_RPS", "10"))
_rate_buckets: dict[str, tuple[float, float]] = {}  # ip -> (tokens, last_update)


def _rate_limit_ok(ip: str) -> bool:
    now = time.time()
    tokens, last = _rate_buckets.get(ip, (_RATE_LIMIT_RPS, now))
    tokens = min(_RATE_LIMIT_RPS, tokens + (now - last) * _RATE_LIMIT_RPS)
    _rate_buckets[ip] = (tokens, now)
    if tokens >= 1.0:
        _rate_buckets[ip] = (tokens - 1.0, now)
        return True
    return False


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


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or str(uuid.uuid4())[:8]


@app.on_event("startup")
def _warm():
    detector.load()   # load the model at boot, not on first request


@app.middleware("http")
async def _middleware(request: Request, call_next):
    rid = _request_id(request)
    request.state.rid = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception("unhandled exception rid=%s path=%s", rid, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "internal error", "request_id": rid},
        )
    response.headers["x-request-id"] = rid
    logger.info("%s %s %s %.3fs rid=%s",
                request.method, request.url.path, response.status_code,
                time.perf_counter() - t0, rid)
    return response


@app.get("/healthz")
def healthz():
    model, device = detector.load()
    return {"status": "ok", "device": device, "model": detector.MODEL_VERSION}


@app.get("/ready")
def ready():
    """Deep readiness: model loaded and a tiny forward pass works."""
    try:
        import numpy as np
        model, device = detector.load()
        # One silent window -> should return a valid verdict without crashing
        arr = np.zeros(detector.model_sls.MAX_LEN, dtype=np.float32)
        res = detector.score_array(arr)
        return {"status": "ready", "device": device, "model": detector.MODEL_VERSION,
                "verdict": res["verdict"]}
    except Exception as exc:
        logger.error("readiness probe failed: %s", exc)
        raise HTTPException(status_code=503, detail="not ready")


@app.get("/version", dependencies=[Depends(require_auth)])
def version():
    return {"model_version": detector.MODEL_VERSION,
            "tau_real": detector.TAU_REAL, "tau_fake": detector.TAU_FAKE}


class ScoreJSON(BaseModel):
    audio_b64: str
    speaker_id: str | None = None
    ts: float | None = None
    voiceprint_b64: str | None = None


def _decode_audio(data: bytes) -> "np.ndarray":
    """Decode wav/flac/ogg bytes to 16 kHz mono float array (re-uses detector logic)."""
    import librosa
    import soundfile as sf
    try:
        wav, sr = sf.read(io.BytesIO(data))
    except Exception:
        wav, sr = librosa.load(io.BytesIO(data), sr=None, mono=True)
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype="float32")
    if sr != detector.model_sls.SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=detector.model_sls.SR)
    return wav


def _maybe_fuse(res: dict, data: bytes, speaker_id: str | None,
                voiceprint_b64: str | None) -> dict:
    """If a voiceprint is provided inline, fuse deepfake score with speaker verification."""
    if not voiceprint_b64 or not speaker_id:
        return res
    try:
        import numpy as np
        vp = np.frombuffer(base64.b64decode(voiceprint_b64), dtype=np.float32)
        wav = _decode_audio(data)
        fr = enroll.fused_risk_with_voiceprint(
            res.get("p_fake", 0.5), speaker_id, wav, vp)
        res["p_fake"] = fr["p_fake"]
        res["risk"] = fr["risk"]
        res["verdict"] = fr["verdict"]
        res["speaker_check"] = fr.get("speaker_check")
        res["match_conf"] = fr.get("match_conf")
        res["mismatch_risk"] = fr.get("mismatch_risk")
    except Exception as exc:
        logger.warning("voiceprint fusion failed rid=%s: %s", res.get("request_id"), exc)
    return res


@app.post("/score", dependencies=[Depends(require_auth), Depends(check_size)])
async def score(request: Request, file: UploadFile = File(...),
                speaker_id: str | None = None, voiceprint_b64: str | None = None):
    client = request.client.host if request.client else "unknown"
    if not _rate_limit_ok(client):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    t0 = time.perf_counter()
    data = await file.read()
    res = detector.score_bytes(data)
    res = _maybe_fuse(res, data, speaker_id, voiceprint_b64)
    res["speaker_id"] = speaker_id
    res["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    res["request_id"] = request.state.rid
    return res


@app.post("/score_clip", dependencies=[Depends(require_auth), Depends(check_size)])
async def score_clip(request: Request, file: UploadFile = File(...),
                     speaker_id: str | None = None, voiceprint_b64: str | None = None):
    """Score a WHOLE clip (windowed mean) — the live-monitor endpoint."""
    client = request.client.host if request.client else "unknown"
    if not _rate_limit_ok(client):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    t0 = time.perf_counter()
    data = await file.read()
    res = detector.score_clip(data)
    res = _maybe_fuse(res, data, speaker_id, voiceprint_b64)
    res["speaker_id"] = speaker_id
    res["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    res["request_id"] = request.state.rid
    return res


@app.post("/score_json", dependencies=[Depends(require_auth), Depends(check_size)])
def score_json(request: Request, body: ScoreJSON):
    t0 = time.perf_counter()
    data = base64.b64decode(body.audio_b64)
    res = detector.score_bytes(data)
    res = _maybe_fuse(res, data, body.speaker_id, body.voiceprint_b64)
    res["speaker_id"] = body.speaker_id
    res["ts"] = body.ts
    res["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    res["request_id"] = request.state.rid
    return res
