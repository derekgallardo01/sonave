"""
Sonave capture service — deploys to Railway (CPU-only, no model needed).

Its ONE job: let you drop the Sonave bot into any meeting and save the real
Meet-piped audio for training. Scoring/retraining happens offline on your GPU box;
this just collects ground-truth domain data at scale.

With speaker enrollment:
  - Enroll a speaker from captured clips (CPU ECAPA-TDNN)
  - Live scoring fuses deepfake detection + voiceprint verification on Modal GPU
  - Voiceprints persist on Railway's /data volume

Dependency-light on purpose: FastAPI + stdlib wave (no torch / numpy / soundfile),
so the Railway image is tiny and builds in seconds. Enrollment adds torch CPU +
speechbrain on-demand (one-time per speaker).

Endpoints:
  GET  /                 marketing landing page (public)
  GET  /console          operator console: send a bot, live verdicts, captures, enrollment
  POST /bot              {meeting_url} -> Recall bot streams audio here
  WS   /api/ws/audio     Recall real-time audio -> saved per speaker on disconnect
  GET  /captures         list saved files (JSON)
  GET  /download/{name}  download a capture
  POST /api/enroll       {speaker_id, clip_names?} -> enroll speaker from captures
  GET  /api/enrolled     list enrolled speakers
  DELETE /api/enroll/{speaker} -> remove enrollment
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import re
import secrets
import string
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, field_validator

# Make sibling modules importable (incidents, enroll)
_HERE = Path(__file__).resolve().parent
_sys_path_inserted = str(_HERE)
if _sys_path_inserted not in sys.path:
    sys.path.insert(0, _sys_path_inserted)
import incidents   # incident store + alerting (torch-free)
import enroll      # local speaker enrollment (vendored from service/enroll.py)
os.environ.setdefault("SONAVE_ENROLL_DIR", "/data/enrollments")
os.environ.setdefault("SONAVE_MODEL_CACHE", "/data/models/ecapa")

# --- logging -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("sonave.capture")

# --- config (Railway env vars) ----------------------------------------------
RECALL_API_KEY = os.environ.get("SONAVE_RECALL_API_KEY")
RECALL_BASE = os.environ.get("SONAVE_RECALL_BASE", "https://us-west-2.recall.ai/api/v1")
DATA_DIR = Path(os.environ.get("SONAVE_DATA_DIR", "/data/captured"))
SR = 16_000
# Optional: a hosted detector (e.g. the Modal /score_clip host). When set, each flushed
# chunk is scored there in a background thread and the verdict shows on the page — no
# local GPU / monitor process needed. Unset = capture only (page shows "verdict pending").
SCORER_URL = os.environ.get("SONAVE_SCORER_URL", "").rstrip("/")

# Auth is OPT-IN: unset SONAVE_API_TOKEN => service is open (as before), so deploying
# this is a safe no-op. Set the token and every sensitive endpoint requires it —
# browser via a `sonave_token` cookie, machines via `Authorization: Bearer`/`X-Sonave-Token`,
# the Recall WebSocket via a `?token=` query param.
API_TOKEN = os.environ.get("SONAVE_API_TOKEN", "")
ALLOWED_MEET_HOSTS = ("meet.google.com", "zoom.us", "teams.microsoft.com", "teams.live.com")
_SPK_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _token_ok(v: str | None) -> bool:
    return bool(v) and bool(API_TOKEN) and secrets.compare_digest(v, API_TOKEN)


def require_auth(request: Request):
    """Endpoint guard. No-op when SONAVE_API_TOKEN is unset."""
    if not API_TOKEN:
        return
    auth = request.headers.get("authorization", "")
    bearer = auth[7:] if auth.lower().startswith("bearer ") else ""
    tok = (request.headers.get("x-sonave-token") or bearer
           or request.cookies.get("sonave_token") or request.query_params.get("token"))
    if not _token_ok(tok):
        raise HTTPException(status_code=401, detail="unauthorized")

app = FastAPI(title="Sonave Capture")

# Inline favicon: white audio bars on the brand-blue rounded square (no file needed).
_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="7" fill="#2f6df6"/>'
    '<g fill="#fff">'
    '<rect x="6" y="13" width="3" height="6" rx="1.5"/>'
    '<rect x="11" y="9" width="3" height="14" rx="1.5"/>'
    '<rect x="16" y="5" width="3" height="22" rx="1.5"/>'
    '<rect x="21" y="10" width="3" height="12" rx="1.5"/>'
    '<rect x="26" y="14" width="3" height="4" rx="1.5"/>'
    '</g></svg>'
)
_FAVICON_B64 = base64.b64encode(_FAVICON_SVG.encode()).decode()


def _domain(request: Request | None = None) -> str:
    """Public hostname. Prefer an explicit env override, else the actual request
    Host header (works on Railway with zero config), else Railway's auto var."""
    env = os.environ.get("SONAVE_PUBLIC_DOMAIN") or os.environ.get("RAILWAY_PUBLIC_DOMAIN")
    if env:
        return env
    if request is not None:
        host = request.headers.get("host")
        if host:
            return host
    return ""


def _ws_url(request: Request) -> str:
    d = _domain(request)
    if not d:
        raise RuntimeError("Could not determine public domain from request.")
    return f"wss://{d}/api/ws/audio"


# --- send a bot to a meeting -------------------------------------------------
class BotReq(BaseModel):
    meeting_url: str
    bot_name: str = "Sonave"


@app.post("/bot", dependencies=[Depends(require_auth)])
def send_bot(req: BotReq, request: Request):
    if not RECALL_API_KEY:
        return {"error": "SONAVE_RECALL_API_KEY not set on the service"}
    u = urlparse(req.meeting_url.strip())
    if u.scheme not in ("http", "https") or not any(
            u.netloc == h or u.netloc.endswith("." + h) for h in ALLOWED_MEET_HOSTS):
        return {"ok": False, "detail": "meeting_url must be a Google Meet / Zoom / Teams link"}
    ws = _ws_url(request)
    if API_TOKEN:                       # the Recall bot must authenticate to our WS
        ws = f"{ws}?token={API_TOKEN}"
    payload = {
        "meeting_url": req.meeting_url,
        "bot_name": req.bot_name,
        "recording_config": {
            "audio_separate_raw": {},
            "realtime_endpoints": [
                {"type": "websocket", "url": ws, "events": ["audio_separate_raw.data"]}
            ],
        },
    }
    r = urllib.request.Request(f"{RECALL_BASE}/bot", data=json.dumps(payload).encode(),
                               headers={"Authorization": f"Token {RECALL_API_KEY}",
                                        "Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(r, timeout=20).read())
        bot_id = resp.get("id")
        logger.info("bot created id=%s for %s", bot_id, req.meeting_url)
        return {"ok": True, "bot_id": bot_id, "ws": ws}
    except urllib.error.HTTPError as e:
        logger.error("bot creation failed: HTTP %s", e.code)
        return {"ok": False, "status": e.code, "detail": e.read().decode()[:300]}


# --- real-time audio capture -------------------------------------------------
CHUNK_SEC = 120          # flush each speaker's audio every ~2 min (all-day safe)
_CHUNK_BYTES = CHUNK_SEC * SR * 2

# --- live stream-quality monitoring -----------------------------------------
QUALITY: dict[str, dict] = {}
# authenticity verdicts pushed up from the local GPU scorer (tools/verdict_monitor.py)
VERDICTS: dict[str, dict] = {}
ROLL: dict[str, float] = {}      # speaker -> rolling P(fake), for the hosted scorer
_STATE_LOCK = threading.Lock()   # guards ROLL/VERDICTS across scoring threads

# Real-time scoring: decouple from the 2-min capture-file flush. Score a short sliding
# window every SCORE_SEC so a verdict appears in seconds, not minutes. Capture still
# writes 2-min WAVs for training; only the *scoring* cadence is fast.
SCORE_SEC = int(os.environ.get("SONAVE_SCORE_SEC", "10"))       # cadence between scores
SCORE_WIN_SEC = int(os.environ.get("SONAVE_SCORE_WIN_SEC", "8"))  # window length scored
SCORE_EMA = float(os.environ.get("SONAVE_SCORE_EMA", "0.6"))     # weight on the newest score
_SCORE_HOP_BYTES = SCORE_SEC * SR * 2
_SCORE_WIN_BYTES = SCORE_WIN_SEC * SR * 2


def _quality(spk: str, pcm: bytes):
    """Update rolling audio-quality stats for a speaker from a raw PCM16 chunk.
    Uses stdlib array/math (audioop was removed in Python 3.13)."""
    import array
    import math
    s = array.array("h")
    s.frombytes(pcm if len(pcm) % 2 == 0 else pcm[:-1])
    n = len(s)
    if n == 0:
        return
    peak = max(abs(min(s)), abs(max(s))) / 32768.0
    step = max(1, n // 2000)                       # subsample for cheap RMS
    ss = sum(s[i] * s[i] for i in range(0, n, step))
    rms = math.sqrt(ss / (n // step + 1)) / 32768.0
    sec = n / SR
    q = QUALITY.setdefault(spk, {"level": 0.0, "peak": 0.0, "clips": 0,
                                 "speech_sec": 0.0, "total_sec": 0.0})
    q["level"] = 0.25 * rms + 0.75 * q["level"]       # smoothed current level
    q["peak"] = max(peak, q["peak"] * 0.99)            # decaying peak-hold
    if peak >= 0.99:
        q["clips"] += 1
    if rms > 0.01:
        q["speech_sec"] += sec
    q["total_sec"] += sec


def _quality_verdict(q: dict) -> str:
    if q["total_sec"] < 3:
        return "warming up"
    if q["peak"] >= 0.985 or q["clips"] > 5:
        return "CLIPPING — lower volume"
    if q["level"] < 0.01:
        return "TOO QUIET — raise volume"
    speech = q["speech_sec"] / max(q["total_sec"], 1e-6)
    if speech < 0.2:
        return "mostly silence — is audio playing?"
    return "good"


@app.websocket("/api/ws/audio")
async def ws_audio(ws: WebSocket):
    if API_TOKEN and not _token_ok(ws.query_params.get("token")):
        await ws.close(code=1008)        # policy violation — bot lacked the token
        return
    await ws.accept()
    buffers: dict[str, bytearray] = {}
    idx: dict[str, int] = {}
    seen: dict[str, int] = {}        # cumulative bytes per speaker (drives scoring cadence)
    scored: dict[str, int] = {}      # cumulative bytes at last score
    session = int(time.time())
    msgs = 0
    try:
        while True:
            msg = await ws.receive_text()
            msgs += 1
            try:
                d = (json.loads(msg).get("data") or {}).get("data") or {}
                buf = d.get("buffer")
                if not buf:
                    continue
                # whitelist the speaker name -> safe for use in the capture filename
                spk = _SPK_RE.sub("_", ((d.get("participant") or {}).get("name") or "unknown")).strip("_") or "unknown"
                raw = base64.b64decode(buf)
                b = buffers.setdefault(spk, bytearray())     # CAPTURE FIRST (critical path)
                b.extend(raw)
                if len(b) >= _CHUNK_BYTES:               # periodic flush -> ~2 min training files
                    _write(spk, bytes(b), session, idx.get(spk, 0))
                    idx[spk] = idx.get(spk, 0) + 1
                    b.clear()
                # real-time scoring: every SCORE_SEC, score the last ~SCORE_WIN_SEC off-path
                seen[spk] = seen.get(spk, 0) + len(raw)
                if SCORER_URL and seen[spk] - scored.get(spk, 0) >= _SCORE_HOP_BYTES:
                    scored[spk] = seen[spk]
                    window = _pcm_to_wav(bytes(b[-_SCORE_WIN_BYTES:]))
                    threading.Thread(target=_score_and_store, args=(spk, window), daemon=True).start()
                try:
                    _quality(spk, raw)                   # quality is best-effort, never breaks capture
                except Exception:
                    pass
            except json.JSONDecodeError:
                logger.warning("ws: malformed JSON frame")
            except Exception as exc:
                logger.warning("ws: frame error: %s", repr(exc)[:80])
    except WebSocketDisconnect:
        logger.info("ws: disconnect after %s msgs", msgs)
    except Exception as exc:
        logger.error("ws: unexpected close: %s", repr(exc)[:120])
    finally:
        for spk, b in buffers.items():                   # flush the remainder to disk
            if len(b) >= SR * 2:
                _write(spk, bytes(b), session, idx.get(spk, 0))


def _write(spk: str, pcm: bytes, session: int, idx: int):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"meet_{spk}_{session}_{idx:03d}.wav"
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)            # 16-bit PCM (S16LE, matches Recall)
        w.setframerate(SR)
        w.writeframes(pcm)
    logger.info("saved %.1fs of '%s' -> %s", len(pcm)/2/SR, spk, out)
    return out


def _av(p: float) -> str:
    return "fake" if p >= 0.7 else "suspect" if p >= 0.4 else "real"


def _pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap raw S16LE 16 kHz mono PCM in a WAV container (for POSTing a live window)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    return buf.getvalue()


def _rand_boundary() -> str:
    """Random multipart boundary to avoid collisions."""
    return "----sonave_" + "".join(random.choices(string.ascii_letters + string.digits, k=16))


def _score_and_store(spk: str, wav_bytes: bytes):
    """Best-effort: POST a live window to the hosted scorer (Modal /score_clip) and store
    the rolling verdict. Runs in a daemon thread with bounded retry — never the capture path."""
    if not SCORER_URL:
        return
    try:
        # Load voiceprint if speaker is enrolled
        voiceprint_b64 = None
        try:
            if enroll.is_enrolled(spk):
                vp = np.load(enroll.ENROLL_DIR / f"{spk}.npy")
                voiceprint_b64 = base64.b64encode(vp.tobytes()).decode()
        except Exception:
            pass

        boundary = _rand_boundary()
        body_parts = [
            f'--{boundary}\r\n',
            'Content-Disposition: form-data; name="file"; filename="c.wav"\r\n',
            'Content-Type: audio/wav\r\n\r\n',
        ]
        body = "".join(body_parts).encode() + wav_bytes + b"\r\n"
        if voiceprint_b64:
            body += (
                f'--{boundary}\r\n'
                f'Content-Disposition: form-data; name="voiceprint_b64"\r\n\r\n'
                f'{voiceprint_b64}\r\n'.encode()
            )
        if spk:
            body += (
                f'--{boundary}\r\n'
                f'Content-Disposition: form-data; name="speaker_id"\r\n\r\n'
                f'{spk}\r\n'.encode()
            )
        body += f'--{boundary}--\r\n'.encode()

        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        if API_TOKEN:
            headers["X-Sonave-Token"] = API_TOKEN     # shared secret; Modal validates it
        res = None
        for attempt in range(3):                      # retry 429 / cold-start / transient
            try:
                req = urllib.request.Request(
                    f"{SCORER_URL}/score_clip", data=body, method="POST", headers=headers)
                with urllib.request.urlopen(req, timeout=60) as r:
                    res = json.loads(r.read())
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(0.5 * (2 ** attempt) + random.random() * 0.3)   # backoff + jitter

        # Modal returns fused risk when voiceprint was provided; fall back to p_fake
        p = res.get("risk") if res.get("speaker_check") else res.get("p_fake")
        if p is None:
            return
        with _STATE_LOCK:
            prev = ROLL.get(spk)
            roll = p if prev is None else SCORE_EMA * p + (1 - SCORE_EMA) * prev
            ROLL[spk] = roll
            verdict = _av(roll)
            VERDICTS[spk] = {
                "p_fake": round(res.get("p_fake", p), 3),
                "rolling": round(roll, 3),
                "verdict": verdict,
                "speaker_check": res.get("speaker_check"),
                "match_conf": res.get("match_conf"),
            }
        logger.info("score %s: p_fake=%.3f risk=%.3f rolling=%.3f -> %s (match=%s)",
                    spk, res.get("p_fake", p), p, roll, verdict,
                    res.get("speaker_check", {}).get("match") if res.get("speaker_check") else None)
        if verdict == "fake":                         # sustained deepfake -> open incident + alert
            inc = incidents.record(spk, roll, res.get("model_version", "?"))
            if inc:
                incidents.notify(inc)
    except Exception as e:  # noqa: BLE001 — scoring must never crash capture
        logger.warning("score skip %s: %s", spk, repr(e)[:80])


# --- enrollment endpoints ----------------------------------------------------
class EnrollReq(BaseModel):
    speaker_id: str
    clip_names: list[str] | None = None

    @field_validator("speaker_id")
    @classmethod
    def _spk(cls, v: str) -> str:
        return (_SPK_RE.sub("_", str(v)).strip("_") or "unknown")[:64]


@app.post("/api/enroll", dependencies=[Depends(require_auth)])
def api_enroll(req: EnrollReq):
    """Enroll a speaker from captured clips. If clip_names is omitted, uses all
    captures whose filename contains the speaker name."""
    speaker = req.speaker_id
    if not DATA_DIR.exists():
        return {"ok": False, "detail": "no capture directory"}

    if req.clip_names:
        paths = [DATA_DIR / Path(n).name for n in req.clip_names]
        paths = [p for p in paths if p.exists()]
    else:
        paths = sorted(DATA_DIR.glob(f"meet_{speaker}_*.wav"))

    if len(paths) < 1:
        return {"ok": False, "detail": f"no captures found for '{speaker}'"}

    try:
        vp = enroll.enroll(speaker, paths)
        return {"ok": True, "speaker": speaker, "clips": len(paths),
                "dim": len(vp), "voiceprint_path": str(enroll.ENROLL_DIR / f"{speaker}.npy")}
    except Exception as exc:
        logger.exception("enrollment failed for %s", speaker)
        return {"ok": False, "detail": str(exc)[:200]}


@app.get("/api/enrolled", dependencies=[Depends(require_auth)])
def api_enrolled():
    """List enrolled speakers with file metadata."""
    out = []
    for sid in enroll.list_enrolled():
        f = enroll.ENROLL_DIR / f"{sid}.npy"
        st = f.stat() if f.exists() else None
        out.append({
            "speaker_id": sid,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(st.st_mtime)) if st else None,
            "size_bytes": st.st_size if st else None,
        })
    return {"enrolled": out}


@app.delete("/api/enroll/{speaker}", dependencies=[Depends(require_auth)])
def api_delete_enroll(speaker: str):
    f = enroll.ENROLL_DIR / f"{_SPK_RE.sub('_', speaker).strip('_') or 'unknown'}.npy"
    if f.exists():
        f.unlink()
        return {"ok": True, "detail": "deleted"}
    return {"ok": False, "detail": "not found"}


# --- retrieval ---------------------------------------------------------------
@app.get("/favicon.ico")
@app.get("/favicon.svg")
def favicon():
    from fastapi.responses import Response
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml")


class VerdictReq(BaseModel):
    speaker: str
    p_fake: float
    rolling: float
    verdict: str

    @field_validator("p_fake", "rolling")
    @classmethod
    def _prob(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @field_validator("verdict")
    @classmethod
    def _verdict(cls, v: str) -> str:
        return v if v in ("real", "suspect", "fake") else "suspect"

    @field_validator("speaker")
    @classmethod
    def _spk(cls, v: str) -> str:
        return (_SPK_RE.sub("_", str(v)).strip("_") or "unknown")[:64]


@app.post("/api/verdict", dependencies=[Depends(require_auth)])
def api_verdict(v: VerdictReq):
    """Local GPU scorer pushes authenticity verdicts here; the page shows them."""
    with _STATE_LOCK:
        VERDICTS[v.speaker] = {"p_fake": round(v.p_fake, 3), "rolling": round(v.rolling, 3),
                               "verdict": v.verdict}
    return {"ok": True}


SKIP_SPEAKERS = ("HealthCheck", "FIXCHECK", "WSTEST", "deploycheck")


@app.get("/api/quality", dependencies=[Depends(require_auth)])
def api_quality():
    out = {}
    speakers = set(QUALITY) | set(VERDICTS)
    for spk in speakers:
        if any(s in spk for s in SKIP_SPEAKERS):
            continue
        q = QUALITY.get(spk)
        row = {"verdict": _quality_verdict(q) if q else "—"}
        if q:
            speech = q["speech_sec"] / max(q["total_sec"], 1e-6)
            row.update({"level": round(q["level"], 3), "peak": round(q["peak"], 3),
                        "clips": q["clips"], "speech_pct": round(speech * 100),
                        "total_sec": round(q["total_sec"])})
        av = VERDICTS.get(spk)
        if av:
            row["auth_verdict"] = av["verdict"]
            row["auth_p"] = av["rolling"]
            if av.get("speaker_check"):
                row["speaker_check"] = av["speaker_check"]
                row["match_conf"] = av.get("match_conf")
        # show enrollment status
        row["enrolled"] = enroll.is_enrolled(spk)
        out[spk] = row
    return out


@app.get("/api/incidents", dependencies=[Depends(require_auth)])
def api_incidents():
    return {"incidents": incidents.list_incidents()}


class AckReq(BaseModel):
    id: int


@app.post("/api/incidents/ack", dependencies=[Depends(require_auth)])
def api_ack(a: AckReq):
    return {"ok": incidents.acknowledge(a.id)}


@app.get("/captures", dependencies=[Depends(require_auth)])
def captures():
    if not DATA_DIR.exists():
        return {"files": []}
    fs = sorted(DATA_DIR.glob("*.wav"))
    return {"files": [{"name": f.name, "mb": round(f.stat().st_size / 1e6, 2)} for f in fs]}


@app.get("/download/{name}", dependencies=[Depends(require_auth)])
def download(name: str):
    f = DATA_DIR / Path(name).name          # prevent path traversal
    return FileResponse(str(f)) if f.exists() else {"error": "not found"}


@app.get("/", response_class=HTMLResponse)
def landing():
    html = (_HERE / "landing.html").read_text(encoding="utf-8")
    return html.replace("__FAVICON__", _FAVICON_B64)


@app.get("/console", response_class=HTMLResponse)
def console():
    html = (_HERE / "console.html").read_text(encoding="utf-8")
    return html.replace("__AUTH__", "1" if API_TOKEN else "0").replace("__FAVICON__", _FAVICON_B64)


