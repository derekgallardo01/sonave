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
  GET  /                 dashboard: send a bot, list/download captures, enroll speakers
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

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, field_validator

# Make repo root importable so we can reuse service/enroll.py
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Enrollment persists on Railway's /data volume; model cache too
os.environ.setdefault("SONAVE_ENROLL_DIR", "/data/enrollments")
os.environ.setdefault("SONAVE_MODEL_CACHE", "/data/models/ecapa")

import service.enroll as enroll  # noqa: E402
import numpy as np  # noqa: E402

_sys_path_inserted = str(Path(__file__).resolve().parent)
if _sys_path_inserted not in sys.path:
    sys.path.insert(0, _sys_path_inserted)
import incidents  # incident store + alerting (torch-free)

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
def index(request: Request):
    domain = _domain(request) or "(unknown — no Host header)"
    key = "set" if RECALL_API_KEY else "MISSING — set SONAVE_RECALL_API_KEY"
    return (_PAGE
            .replace("__DOMAIN__", domain)
            .replace("__KEY__", key)
            .replace("__AUTH__", "1" if API_TOKEN else "0")
            .replace("__FAVICON__", _FAVICON_B64))


_PAGE = r"""<!doctype html><html lang=en><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Sonave — Live Voice Authenticity</title>
<link rel="icon" href="data:image/svg+xml;base64,__FAVICON__">
<style>
:root{--bg:#0d111c;--card:#161c2b;--card2:#1b2233;--line:#28324a;--ink:#e9eef8;
--mut:#8b97ad;--dim:#616d84;--blue:#2f6df6;--green:#33c17f;--amber:#eaa021;--red:#ef4a4a}
*{box-sizing:border-box}
body{font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:840px;
margin:0 auto;padding:34px 18px 80px;background:var(--bg);color:var(--ink)}
a{color:#6aa5ff;text-decoration:none}a:hover{text-decoration:underline}
h1{font-size:21px;margin:0;letter-spacing:-.02em;display:flex;align-items:center;gap:9px}
.mark{width:22px;height:22px;border-radius:6px;background:linear-gradient(135deg,#2f6df6,#7a4dff);
display:inline-block;flex:none;box-shadow:0 2px 10px rgba(47,109,246,.4)}
.sub{color:var(--mut);font-size:13px;margin:5px 0 0}
.meta{color:var(--dim);font-size:12px;margin:3px 0 0}.meta code{color:var(--mut)}
.top{display:flex;justify-content:space-between;align-items:flex-start;gap:14px;flex-wrap:wrap}
.live{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--mut);
background:var(--card);border:1px solid var(--line);padding:6px 11px;border-radius:999px;white-space:nowrap}
.dot{width:8px;height:8px;border-radius:50%;background:var(--dim)}
.dot.on{background:var(--green);box-shadow:0 0 0 0 rgba(51,193,127,.6);animation:pulse 1.8s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(51,193,127,.5)}70%{box-shadow:0 0 0 7px rgba(51,193,127,0)}100%{box-shadow:0 0 0 0 rgba(51,193,127,0)}}
.row{display:flex;gap:8px;margin:20px 0 6px}
input,button,select{font:15px system-ui;padding:11px 13px;border-radius:9px;border:1px solid var(--line);
background:var(--card);color:var(--ink);outline:none}
input:focus,select:focus{border-color:var(--blue)}
button{background:var(--blue);border:0;cursor:pointer;font-weight:600}
button:hover{filter:brightness(1.08)}button:disabled{opacity:.6;cursor:default}
button.secondary{background:var(--card2);border:1px solid var(--line)}
.msg{font-size:13px;color:var(--mut);min-height:18px;margin:2px 0 0}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.09em;color:var(--dim);
margin:30px 0 4px;font-weight:700}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--mut);margin:0 0 12px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.pip{width:9px;height:9px;border-radius:3px;display:inline-block}
.card{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:15px 17px;margin:10px 0}
.chead{display:flex;justify-content:space-between;align-items:center;gap:12px}
.who{font-weight:700;font-size:16px}
.chip{color:#fff;padding:5px 12px;border-radius:8px;font-weight:800;font-size:13px;letter-spacing:.02em;white-space:nowrap}
.chip small{font-weight:600;opacity:.85;font-size:11px;margin-left:5px}
.alert{display:flex;justify-content:space-between;align-items:center;gap:12px;
background:rgba(239,74,74,.12);border:1px solid var(--red);border-radius:11px;padding:12px 15px;margin:10px 0}
.alert b{color:var(--red)}.alert .btn{background:var(--red);padding:8px 13px}
.pend{color:var(--dim);font-size:12px;font-style:italic}
.bar{height:7px;background:#0f1524;border-radius:5px;margin:11px 0 8px;overflow:hidden}
.bar i{display:block;height:100%;transition:width .3s}
.det{color:var(--mut);font-size:12.5px}.det b{color:var(--ink);font-weight:600}
.empty{color:var(--dim);font-size:13px;padding:18px 0}
.sess{border:1px solid var(--line);border-radius:11px;margin:9px 0;overflow:hidden;background:var(--card)}
.shead{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:11px 14px;
cursor:pointer;user-select:none}.shead:hover{background:var(--card2)}
.stitle{font-weight:600;font-size:14px}.smeta{color:var(--dim);font-size:12px}
.sbody{border-top:1px solid var(--line)}
.clip{display:flex;align-items:center;gap:10px;padding:8px 14px;font-size:13px;border-top:1px solid #202839}
.clip:first-child{border-top:0}
.play{width:28px;height:28px;flex:none;border-radius:50%;background:var(--card2);border:1px solid var(--line);
color:var(--ink);cursor:pointer;font-size:11px;padding:0;line-height:1}
.play:hover{background:var(--blue);border-color:var(--blue)}
.cname{color:var(--mut);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cdur{color:var(--dim);font-size:12px;font-variant-numeric:tabular-nums}
.caret{color:var(--dim);transition:transform .15s}.caret.open{transform:rotate(90deg)}
.enroll-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0}
.enroll-row select{flex:1;min-width:140px}
.enroll-row button{padding:9px 14px;font-size:13px}
.enrolled-list{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 0}
.enrolled-tag{display:flex;align-items:center;gap:6px;background:var(--card2);border:1px solid var(--line);
border-radius:8px;padding:6px 12px;font-size:13px}
.enrolled-tag button{background:transparent;border:0;color:var(--red);font-size:15px;padding:0 2px;cursor:pointer}
</style>
<div class=top>
 <div><h1><span class=mark></span>Sonave</h1>
  <p class=sub>Live deepfake-voice monitoring for meetings</p>
  <p class=meta>domain <code>__DOMAIN__</code> · Recall key <code>__KEY__</code></p></div>
 <div class=live><span id=dot class=dot></span><span id=livetxt>idle</span></div>
</div>
<div class=row><input id=u placeholder="paste a Google Meet / Zoom link" style=flex:1>
<button id=sendb onclick=send()>Send bot</button></div>
<p class=msg id=msg></p>

<div id=alerts></div>

<h2>Speaker enrollment</h2>
<div class=card>
 <div class=enroll-row>
  <select id=enrollSpk><option value="">— select speaker from captures —</option></select>
  <button onclick=enrollSpk()>Enroll selected</button>
  <button class=secondary onclick=enrollCustom()>Enroll by name</button>
 </div>
 <p class=msg id=enrollMsg></p>
 <div class=enrolled-list id=enrolledList></div>
</div>

<h2>Live authenticity</h2>
<div class=legend>
 <span><i class=pip style=background:var(--green)></i>REAL &lt;0.40</span>
 <span><i class=pip style=background:var(--amber)></i>SUSPECT 0.40–0.70</span>
 <span><i class=pip style=background:var(--red)></i>FAKE ≥0.70</span>
</div>
<div id=quality><div class=empty>Waiting for audio… send a bot into a meeting to begin.</div></div>

<h2>Captures</h2>
<div id=list><div class=empty>No captures yet.</div></div>

<div id=login style="display:none;position:fixed;inset:0;background:rgba(8,11,18,.94);z-index:99;align-items:center;justify-content:center">
 <div style="background:var(--card);border:1px solid var(--line);border-radius:14px;padding:26px 28px;max-width:340px;width:90%">
  <h1><span class=mark></span>Sonave</h1>
  <p class=sub style="margin:6px 0 14px">Enter the access token to continue.</p>
  <input id=tok type=password placeholder="access token" style="width:100%;margin-bottom:10px" onkeydown="if(event.key=='Enter')dologin()">
  <button class=btn style="width:100%" onclick=dologin()>Unlock</button>
 </div>
</div>

<script>
var SKIP=/HealthCheck|FIXCHECK|WSTEST|deploycheck/i, open={}, lastAudio=null;
var AUTH='__AUTH__'==='1';
function cookie(n){return (document.cookie.match('(^|;)\\s*'+n+'\\s*=\\s*([^;]+)')||[])[2]}
function esc(s){return String(s==null?'':s).replace(/[&<>"']/g,function(c){
 return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]})}
function needlogin(){document.getElementById('login').style.display='flex'}
async function jget(u){var r=await fetch(u);if(r.status===401){needlogin();throw 'auth'}return r.json()}
function dologin(){var t=document.getElementById('tok').value.trim();if(!t)return;
 document.cookie='sonave_token='+t+';path=/;SameSite=Strict;Max-Age=2592000';
 document.getElementById('login').style.display='none';start()}
async function incidents(){
 var d;try{d=await jget('/api/incidents')}catch(e){return}
 var open=(d.incidents||[]).filter(function(i){return i.status=='open'});
 document.getElementById('alerts').innerHTML=open.map(function(i){
  return '<div class=alert><div><b>⛔ '+(i.hold?'WIRE HELD':'ALERT')+'</b> — suspected deepfake voice: <b>'
   +esc(i.speaker)+'</b> · risk '+Math.round(i.rolling*100)+'%'
   +(i.hold?' · re-authenticate the caller before releasing funds':'')+'</div>'
   +'<button class=btn onclick="ack('+i.id+')">Acknowledge</button></div>'}).join('')
}
async function ack(id){
 await fetch('/api/incidents/ack',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({id:id})});incidents()
}
var _started=0;
function start(){if(_started)return;_started=1;
 list();quality();incidents();enrolled();
 setInterval(function(){list()},5000);setInterval(quality,1500);setInterval(incidents,3000);setInterval(enrolled,10000)}
function avcolor(v){return v=='real'?'var(--green)':v=='fake'?'var(--red)':'var(--amber)'}
function qcolor(v){return v=='good'?'var(--green)':/CLIP/.test(v)?'var(--red)':/QUIET|silence/i.test(v)?'var(--amber)':'var(--dim)'}
function hms(s){s=Math.round(s);var h=s/3600|0,m=(s%3600)/60|0,x=s%60;
 return h?h+'h '+m+'m':m?m+'m '+(x<10?'0':'')+x+'s':x+'s'}
function tod(ms){var d=new Date(ms);var h=d.getHours(),m=d.getMinutes();var ap=h<12?'AM':'PM';
 h=h%12||12;return h+':'+(m<10?'0':'')+m+' '+ap}

async function quality(){
 var d;try{d=await jget('/api/quality')}catch(e){return}
 var ks=Object.keys(d).filter(k=>!SKIP.test(k)),active=0;
 if(!ks.length){document.getElementById('quality').innerHTML=
  '<div class=empty>Waiting for audio… send a bot into a meeting to begin.</div>';setlive(0);return}
 document.getElementById('quality').innerHTML=ks.map(k=>{
  var s=d[k],lv=Math.min(100,Math.round((s.level||0)*300)),c=qcolor(s.verdict);
  if(s.level>0.01)active++;
  var chip=s.auth_verdict
   ?'<span class=chip style=background:'+avcolor(s.auth_verdict)+'>'+esc(s.auth_verdict.toUpperCase())
     +(s.auth_p!=null?' '+esc(s.auth_p):'')+'<small>authenticity'+(s.enrolled?' · enrolled':'')+'</small></span>'
   :'<span class=pend>scoring…</span>';
  var det=s.level!=null
   ?'<div class=bar><i style="width:'+lv+'%;background:'+c+'"></i></div>'
    +'<div class=det>audio <b>'+s.verdict.toUpperCase()+'</b> · '+s.speech_pct+'% speech · '
    +hms(s.total_sec)+' · '+s.clips+' clip'+(s.clips==1?'':'s')
    +(s.peak>=0.985?' · <span style=color:var(--red)>⚠ clipping</span>':'')
    +(s.speaker_check&&s.speaker_check.enrolled?' · voiceprint match '+Math.round(s.speaker_check.similarity*100)+'%':'')
    +'</div>':'';
  return '<div class=card><div class=chead><span class=who>'+esc(k)+'</span>'+chip+'</div>'+det+'</div>'
 }).join('');
 setlive(active)
}
function setlive(n){var on=n>0;document.getElementById('dot').className='dot'+(on?' on':'');
 document.getElementById('livetxt').textContent=on?(n+' active'):'idle'}

// --- enrollment UI ---
var _captureSpeakers=new Set();
async function enrolled(){
 var d;try{d=await jget('/api/enrolled')}catch(e){return}
 var list=document.getElementById('enrolledList');
 list.innerHTML=(d.enrolled||[]).map(function(e){
  return '<span class=enrolled-tag><b>'+esc(e.speaker_id)+'</b> · '+e.created_at
   +' <button onclick="delEnroll(\''+esc(e.speaker_id)+'\')">×</button></span>'
 }).join('');
}
async function delEnroll(spk){
 if(!confirm('Delete enrollment for '+spk+'?'))return;
 await fetch('/api/enroll/'+encodeURIComponent(spk),{method:'DELETE'});
 enrolled();
}
async function enrollSpk(){
 var sel=document.getElementById('enrollSpk'),spk=sel.value;
 if(!spk){document.getElementById('enrollMsg').textContent='select a speaker first';return}
 document.getElementById('enrollMsg').textContent='enrolling…';
 var r=await fetch('/api/enroll',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({speaker_id:spk})});
 var d=await r.json();
 document.getElementById('enrollMsg').textContent=d.ok?('✓ enrolled '+esc(d.speaker)+' from '+d.clips+' clips'):('✕ '+esc(d.detail));
 enrolled();
}
async function enrollCustom(){
 var spk=prompt('Speaker name to enroll (uses all captures with this name):');
 if(!spk)return;
 document.getElementById('enrollMsg').textContent='enrolling…';
 var r=await fetch('/api/enroll',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({speaker_id:spk})});
 var d=await r.json();
 document.getElementById('enrollMsg').textContent=d.ok?('✓ enrolled '+esc(d.speaker)+' from '+d.clips+' clips'):('✕ '+esc(d.detail));
 enrolled();
}
function updateEnrollDropdown(files){
 var sel=document.getElementById('enrollSpk');
 var speakers=new Set();
 files.forEach(function(f){
  var m=f.name.match(/^meet_([A-Za-z0-9_]+)_/);
  if(m)speakers.add(m[1]);
 });
 _captureSpeakers=speakers;
 var opts='<option value="">— select speaker from captures —</option>';
 speakers.forEach(function(s){opts+='<option value="'+esc(s)+'">'+esc(s)+'</option>'});
 sel.innerHTML=opts;
}

function play(name,btn){
 if(lastAudio){lastAudio.pause();if(lastAudio._b)lastAudio._b.textContent='▶'}
 var a=new Audio('/download/'+name);a._b=btn;a.play();btn.textContent='❚❚';lastAudio=a;
 a.onended=function(){btn.textContent='▶';if(lastAudio==a)lastAudio=null}
}
function toggle(id){open[id]=!open[id];list(true)}
var _cache=null;
async function list(fromToggle){
 if(!fromToggle){try{_cache=(await jget('/captures')).files}catch(e){return}}
 var files=(_cache||[]).filter(f=>!SKIP.test(f.name));
 if(!files.length){document.getElementById('list').innerHTML='<div class=empty>No captures yet.</div>';return}
 updateEnrollDropdown(files);
 var sess={};
 files.forEach(f=>{var p=f.name.replace(/\.wav$/,'').split('_');
  var ts=p.length>=3?+p[p.length-2]:0, spk=p.slice(1,-2).join('_')||'?';
  var key=spk+'@'+ts;(sess[key]=sess[key]||{spk:spk,ts:ts,items:[]}).items.push(f)});
 var groups=Object.values(sess).sort((a,b)=>b.ts-a.ts);
 document.getElementById('list').innerHTML=groups.map((g,i)=>{
  var id=g.spk+g.ts, mb=g.items.reduce((s,f)=>s+f.mb,0), dur=mb*1e6/32000;
  var isopen=i in open?open[id]:(i==0);open[id]=isopen;
  var body=isopen?'<div class=sbody>'+g.items.map(f=>{
   var d=f.mb*1e6/32000;
   return '<div class=clip><button class=play onclick="play(\''+esc(f.name)+'\',this)">▶</button>'
    +'<span class=cname>'+esc(f.name)+'</span><span class=cdur>'+hms(d)+' · '+f.mb+' MB</span></div>'
  }).join('')+'</div>':'';
  return '<div class=sess><div class=shead onclick="toggle(\''+esc(id)+'\')">'
   +'<span class=stitle><span class="caret'+(isopen?' open':'')+'">▸</span> '+esc(g.spk)
   +' · '+(g.ts?tod(g.ts*1000):'session')+'</span>'
   +'<span class=smeta>'+g.items.length+' clips · '+mb.toFixed(1)+' MB · '+hms(dur)+'</span></div>'
   +body+'</div>'
 }).join('')
}

async function send(){
 var b=document.getElementById('sendb'),u=document.getElementById('u').value.trim();
 if(!u){document.getElementById('msg').textContent='paste a meeting link first';return}
 b.disabled=true;b.textContent='sending…';
 try{var r=await fetch('/bot',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({meeting_url:u})});
  if(r.status===401){needlogin();b.disabled=false;b.textContent='Send bot';return}
  var d=await r.json();
  document.getElementById('msg').textContent=d.ok?('✓ bot sent · '+d.bot_id):('✕ '+esc(d.detail||d.error||d.status))}
 catch(e){document.getElementById('msg').textContent='✕ '+e}
 b.disabled=false;b.textContent='Send bot';list()
}
if(AUTH && !cookie('sonave_token')) needlogin(); else start();
</script>"""
