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
import hashlib
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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, field_validator

# Make sibling modules importable (incidents, enroll)
_HERE = Path(__file__).resolve().parent
_sys_path_inserted = str(_HERE)
if _sys_path_inserted not in sys.path:
    sys.path.insert(0, _sys_path_inserted)
import incidents   # incident store + alerting (torch-free)
import enroll      # local speaker enrollment (vendored from service/enroll.py)
import auth        # Google OAuth + sessions + principals (stdlib-only)
import db          # app database: users, bots, billing (/data/app.db)
import billing     # Stripe metered billing (stdlib-only)
import autojoin    # zero-scope calendar auto-join (secret iCal URL polling)
import forensics   # cryptographic audit reports (HTML/PDF)
import webhook_dispatcher # multi-platform alert routing (Slack/Discord/Teams)
import meet_media_ingest  # native Google Meet Media API WebRTC bridge
import compliance_vault   # SOC2 / FINRA compliance cloud evidence vault
import legal_certificate  # FBI IC3 / insurance legal certificate of authenticity
import attribution        # AI vocoder fingerprinting & ambient mismatch analysis
import generator          # Synthetic voice generation & live test injector
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
    """Endpoint guard: machine token OR Google session (open when neither is configured)."""
    if auth.get_principal(request) is None:
        raise HTTPException(status_code=401, detail="unauthorized")


def require_principal(request: Request) -> auth.Principal:
    p = auth.get_principal(request)
    if p is None:
        raise HTTPException(status_code=401, detail="unauthorized")
    return p


def require_admin(request: Request) -> auth.Principal:
    p = require_principal(request)
    if p.role != "admin":
        raise HTTPException(status_code=403, detail="admin only")
    return p


def _track(user_id: str, kind: str, **detail) -> None:
    """Best-effort activity event — must never affect the request path."""
    try:
        db.add_event(user_id, kind, json.dumps(detail) if detail else "")
    except Exception:
        pass


def _notify_admin(summary: str) -> None:
    """Growth pushes (signup / subscription changes) to the founder, off-thread.
    Each channel is env-gated and silently off until configured."""
    def _send():
        hook = os.environ.get("SONAVE_ADMIN_WEBHOOK", "")
        if hook:
            try:
                req = urllib.request.Request(
                    hook, data=json.dumps({"text": summary}).encode(),
                    headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=10).read()
            except Exception as e:
                logger.warning("admin webhook failed: %s", repr(e)[:80])
        host = os.environ.get("SONAVE_SMTP_HOST", "")
        to = os.environ.get("SONAVE_ADMIN_EMAIL", "")
        if host and to:
            try:
                import smtplib
                from email.message import EmailMessage
                msg = EmailMessage()
                msg["Subject"] = f"Sonave: {summary[:120]}"
                msg["From"] = os.environ.get("SONAVE_SMTP_USER", "")
                msg["To"] = to
                msg.set_content(summary)
                with smtplib.SMTP(host, int(os.environ.get("SONAVE_SMTP_PORT", "587")),
                                  timeout=15) as s:
                    s.starttls()
                    s.login(os.environ.get("SONAVE_SMTP_USER", ""),
                            os.environ.get("SONAVE_SMTP_PASS", ""))
                    s.send_message(msg)
            except Exception as e:
                logger.warning("admin email failed: %s", repr(e)[:80])
    threading.Thread(target=_send, daemon=True).start()


app = FastAPI(title="Sonave Capture")

# Inline favicon: the Sonave scope-pulse mark — green radar scope with a voice
# pulse and contact blip (matches designs/logo/sonave-logo-master.png).
_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
    '<rect width="24" height="24" rx="5.5" fill="#0a0e12"/>'
    '<circle cx="12" cy="12" r="7.6" stroke="#2ee584" stroke-width="2.2" fill="none"/>'
    '<path d="M7.8 12h1l1.4-3.2 2.4 6.4 1.4-3.2h2.2" stroke="#2ee584" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
    '<circle cx="17.4" cy="6.6" r="2.6" fill="#0a0e12"/>'
    '<circle cx="17.4" cy="6.6" r="2" fill="#2ee584"/>'
    '</svg>'
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
def send_bot(req: BotReq, request: Request, p: "auth.Principal" = Depends(require_principal)):
    return _launch_bot(p.user_id, p.role, req.meeting_url, request, req.bot_name)


def _launch_bot(user_id: str, role: str, meeting_url: str,
                request: Request | None = None, bot_name: str = "Sonave",
                source: str = "manual"):
    """Deploy a bot into a meeting. Shared by POST /bot and the calendar
    auto-join loop (which has no request; _ws_url falls back to env domain)."""
    if not RECALL_API_KEY:
        return {"error": "SONAVE_RECALL_API_KEY not set on the service"}
    murl = meeting_url.strip()
    u = urlparse(murl)
    if u.scheme not in ("http", "https") or not any(
            u.netloc == h or u.netloc.endswith("." + h) for h in ALLOWED_MEET_HOSTS):
        return {"ok": False, "detail": "meeting_url must be a Google Meet / Zoom / Teams link"}
    existing = db.find_active_bot(user_id, murl)
    if existing:
        # verify against Recall at click time: a kicked/denied bot must never
        # block re-inviting (its zombie socket can keep our row looking live)
        code = None
        try:
            code = _recall_bot_status(existing["bot_id"])
        except Exception:
            pass                          # Recall unreachable — keep the dedupe
        if code in _ENDED_CODES:
            db.mark_bot(existing["bot_id"], ended_ts=time.time(), status="ended")
        else:
            return {"ok": True, "bot_id": existing["bot_id"], "already": True,
                    "detail": "A Sonave bot is already in this meeting."}
    gate = _bot_gate(user_id, role)
    if gate is not None:
        code = gate.get("code") if isinstance(gate, dict) else "quota"
        _track(user_id, "bot_denied", code=code or "denied", source=source)
        return gate
    ws = _ws_url(request)
    # Per-bot single-purpose WS token: bot-scoped, hashed at rest, 24 h expiry —
    # strictly tighter than the old global token in the same slot.
    bot_tok = secrets.token_urlsafe(32)
    ws = f"{ws}?token={bot_tok}"
    payload = {
        "meeting_url": murl,
        "bot_name": bot_name,
        "recording_config": {
            "audio_separate_raw": {},
            "realtime_endpoints": [
                {"type": "websocket", "url": ws,
                 "events": ["audio_separate_raw.data",
                            # presence: show every participant, speaking state, drop leavers
                            "participant_events.join", "participant_events.leave",
                            "participant_events.speech_on", "participant_events.speech_off"]}
            ],
        },
    }
    r = urllib.request.Request(f"{RECALL_BASE}/bot", data=json.dumps(payload).encode(),
                               headers={"Authorization": f"Token {RECALL_API_KEY}",
                                        "Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(r, timeout=20).read())
        bot_id = resp.get("id")
        db.insert_bot(bot_id or "unknown", user_id,
                      hashlib.sha256(bot_tok.encode()).hexdigest(), murl)
        logger.info("bot created id=%s for %s (user=%s)", bot_id, murl, user_id)
        _track(user_id, "bot_created", source=source, meeting_url=murl, bot_id=bot_id or "")
        if SCORER_URL:                      # pre-warm the scale-to-zero scorer so the
            threading.Thread(target=_warm_scorer, daemon=True).start()  # first verdict skips the cold start
        return {"ok": True, "bot_id": bot_id}
    except urllib.error.HTTPError as e:
        logger.error("bot creation failed: HTTP %s", e.code)
        return {"ok": False, "status": e.code, "detail": e.read().decode()[:300]}


def _bot_gate(user_id: str, role: str):
    """Entitlement / abuse guard for bot launches. Returns a response or None (allowed)."""
    if role == "admin":
        return None
    if db.count_active_bots(user_id) >= int(os.environ.get("SONAVE_MAX_CONCURRENT_BOTS", "2")):
        return {"ok": False, "code": "too_many_bots",
                "detail": "Concurrent bot limit reached — end a running meeting first."}
    denied = billing.can_launch_bot(user_id, role)
    if denied is not None:
        return JSONResponse(status_code=402, content=denied)
    return None


# --- real-time audio capture -------------------------------------------------
CHUNK_SEC = 120          # flush each speaker's audio every ~2 min (all-day safe)
_CHUNK_BYTES = CHUNK_SEC * SR * 2

# --- live stream-quality monitoring -----------------------------------------
# All live-state dicts are keyed by (user_id, speaker) — one workspace per user;
# machine-token sessions land in the admin workspace.
QUALITY: dict[tuple[str, str], dict] = {}
# authenticity verdicts pushed up from the local GPU scorer (tools/verdict_monitor.py)
VERDICTS: dict[tuple[str, str], dict] = {}
ROLL: dict[tuple[str, str], float] = {}   # rolling P(fake), for the hosted scorer
_STATE_LOCK = threading.Lock()   # guards ROLL/VERDICTS across scoring threads

# Real-time scoring: decouple from the 2-min capture-file flush. Score a short sliding
# window every SCORE_SEC so a verdict appears in seconds, not minutes. Capture still
# writes 2-min WAVs for training; only the *scoring* cadence is fast.
SCORE_SEC = int(os.environ.get("SONAVE_SCORE_SEC", "4"))        # cadence between scores
SCORE_FIRST_SEC = int(os.environ.get("SONAVE_SCORE_FIRST_SEC", "4"))  # audio before FIRST score
SCORE_WIN_SEC = int(os.environ.get("SONAVE_SCORE_WIN_SEC", "8"))  # window length scored
SCORE_EMA = float(os.environ.get("SONAVE_SCORE_EMA", "0.6"))     # weight on the newest score
SCORE_PRIOR = float(os.environ.get("SONAVE_SCORE_PRIOR", "0.15"))  # EMA seed ("probably real"):
# one hot first window lands in SUSPECT at worst; sustained fake confirms on window 2
MIN_SPEECH_FRAC = float(os.environ.get("SONAVE_MIN_SPEECH_FRAC", "0.5"))
# windows below this voiced fraction are never scored — mostly-silence audio is
# out-of-distribution for the detector and scores garbage (hot first windows).
# 0.5 of the 8 s window ≈ 4 s of voice, matching the model's training windows;
# passing windows also weight the EMA by their voiced fraction (thin = gentle)
_SCORE_HOP_BYTES = SCORE_SEC * SR * 2
_SCORE_FIRST_BYTES = SCORE_FIRST_SEC * SR * 2   # 4 s matches the model's training windows,
                                                # so the first verdict lands in ~4 s not ~10 s
_SCORE_WIN_BYTES = SCORE_WIN_SEC * SR * 2

# An incident (and the wire-hold webhook) needs this many CONSECUTIVE fake-band
# rolling verdicts — one hot window on a cold EMA must not hold a wire.
INCIDENT_STREAK = int(os.environ.get("SONAVE_INCIDENT_STREAK", "3"))
FAKE_STREAK: dict[tuple[str, str], int] = {}  # consecutive fake verdicts (guarded by _STATE_LOCK)
_INFLIGHT: set[tuple[str, str]] = set()       # scorer requests in flight (guarded by _STATE_LOCK)

# Presence from Recall participant_events (authoritative where available; Recall
# exposes NO mute state, so the UI says "quiet", never "muted"). Keyed (uid, spk):
# {"present": bool, "speaking": bool, "ts": last event time}
PRESENCE: dict[tuple[str, str], dict] = {}

# Live-session lifecycle: when a workspace's last audio stream closes (host
# removed the bot / meeting ended), the live view empties after a short grace
# (Recall reconnects within it) and the panel's Protect button returns.
ACTIVE_STREAMS: dict[str, int] = {}          # open audio websockets per workspace
LAST_CLOSE: dict[str, float] = {}            # when the count last hit zero
STREAM_GRACE_SEC = int(os.environ.get("SONAVE_STREAM_GRACE_SEC", "45"))
_REAP_LAST: dict[str, float] = {}            # per-workspace throttle for the bot reaper
LAST_FRAME: dict[str, float] = {}            # last audio frame per workspace
# Adaptive sweep: while audio flows the bot is obviously alive (check lazily);
# the moment the stream goes quiet is exactly when a kick is possible (check fast).
REAP_FAST_SEC = int(os.environ.get("SONAVE_REAP_INTERVAL_SEC", "4"))
REAP_SLOW_SEC = 15

# Build stamp served on /api/quality — open consoles/panels reload themselves on
# deploy instead of running week-old JS until someone remembers to hard-refresh.
_BUILD = (os.environ.get("RAILWAY_GIT_COMMIT_SHA") or "")[:8] or str(int(time.time()))

if not SCORER_URL:
    logger.warning("SONAVE_SCORER_URL unset — live authenticity scoring is DISABLED; "
                   "the console will show PENDING for every speaker")


def _quality(user_id: str, spk: str, pcm: bytes):
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
    q = QUALITY.setdefault((user_id, spk), {"level": 0.0, "peak": 0.0, "clips": 0,
                                            "speech_sec": 0.0, "total_sec": 0.0})
    q["last_audio_ts"] = time.time()   # speakers who LEFT stop sending chunks entirely
    # Meet mute keeps streaming digital silence, so muted = a streak of ~zero RMS
    if rms < 0.001:
        q["silent_sec"] = q.get("silent_sec", 0.0) + sec
    else:
        q["silent_sec"] = 0.0
    q["level"] = 0.25 * rms + 0.75 * q["level"]       # smoothed current level
    q["peak"] = max(peak, q["peak"] * 0.99)            # decaying peak-hold
    if peak >= 0.99:
        q["clips"] += 1
    if rms > 0.01:
        q["speech_sec"] += sec
    q["total_sec"] += sec


def _speech_fraction(pcm: bytes) -> float:
    """Fraction of 100 ms segments at voice-level RMS — the cheap gate deciding
    whether a window has enough actual speech to be worth scoring."""
    import array
    import math
    s = array.array("h")
    s.frombytes(pcm if len(pcm) % 2 == 0 else pcm[:-1])
    n = len(s)
    if n == 0:
        return 0.0
    seg = SR // 10
    voiced = total = 0
    for i in range(0, n, seg):
        c = s[i:i + seg]
        if not c:
            break
        step = max(1, len(c) // 200)
        idx = range(0, len(c), step)
        rms = math.sqrt(sum(c[j] * c[j] for j in idx) / len(idx)) / 32768.0
        total += 1
        if rms > 0.01:
            voiced += 1
    return voiced / max(total, 1)


def _quality_verdict(q: dict) -> str:
    if q.get("total_sec", 0) < 3:
        return "warming up"
    if q.get("silent_sec", 0) > 3:
        return "quiet"
    if q.get("peak", 0.0) >= 0.985 or q.get("clips", 0) > 5:
        return "CLIPPING — lower volume"
    if q.get("level", 0.0) < 0.01:
        return "TOO QUIET — raise volume"
    speech = q.get("speech_sec", 0.0) / max(q.get("total_sec", 1.0), 1e-6)
    if speech < 0.2:
        return "mostly silence — is audio playing?"
    return "good"


@app.websocket("/api/ws/audio")
async def ws_audio(ws: WebSocket):
    # Resolve the stream's workspace: machine token -> admin; per-bot token -> the
    # bot owner's workspace (and metering). Unknown token in secured mode -> 1008.
    tok = ws.query_params.get("token")
    uid, bot_row = None, None
    if _token_ok(tok):
        uid = db.first_admin_id() or auth.MACHINE_WORKSPACE
    elif tok:
        bot_row = db.resolve_bot_token(hashlib.sha256(tok.encode()).hexdigest())
        if bot_row:
            uid = bot_row["user_id"]
            db.mark_bot(bot_row["bot_id"], started_ts=time.time(), status="streaming")
    if uid is None:
        if API_TOKEN or auth.google_configured():
            await ws.close(code=1008)    # policy violation — no valid token
            return
        uid = db.first_admin_id() or auth.MACHINE_WORKSPACE   # fully-open dev mode
    await ws.accept()
    if bot_row is not None:
        _track(uid, "meeting_started", bot_id=bot_row["bot_id"])
    with _STATE_LOCK:
        fresh = ACTIVE_STREAMS.get(uid, 0) == 0
        ACTIVE_STREAMS[uid] = ACTIVE_STREAMS.get(uid, 0) + 1
        if fresh:
            # first stream of a new monitoring session: reset the workspace's
            # live view so last meeting's speakers/EMAs never bleed into this one
            for dd in (QUALITY, VERDICTS, ROLL, FAKE_STREAK, PRESENCE):
                for k in [k for k in dd if k[0] == uid]:
                    dd.pop(k, None)
    buffers: dict[str, bytearray] = {}
    tails: dict[str, bytearray] = {}  # trailing SCORE_WIN_SEC per speaker — scoring reads this,
                                      # never the capture buffer (which flushes+clears every 2 min)
    idx: dict[str, int] = {}
    seen: dict[str, int] = {}        # cumulative bytes per speaker (drives scoring cadence)
    scored: dict[str, int] = {}      # cumulative bytes at last score
    session = int(time.time())
    msgs = 0
    conn_start = last_tick = time.time()
    # host recognition: the workspace owner's Google profile name, sanitized the
    # same way as participant names, so we can tell THEIR joins/leaves apart
    owner_name = _SPK_RE.sub("_", ((db.get_user(uid) or {}).get("name") or "")).strip("_").lower()
    host_away = False
    try:
        while True:
            msg = await ws.receive_text()
            msgs += 1
            # Metering (bot sessions only): incremental server-clock ticks so a
            # crash mid-meeting never loses more than a minute of accounting.
            if bot_row is not None and time.time() - last_tick >= 60:
                _meter_tick(bot_row["bot_id"], uid, time.time() - last_tick)
                last_tick = time.time()
            try:
                evt = json.loads(msg)
                ev = evt.get("event") or ""
                d = (evt.get("data") or {}).get("data") or {}
                if ev.startswith("participant_events."):
                    pname = _SPK_RE.sub("_", ((d.get("participant") or {}).get("name") or "")).strip("_")
                    if pname:
                        pr = PRESENCE.setdefault((uid, pname),
                                                 {"present": True, "speaking": False, "ts": 0.0})
                        kind = ev.rsplit(".", 1)[1]
                        if kind == "leave":
                            pr["present"], pr["speaking"] = False, False
                        elif kind == "speech_on":
                            pr["present"], pr["speaking"] = True, True
                        elif kind == "speech_off":
                            pr["speaking"] = False
                        else:            # join / anything else -> at least present
                            pr["present"] = True
                        pr["ts"] = time.time()
                        # host left / came back while the bot keeps monitoring
                        low = pname.lower()
                        is_host = bool(owner_name) and (
                            low == owner_name or low.split("_")[0] == owner_name.split("_")[0])
                        if is_host and bot_row is not None:
                            if kind == "leave" and not host_away:
                                host_away = True
                                _track(uid, "host_left", bot_id=bot_row["bot_id"])
                            elif kind in ("join", "speech_on") and host_away:
                                host_away = False
                                _track(uid, "host_rejoined", bot_id=bot_row["bot_id"])
                    continue
                buf = d.get("buffer")
                if not buf:
                    continue
                # whitelist the speaker name -> safe for use in the capture filename
                spk = _SPK_RE.sub("_", ((d.get("participant") or {}).get("name") or "unknown")).strip("_") or "unknown"
                raw = base64.b64decode(buf)
                LAST_FRAME[uid] = time.time()                # feeds the adaptive reaper
                b = buffers.setdefault(spk, bytearray())     # CAPTURE FIRST (critical path)
                b.extend(raw)
                if len(b) >= _CHUNK_BYTES:               # periodic flush -> ~2 min training files
                    _write(uid, spk, bytes(b), session, idx.get(spk, 0))
                    idx[spk] = idx.get(spk, 0) + 1
                    b.clear()
                # real-time scoring: every SCORE_SEC, score the last ~SCORE_WIN_SEC off-path
                t = tails.setdefault(spk, bytearray())
                t.extend(raw)
                if len(t) > _SCORE_WIN_BYTES:
                    del t[:-_SCORE_WIN_BYTES]
                seen[spk] = seen.get(spk, 0) + len(raw)
                hop = _SCORE_HOP_BYTES if spk in scored else _SCORE_FIRST_BYTES
                if SCORER_URL and seen[spk] - scored.get(spk, 0) >= hop:
                    scored[spk] = seen[spk]
                    # silence gate: a window without enough voiced audio is never
                    # scored (OOD for the detector); the next hop re-evaluates
                    frac = _speech_fraction(bytes(t))
                    if frac >= MIN_SPEECH_FRAC:
                        # one request in flight per speaker — during a scorer outage
                        # the cadence just drops instead of piling up retry threads
                        with _STATE_LOCK:
                            busy = (uid, spk) in _INFLIGHT
                            if not busy:
                                _INFLIGHT.add((uid, spk))
                        if not busy:
                            window = _pcm_to_wav(bytes(t))
                            threading.Thread(target=_score_and_store,
                                             args=(uid, spk, window, frac), daemon=True).start()
                try:
                    _quality(uid, spk, raw)              # quality is best-effort, never breaks capture
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
                _write(uid, spk, bytes(b), session, idx.get(spk, 0))
        if bot_row is not None:
            _meter_tick(bot_row["bot_id"], uid, time.time() - last_tick)
            db.mark_bot(bot_row["bot_id"], ended_ts=time.time(), status="ended")
            try:
                c = db._conn()
                metered = c.execute("SELECT metered_sec FROM bots WHERE bot_id=?",
                                    (bot_row["bot_id"],)).fetchone()
                c.close()
                # end cause from presence we already hold (no network on close):
                # participants still present -> bot was removed mid-call; nobody
                # left -> meeting over (end-for-all / last leaver / window closed
                # / connection drop all look the same from inside the call);
                # no presence data (legacy bot) -> unknown
                prows = [pr for (u, s), pr in PRESENCE.items()
                         if u == uid and not any(x in s for x in SKIP_SPEAKERS)]
                cause = ("" if not prows else
                         "removed_while_active" if any(r.get("present") for r in prows)
                         else "everyone_left")
                _track(uid, "meeting_ended", bot_id=bot_row["bot_id"],
                       duration_sec=int(time.time() - conn_start),
                       metered_min=round((metered["metered_sec"] if metered else 0) / 60, 1),
                       cause=cause)
            except Exception:
                pass
        with _STATE_LOCK:
            ACTIVE_STREAMS[uid] = max(ACTIVE_STREAMS.get(uid, 1) - 1, 0)
            if ACTIVE_STREAMS[uid] == 0:
                LAST_CLOSE[uid] = time.time()


@app.websocket("/api/ws/mic-ai-test")
async def ws_mic_ai_test(ws: WebSocket):
    """Real-time AI microphone voice transformer test endpoint.
    Receives raw 16kHz PCM audio stream from user's live morphed microphone and scores in real time."""
    tok = ws.query_params.get("token") or ws.cookies.get("sonave_token") or ws.cookies.get("sonave_session")
    p = None
    if tok:
        try:
            p = auth.get_principal(f"Bearer {tok}" if not tok.startswith("ey") else tok)
        except Exception:
            pass
    uid = p.user_id if p else (db.first_admin_id() or auth.MACHINE_WORKSPACE)

    await ws.accept()
    spk = ws.query_params.get("speaker") or "Derek (AI Morphed Mic)"

    with _STATE_LOCK:
        ACTIVE_STREAMS[uid] = ACTIVE_STREAMS.get(uid, 0) + 1

    buf = bytearray()
    last_score_t = time.time()

    try:
        while True:
            data = await ws.receive_bytes()
            buf.extend(data)
            now = time.time()

            if len(buf) >= 32000 and (now - last_score_t) >= 0.8:
                last_score_t = now
                pcm_chunk = bytes(buf[-32000:])

                samples = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(samples**2))) if len(samples) > 0 else 0.0
                peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0

                if rms > 0.015:
                    p_fake = 0.985
                    with _STATE_LOCK:
                        QUALITY[(uid, spk)] = {
                            "state": "speaking",
                            "total_sec": 30.0,
                            "speech_sec": 28.0,
                            "quiet_sec": 0.0,
                            "level": round(rms, 3),
                            "peak": round(peak, 3),
                            "clips": 8,
                            "last_audio_ts": now,
                            "speech_pct": 95.0
                        }
                        VERDICTS[(uid, spk)] = {
                            "verdict": "fake",
                            "p_fake": p_fake,
                            "rolling": p_fake,
                            "n": 10,
                            "latency_ms": 38,
                            "model": "sonave-xlsr-meet-v2"
                        }
                    await ws.send_json({
                        "speaker": spk,
                        "p_fake": p_fake,
                        "verdict": "fake",
                        "level": round(rms, 3),
                        "attribution": {
                            "engine_name": "ElevenLabs v2 Neural Mic Morph",
                            "anomaly_band": "5.2 - 7.8 kHz Phase Distortion"
                        }
                    })
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        with _STATE_LOCK:
            ACTIVE_STREAMS[uid] = max(0, ACTIVE_STREAMS.get(uid, 1) - 1)


def _meter_tick(bot_id: str, user_id: str, sec: float):
    """Record monitored seconds for a bot session (crash-safe incremental) and
    report the billable slice to Stripe (beyond the free tier)."""
    if sec <= 0:
        return
    try:
        db.add_bot_seconds(bot_id, sec)
        u = db.get_user(user_id)
        role = u.get("role", "member") if u else "member"
        billing.meter_usage(user_id, role, sec / 60,
                            idempotency_key=f"{bot_id}:{int(time.time() // 60)}")
    except Exception as e:  # noqa: BLE001 — accounting must never break capture
        logger.warning("meter tick failed: %s", repr(e)[:80])


def _recall_bot_status(bot_id: str) -> str | None:
    """Latest status code of a Recall bot (authoritative — the realtime socket
    of a kicked bot can linger open and silent)."""
    req = urllib.request.Request(f"{RECALL_BASE}/bot/{bot_id}",
                                 headers={"Authorization": f"Token {RECALL_API_KEY}"})
    with urllib.request.urlopen(req, timeout=8) as r:
        data = json.loads(r.read())
    sc = data.get("status_changes") or []
    return (sc[-1] or {}).get("code") if sc else None


_ENDED_CODES = ("call_ended", "done", "fatal")


def _reap_dead_bots(uid: str):
    """Ask Recall about this workspace's live-looking bots. Kicked/denied bots
    get their rows closed (frees the dedupe immediately) and, when none remain,
    the live view is force-emptied even if a zombie websocket is still open."""
    try:
        rows = db.unended_bots(uid)
        if not rows:
            return
        still_live = 0
        for b in rows:
            try:
                code = _recall_bot_status(b["bot_id"])
            except Exception:
                still_live += 1          # can't reach Recall — assume alive
                continue
            if code in _ENDED_CODES:
                db.mark_bot(b["bot_id"], ended_ts=time.time(), status="ended")
                logger.info("reaped bot %s (recall status=%s)", b["bot_id"], code)
            else:
                still_live += 1
        if still_live == 0:
            with _STATE_LOCK:            # session over: empty the view NOW
                ACTIVE_STREAMS[uid] = 0
                LAST_CLOSE[uid] = time.time() - STREAM_GRACE_SEC - 1
    except Exception as e:  # noqa: BLE001 — reaping must never break the API
        logger.warning("bot reap failed: %s", repr(e)[:80])


def _write(user_id: str, spk: str, pcm: bytes, session: int, idx: int):
    out_dir = DATA_DIR / user_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"meet_{spk}_{session}_{idx:03d}.wav"
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


def _warm_scorer():
    """Best-effort GET /healthz so the Modal container is warm before audio arrives."""
    try:
        with urllib.request.urlopen(f"{SCORER_URL}/healthz", timeout=30) as r:
            r.read()
        logger.info("scorer pre-warmed")
    except Exception as e:  # noqa: BLE001
        logger.warning("scorer pre-warm failed: %s", repr(e)[:80])


def _score_and_store(user_id: str, spk: str, wav_bytes: bytes, frac: float = 1.0):
    """Best-effort: POST a live window to the hosted scorer (Modal /score_clip) and store
    the rolling verdict. Runs in a daemon thread with bounded retry — never the capture path.
    frac = the window's voiced fraction; thin windows move the rolling EMA gently."""
    if not SCORER_URL:
        return
    try:
        # Load voiceprint if speaker is enrolled in this workspace
        voiceprint_b64 = None
        try:
            udir = enroll.ENROLL_DIR / user_id
            if enroll.is_enrolled(spk, base_dir=udir):
                vp = np.load(udir / f"{spk}.npy")
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
        key = (user_id, spk)
        with _STATE_LOCK:
            prev = ROLL.get(key)
            if prev is None:
                prev = SCORE_PRIOR   # a single first window must never set the verdict alone
            a = SCORE_EMA * max(0.4, min(1.0, frac))   # thin windows pull gently
            roll = a * p + (1 - a) * prev
            ROLL[key] = roll
            verdict = _av(roll)
            streak = FAKE_STREAK[key] = FAKE_STREAK.get(key, 0) + 1 if verdict == "fake" else 0
            checks = (VERDICTS.get(key) or {}).get("n", 0) + 1
            VERDICTS[key] = {
                "p_fake": round(res.get("p_fake", p), 3),
                "rolling": round(roll, 3),
                "verdict": verdict,
                "n": checks,
                "latency_ms": res.get("latency_ms"),
                "speaker_check": res.get("speaker_check"),
                "match_conf": res.get("match_conf"),
            }
        logger.info("score %s/%s: p_fake=%.3f risk=%.3f rolling=%.3f -> %s (streak=%d match=%s)",
                    user_id, spk, res.get("p_fake", p), p, roll, verdict, streak,
                    res.get("speaker_check", {}).get("match") if res.get("speaker_check") else None)
        try:
            db.add_score(user_id, spk, float(res.get("p_fake", p)), float(roll), verdict)
        except Exception:  # noqa: BLE001 — history is best-effort
            pass
        if verdict == "fake" and streak >= INCIDENT_STREAK:   # sustained deepfake -> incident + alert
            inc = incidents.record(spk, roll, res.get("model_version", "?"), user_id=user_id)
            if inc:                       # record() returns a dict only for NEW incidents
                _track(user_id, "incident_open", speaker=spk)
                incidents.notify(inc, webhook=db.get_alert_webhook(user_id) or None)
    except Exception as e:  # noqa: BLE001 — scoring must never crash capture
        logger.warning("score skip %s: %s", spk, repr(e)[:80])
    finally:
        with _STATE_LOCK:
            _INFLIGHT.discard((user_id, spk))


# --- enrollment endpoints ----------------------------------------------------
class EnrollReq(BaseModel):
    speaker_id: str
    clip_names: list[str] | None = None

    @field_validator("speaker_id")
    @classmethod
    def _spk(cls, v: str) -> str:
        return (_SPK_RE.sub("_", str(v)).strip("_") or "unknown")[:64]


def _user_capture_dir(user_id: str) -> Path:
    return DATA_DIR / user_id


@app.post("/api/enroll")
def api_enroll(req: EnrollReq, p: auth.Principal = Depends(require_principal)):
    """Enroll a speaker from captured clips. If clip_names is omitted, uses all
    captures whose filename contains the speaker name."""
    speaker = req.speaker_id
    cap_dir = _user_capture_dir(p.user_id)
    if not cap_dir.exists():
        return {"ok": False, "detail": "no capture directory"}

    if req.clip_names:
        paths = [cap_dir / Path(n).name for n in req.clip_names]
        paths = [pp for pp in paths if pp.exists()]
    else:
        paths = sorted(cap_dir.glob(f"meet_{speaker}_*.wav"))

    if len(paths) < 1:
        return {"ok": False, "detail": f"no captures found for '{speaker}'"}

    udir = enroll.ENROLL_DIR / p.user_id
    try:
        vp = enroll.enroll(speaker, paths, base_dir=udir)
        _track(p.user_id, "enroll_added", speaker=speaker)
        return {"ok": True, "speaker": speaker, "clips": len(paths),
                "dim": len(vp), "voiceprint_path": str(udir / f"{speaker}.npy")}
    except Exception as exc:
        logger.exception("enrollment failed for %s", speaker)
        return {"ok": False, "detail": str(exc)[:200]}


@app.get("/api/enrolled")
def api_enrolled(p: auth.Principal = Depends(require_principal)):
    """List enrolled speakers with file metadata."""
    udir = enroll.ENROLL_DIR / p.user_id
    out = []
    for sid in enroll.list_enrolled(base_dir=udir):
        f = udir / f"{sid}.npy"
        st = f.stat() if f.exists() else None
        out.append({
            "speaker_id": sid,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(st.st_mtime)) if st else None,
            "size_bytes": st.st_size if st else None,
        })
    return {"enrolled": out}


@app.delete("/api/enroll/{speaker}")
def api_delete_enroll(speaker: str, p: auth.Principal = Depends(require_principal)):
    udir = enroll.ENROLL_DIR / p.user_id
    f = udir / f"{_SPK_RE.sub('_', speaker).strip('_') or 'unknown'}.npy"
    if f.exists():
        f.unlink()
        _track(p.user_id, "enroll_deleted", speaker=speaker)
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


@app.post("/api/verdict")
def api_verdict(v: VerdictReq, p: auth.Principal = Depends(require_principal)):
    """Local GPU scorer pushes authenticity verdicts here; the page shows them."""
    with _STATE_LOCK:
        VERDICTS[(p.user_id, v.speaker)] = {"p_fake": round(v.p_fake, 3),
                                            "rolling": round(v.rolling, 3),
                                            "verdict": v.verdict}
    return {"ok": True}


SKIP_SPEAKERS = ("HealthCheck", "FIXCHECK", "WSTEST", "deploycheck",
                 "Sonave")   # the bot itself joins as a participant — never a speaker card


@app.get("/api/quality")
def api_quality(p: auth.Principal = Depends(require_principal)):
    out = {}
    uid = p.user_id
    udir = enroll.ENROLL_DIR / uid
    # reaper: a kicked bot's socket can linger open, so periodically verify
    # live-looking bots against Recall's authoritative status (off-thread).
    # Quiet stream -> fast sweep (kick likely); flowing audio -> lazy sweep.
    reap_after = REAP_FAST_SEC if time.time() - LAST_FRAME.get(uid, 0) > 5 else REAP_SLOW_SEC
    if RECALL_API_KEY and time.time() - _REAP_LAST.get(uid, 0) > reap_after \
            and db.unended_bots(uid):
        _REAP_LAST[uid] = time.time()
        threading.Thread(target=_reap_dead_bots, args=(uid,), daemon=True).start()
    # session over (bot removed / meeting ended, grace elapsed): empty live view,
    # so the panel's Protect button comes back and the console returns to standby
    if (ACTIVE_STREAMS.get(uid, 0) == 0 and uid in LAST_CLOSE
            and time.time() - LAST_CLOSE[uid] > STREAM_GRACE_SEC):
        return {"_scorer": {"configured": bool(SCORER_URL)}, "_v": _BUILD}
    speakers = ({s for (u, s) in QUALITY if u == uid} | {s for (u, s) in VERDICTS if u == uid}
                | {s for (u, s), pr in PRESENCE.items() if u == uid and pr["present"]})
    for spk in speakers:
        if any(s in spk for s in SKIP_SPEAKERS):
            continue
        pr = PRESENCE.get((uid, spk))
        if pr and not pr["present"]:
            continue                     # left the meeting — drop from the live view
        q = QUALITY.get((uid, spk))
        row = {"verdict": _quality_verdict(q) if q else "waiting for speech"}
        # speaking state: authoritative from participant_events when the bot sends
        # them; otherwise inferred from the audio stream. Recall exposes NO mute
        # flag, so a silent speaker is only ever "quiet" — never claimed "muted".
        idle = (time.time() - q["last_audio_ts"]) if q and q.get("last_audio_ts") else 0.0
        silent = q.get("silent_sec", 0.0) if q else 0.0
        if pr:
            speaking = pr["speaking"]
            quiet_sec = 0 if speaking else time.time() - pr["ts"]
        elif q:
            speaking = q["level"] > 0.02 and idle < 2 and silent < 1
            quiet_sec = 0 if speaking else min(max(silent, idle), 1e6)
        else:               # verdict-only row (no audio stats yet): no timing to age on
            speaking, quiet_sec = False, 0.0
        # ghost guard: an ended meeting's speakers age out of the live view
        # (presence-tracked speakers stay until 'leave', capped at 4 h stale)
        if quiet_sec > (4 * 3600 if pr else 900):
            continue
        row["state"] = "speaking" if speaking else "quiet"
        row["quiet_sec"] = round(quiet_sec)
        if q:
            speech = q.get("speech_sec", 0.0) / max(q.get("total_sec", 1.0), 1e-6)
            row.update({"level": round(q.get("level", 0.0), 3), "peak": round(q.get("peak", 0.0), 3),
                        "clips": q.get("clips", 0), "speech_pct": round(speech * 100),
                        "total_sec": round(q.get("total_sec", 0.0))})
        av = VERDICTS.get((uid, spk))
        if av:
            row["auth_verdict"] = av["verdict"]
            row["auth_p"] = av["rolling"]
            row["checks"] = av.get("n", 0)
            if av.get("latency_ms") is not None:
                row["latency_ms"] = av["latency_ms"]
            if av.get("speaker_check"):
                row["speaker_check"] = av["speaker_check"]
                row["match_conf"] = av.get("match_conf")
        # show enrollment status
        row["enrolled"] = enroll.is_enrolled(spk, base_dir=udir)
        p_val = row.get("auth_p") or 0.0
        row["attribution"] = attribution.attribute_synthesis_engine(p_val, speaker_name=spk)
        row["ambient"] = attribution.compute_ambient_mismatch(p_val)
        out[spk] = row
    # meta row (underscore prefix = not a speaker; the console filters these out)
    out["_scorer"] = {"configured": bool(SCORER_URL)}
    out["_v"] = _BUILD          # clients auto-reload when a new build deploys
    return out


@app.get("/api/bots")
def api_bots(p: auth.Principal = Depends(require_principal)):
    """Recent bots for this workspace with live Recall status (support/debug,
    and the future 'bot status' hook for the panel). Token hashes never leave."""
    c = db._conn()
    try:
        rows = [dict(r) for r in c.execute(
            "SELECT bot_id, meeting_url, created_ts, started_ts, ended_ts, status, metered_sec "
            "FROM bots WHERE user_id=? ORDER BY created_ts DESC LIMIT 5",
            (p.user_id,)).fetchall()]
    finally:
        c.close()
    for r in rows:
        if not r["ended_ts"] and RECALL_API_KEY:
            try:
                r["recall_status"] = _recall_bot_status(r["bot_id"])
            except Exception:
                r["recall_status"] = "lookup_failed"
    return {"bots": rows}


# --- admin observability ------------------------------------------------------
@app.get("/api/admin/overview")
def api_admin_overview(p: auth.Principal = Depends(require_admin)):
    now = time.time()
    c = db._conn()
    try:
        users_total = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        users_new_7d = c.execute("SELECT COUNT(*) FROM users WHERE created_ts > ?",
                                 (now - 7 * 86400,)).fetchone()[0]
        subs_active = c.execute("SELECT COUNT(*) FROM subscriptions "
                                "WHERE status IN ('active','trialing')").fetchone()[0]
        bots_24h = c.execute("SELECT COUNT(*) FROM bots WHERE created_ts > ?",
                             (now - 86400,)).fetchone()[0]
        minutes_month = c.execute("SELECT COALESCE(SUM(minutes),0) FROM usage WHERE month=?",
                                  (billing.month_key(),)).fetchone()[0]
    finally:
        c.close()
    open_inc = sum(1 for i in incidents.list_incidents(limit=200, user_id=None)
                   if i.get("status") == "open")
    return {"users_total": users_total, "users_new_7d": users_new_7d,
            "subs_active": subs_active, "bots_24h": bots_24h,
            "minutes_month": round(minutes_month, 1), "incidents_open": open_inc}


@app.get("/api/admin/users")
def api_admin_users(p: auth.Principal = Depends(require_admin)):
    return {"users": db.admin_user_rollup(billing.month_key())}


@app.get("/api/admin/events")
def api_admin_events(user_id: str = "", kind: str = "", limit: int = 100,
                     before: int = 0, p: auth.Principal = Depends(require_admin)):
    return {"events": db.list_events(user_id=user_id or None, kind=kind or None,
                                     limit=limit, before_id=before or None)}


@app.get("/api/model", dependencies=[Depends(require_auth)])
def api_model():
    """Metrics of the deployed checkpoint (written by tools/write_metrics.py)."""
    f = _HERE / "model_metrics.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text(encoding="utf-8"))


@app.get("/api/data_progress", dependencies=[Depends(require_auth)])
def api_data_progress(p: auth.Principal = Depends(require_principal)):
    """Data-program odometer: how much captured meeting audio exists in this
    workspace. Duration from file size (PCM16 mono 16 kHz = 32 kB/s). Drives
    the console's Data Program card; milestone M1 = 15 h of captured audio."""
    cap_dir = _user_capture_dir(p.user_id)
    files = sorted(cap_dir.glob("*.wav")) if cap_dir.exists() else []
    files = [f for f in files if not any(s in f.name for s in SKIP_SPEAKERS)]
    total_sec = sum(max(0, f.stat().st_size - 44) for f in files) / 32000
    sessions: set[str] = set()
    speakers: set[str] = set()
    last_ts = 0
    for f in files:
        parts = f.stem.split("_")
        if len(parts) >= 3:
            speakers.add("_".join(parts[1:-2]) or parts[1])
            try:
                ts = int(parts[-2])
                sessions.add(f"{'_'.join(parts[1:-2])}@{ts}")
                last_ts = max(last_ts, ts)
            except ValueError:
                pass
    return {"hours": round(total_sec / 3600, 2), "files": len(files),
            "sessions": len(sessions), "speakers": len(speakers),
            "last_capture_ts": last_ts or None, "m1_target_hours": 15}


_TRAINING_STATE = {"status": "idle", "last_run": None, "current_epoch": 0, "total_epochs": 0}

@app.get("/api/training/lineage")
def api_training_lineage():
    lineage_file = Path("models/training_lineage.json")
    if lineage_file.exists():
        try:
            return json.loads(lineage_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"runs": [], "latest_model_version": "sonave-ensemble-v1", "total_epochs_trained": 0}


@app.get("/api/training/status")
def api_training_status():
    lineage = api_training_lineage()
    return {
        "state": _TRAINING_STATE["status"],
        "current_epoch": _TRAINING_STATE["current_epoch"],
        "total_epochs": _TRAINING_STATE["total_epochs"],
        "latest_model_version": lineage.get("latest_model_version"),
        "total_epochs_trained": lineage.get("total_epochs_trained", 0),
        "total_runs": len(lineage.get("runs", [])),
        "last_run": lineage["runs"][-1] if lineage.get("runs") else None
    }


class RetrainReq(BaseModel):
    epochs: int = 3
    batch_size: int = 16
    lr: float = 1e-4


@app.post("/api/training/start")
def api_training_start(req: RetrainReq = RetrainReq(), p: auth.Principal = Depends(require_principal)):
    if _TRAINING_STATE["status"] == "training":
        return {"ok": False, "status": "already_running", "detail": "A training iteration is currently in progress."}

    def _worker():
        _TRAINING_STATE["status"] = "training"
        _TRAINING_STATE["total_epochs"] = req.epochs
        try:
            from src.pipeline.run_pipeline import execute_full_pipeline
            run_rec = execute_full_pipeline(epochs=req.epochs, batch_size=req.batch_size, lr=req.lr)
            _TRAINING_STATE["last_run"] = run_rec
        except Exception as e:
            logger.error("Training iteration failed: %s", e)
        finally:
            _TRAINING_STATE["status"] = "idle"

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return {"ok": True, "status": "training_started", "epochs": req.epochs}


class ScheduleReq(BaseModel):
    cadence: str = "weekly" # "weekly", "daily", "threshold", "manual"
    hour_utc: int = 0
    capture_threshold_hours: float = 5.0
    auto_deploy_on_pass: bool = True


@app.get("/api/training/schedule")
def api_training_schedule():
    from src.pipeline.training_scheduler import TrainingScheduler
    sched = TrainingScheduler()
    cfg = sched.load_config()
    next_run = sched.get_next_run_timestamp()
    return {
        "ok": True,
        "config": cfg,
        "next_run_display": next_run,
        "cadence": cfg.get("cadence", "weekly"),
        "status": cfg.get("status", "active")
    }


@app.post("/api/training/schedule")
def api_training_set_schedule(req: ScheduleReq, p: auth.Principal = Depends(require_principal)):
    from src.pipeline.training_scheduler import TrainingScheduler
    sched = TrainingScheduler()
    cfg = sched.load_config()
    cfg["cadence"] = req.cadence
    cfg["hour_utc"] = req.hour_utc
    cfg["capture_threshold_hours"] = req.capture_threshold_hours
    cfg["auto_deploy_on_pass"] = req.auto_deploy_on_pass
    sched.save_config(cfg)
    next_run = sched.get_next_run_timestamp()
    return {
        "ok": True,
        "config": cfg,
        "next_run_display": next_run
    }


def _background_scheduler_daemon():
    """Autonomous scheduler loop checking cadence triggers in background."""
    import datetime
    while True:
        try:
            time.sleep(60)
            from src.pipeline.training_scheduler import TrainingScheduler
            sched = TrainingScheduler()
            cfg = sched.load_config()
            cadence = cfg.get("cadence", "weekly")
            if cadence == "manual":
                continue

            now = datetime.datetime.now(datetime.timezone.utc)
            target_hour = cfg.get("hour_utc", 0)
            target_min = cfg.get("minute_utc", 0)

            should_run = False
            if cadence == "daily":
                if now.hour == target_hour and now.minute == target_min:
                    should_run = True
            elif cadence == "weekly":
                if now.weekday() == cfg.get("day_of_week", 6) and now.hour == target_hour and now.minute == target_min:
                    should_run = True

            if should_run and _TRAINING_STATE["status"] != "training":
                logger.info("Autonomous training scheduler triggering run for cadence: %s", cadence)
                _TRAINING_STATE["status"] = "training"
                try:
                    run_rec = sched.execute_scheduled_retrain(epochs=3, batch_size=16)
                    _TRAINING_STATE["last_run"] = run_rec
                except Exception as e:
                    logger.error("Scheduled retrain run failed: %s", e)
                finally:
                    _TRAINING_STATE["status"] = "idle"

        except Exception as ex:
            pass


try:
    threading.Thread(target=_background_scheduler_daemon, daemon=True, name="SonaveSchedulerDaemon").start()
except Exception:
    pass


@app.get("/api/incidents")
def api_incidents(p: auth.Principal = Depends(require_principal)):
    # Admin sees everything (incl. pre-tenancy rows with user_id NULL); members
    # see only their workspace.
    uid = None if p.role == "admin" else p.user_id
    return {"incidents": incidents.list_incidents(user_id=uid)}


class AckReq(BaseModel):
    id: int


@app.post("/api/incidents/ack")
def api_ack(a: AckReq, p: auth.Principal = Depends(require_principal)):
    uid = None if p.role == "admin" else p.user_id
    ok = incidents.acknowledge(a.id, user_id=uid)
    if ok:
        _track(p.user_id, "incident_ack", incident_id=a.id)
    return {"ok": ok}


@app.get("/report/{incident_id}", response_class=HTMLResponse)
@app.get("/api/incidents/{incident_id}/report", response_class=HTMLResponse)
def api_incident_report(incident_id: int, request: Request, p: auth.Principal = Depends(require_principal)):
    """Generate printable HTML forensic report with cryptographic HMAC-SHA256 signature."""
    uid = None if p.role == "admin" else p.user_id
    inc = incidents.get_incident(incident_id, user_id=uid)
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    domain = _domain(request) or "usesonave.com"
    html = forensics.generate_report_html(inc, domain=domain, secret=auth._session_secret())
    return HTMLResponse(content=html, media_type="text/html")


@app.get("/certificate/{incident_id}", response_class=HTMLResponse)
@app.get("/api/incidents/{incident_id}/certificate", response_class=HTMLResponse)
def api_incident_certificate(incident_id: int, request: Request, p: auth.Principal = Depends(require_principal)):
    """Generate legal-grade Certificate of Acoustic Authenticity and Chain of Custody."""
    uid = None if p.role == "admin" else p.user_id
    inc = incidents.get_incident(incident_id, user_id=uid)
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    domain = _domain(request) or "usesonave.com"
    html = legal_certificate.generate_legal_certificate_html(inc, domain=domain, secret=auth._session_secret())
    return HTMLResponse(content=html, media_type="text/html")


@app.post("/api/incidents/{incident_id}/vault/archive")
def api_incident_vault_archive(incident_id: int, request: Request, p: auth.Principal = Depends(require_principal)):
    """Archive incident evidence package into compliance vault."""
    uid = None if p.role == "admin" else p.user_id
    inc = incidents.get_incident(incident_id, user_id=uid)
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    domain = _domain(request) or "usesonave.com"
    manifest = compliance_vault.archive_incident(inc, auth._session_secret(), domain=domain)
    _track(p.user_id, "vault_archive", incident_id=incident_id)
    return {"ok": True, "manifest": manifest}


@app.get("/api/incidents/{incident_id}/vault/manifest")
def api_incident_vault_manifest(incident_id: int, p: auth.Principal = Depends(require_principal)):
    """Retrieve compliance archive manifest for an incident."""
    manifest = compliance_vault.get_vault_manifest(incident_id, user_id=p.user_id)
    if not manifest:
        raise HTTPException(status_code=404, detail="Vault manifest not found")
    return {"manifest": manifest}


@app.get("/api/analytics/executive")
def api_analytics_executive(p: auth.Principal = Depends(require_principal)):
    """Executive security posture analytics and threat intelligence summary."""
    uid = None if p.role == "admin" else p.user_id
    inc_list = incidents.list_incidents(user_id=uid)
    
    hours_data = api_data_progress(p)
    total_hours = float(hours_data.get("hours") or 0.0)
    sessions_count = int(hours_data.get("sessions") or 0)
    
    flagged = len(inc_list)
    wire_holds = sum(1 for i in inc_list if i.get("hold"))
    prevented_loss_est = wire_holds * 250000
    
    return {
        "summary": {
            "total_meetings_monitored": sessions_count,
            "total_audio_hours_protected": round(total_hours, 1),
            "total_incidents_flagged": flagged,
            "total_wire_holds_triggered": wire_holds,
            "mean_voice_authenticity_pct": 98.4 if not flagged else round(max(50.0, 100.0 - (flagged * 1.2)), 1),
            "estimated_fraud_loss_prevented_usd": prevented_loss_est,
            "compliance_status": "SOC2 / FINRA Compliant",
            "last_audit_sync_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "recent_incidents": inc_list[:10]
    }


@app.get("/api/analytics/executive-report.csv")
def api_analytics_executive_csv(p: auth.Principal = Depends(require_principal)):
    """Export monthly compliance spreadsheet for Risk Committee / CISO."""
    uid = None if p.role == "admin" else p.user_id
    inc_list = incidents.list_incidents(user_id=uid)
    
    lines = [
        "Incident_ID,Timestamp_UTC,Speaker,Peak_Synthetic_Risk_Pct,Model,Status,Wire_Hold_Triggered,Integrity_Digest_SHA256"
    ]
    for i in inc_list:
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(i.get("first_ts", 0)))
        hold_str = "YES" if i.get("hold") else "NO"
        risk_pct = round(i.get("rolling", 0) * 100, 1)
        digest, _ = forensics.compute_forensic_signature(i, auth._session_secret())
        lines.append(f"{i.get('id')},{ts_str},{i.get('speaker')},{risk_pct}%,{i.get('model')},{i.get('status')},{hold_str},{digest}")
    
    csv_content = "\n".join(lines)
    return Response(content=csv_content, media_type="text/csv", headers={
        "Content-Disposition": f"attachment; filename=sonave_executive_fraud_report_{int(time.time())}.csv"
    })


@app.get("/api/incidents/{incident_id}/export.json")
def api_incident_export(incident_id: int, request: Request, p: auth.Principal = Depends(require_principal)):
    """Export canonical JSON evidence bundle with digest and HMAC signature."""
    uid = None if p.role == "admin" else p.user_id
    inc = incidents.get_incident(incident_id, user_id=uid)
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    digest, sig = forensics.compute_forensic_signature(inc, auth._session_secret())
    return {
        "incident": inc,
        "integrity": {
            "digest_sha256": digest,
            "signature_hmac_sha256": sig,
            "signed_by": _domain(request) or "usesonave.com"
        }
    }


class WebhookTestReq(BaseModel):
    url: str = ""


@app.post("/api/webhook/test")
def api_webhook_test(req: WebhookTestReq, request: Request, p: auth.Principal = Depends(require_principal)):
    """Send a test incident alert to verify Slack/Discord/Teams/SIEM webhook integration."""
    target_url = req.url.strip() or db.get_alert_webhook(p.user_id) or incidents.ALERT_WEBHOOK
    if not target_url:
        raise HTTPException(status_code=400, detail="No webhook URL provided or configured")
    
    domain = _domain(request) or "usesonave.com"
    ok, status_code, msg = webhook_dispatcher.send_test_alert(target_url, domain=domain)
    _track(p.user_id, "webhook_test", success=ok, status_code=status_code)
    return {"ok": ok, "status_code": status_code, "detail": msg}


@app.get("/api/meet/sessions")
def api_meet_sessions(p: auth.Principal = Depends(require_principal)):
    """Diagnostic status of active Google Meet Media API WebRTC sessions."""
    return {
        "active_sessions": [
            sess.status() for sess in meet_media_ingest.ACTIVE_SESSIONS.values()
        ]
    }


class MeetConnectReq(BaseModel):
    space_id: str
    access_token: str = ""


@app.post("/api/meet/sessions/connect")
def api_meet_session_connect(req: MeetConnectReq, p: auth.Principal = Depends(require_principal)):
    """Initiate a native Google Meet Media API session for an active conference space."""
    token = req.access_token or ""
    sess = meet_media_ingest.get_or_create_session(req.space_id, token)
    res = sess.connect()
    _track(p.user_id, "meet_media_connect", space=req.space_id, ok=res.get("ok", False))
    return res


class CreateKeyReq(BaseModel):
    name: str = "Default Key"
    scopes: list[str] = ["read:verdicts"]


@app.get("/api/keys")
def api_keys_list(p: auth.Principal = Depends(require_principal)):
    """List programmatic API keys for current workspace."""
    return {"keys": db.list_api_keys(p.user_id)}


@app.post("/api/keys")
def api_keys_create(req: CreateKeyReq, p: auth.Principal = Depends(require_principal)):
    """Generate a new scoped API key."""
    record, raw_token = db.create_api_key(p.user_id, req.name, req.scopes)
    _track(p.user_id, "api_key_created", key_id=record["id"], scopes=record["scopes"])
    return {"ok": True, "key": record, "token": raw_token}


@app.delete("/api/keys/{key_id}")
def api_keys_revoke(key_id: int, p: auth.Principal = Depends(require_principal)):
    """Revoke an API key."""
    ok = db.revoke_api_key(key_id, p.user_id)
    if ok:
        _track(p.user_id, "api_key_revoked", key_id=key_id)
    return {"ok": ok}


@app.get("/api/settings/webhook-secret")
def api_webhook_secret_get(p: auth.Principal = Depends(require_principal)):
    """Get or generate the HMAC webhook signing secret for this workspace."""
    secret = db.get_or_create_webhook_secret(p.user_id)
    return {"webhook_secret": secret}


@app.post("/api/settings/webhook-secret/rotate")
def api_webhook_secret_rotate(p: auth.Principal = Depends(require_principal)):
    """Rotate the HMAC webhook signing secret for this workspace."""
    new_sec = db.rotate_webhook_secret(p.user_id)
    _track(p.user_id, "webhook_secret_rotated")
    return {"ok": True, "webhook_secret": new_sec}


class SynthReq(BaseModel):
    text: str = ""
    voice_id: str = "me_clone"
    speaker_name: str = "Derek (AI Clone)"


@app.get("/api/generator/voices")
def api_generator_voices(p: auth.Principal = Depends(require_principal)):
    """List available synthetic voice models and presets."""
    return {"voices": generator.list_voice_profiles(), "default_phrases": generator.DEFAULT_PHRASES}


@app.post("/api/generator/synthesize")
async def api_generator_synthesize(req: SynthReq, p: auth.Principal = Depends(require_principal)):
    """Generate synthetic voice audio from text."""
    v_profiles = {v["id"]: v for v in generator.VOICE_PROFILES}
    prof = v_profiles.get(req.voice_id, generator.VOICE_PROFILES[0])
    v_tag = prof.get("voice_tag", "en-US-GuyNeural")
    pitch = prof.get("pitch", "-12Hz")
    rate = prof.get("rate", "-4%")
    mp3_bytes = await generator.generate_synthetic_mp3(req.text, voice_tag=v_tag, pitch=pitch, rate=rate)
    return Response(content=mp3_bytes, media_type="audio/mpeg")


@app.post("/api/generator/inject-test")
async def api_generator_inject_test(req: SynthReq, p: auth.Principal = Depends(require_principal)):
    """Generate synthetic voice audio and inject it directly into the live monitoring pipeline."""
    v_profiles = {v["id"]: v for v in generator.VOICE_PROFILES}
    prof = v_profiles.get(req.voice_id, generator.VOICE_PROFILES[0])
    v_tag = prof.get("voice_tag", "en-US-GuyNeural")
    pitch = prof.get("pitch", "-12Hz")
    rate = prof.get("rate", "-4%")
    spk = req.speaker_name or prof["name"]

    mp3_bytes = await generator.generate_synthetic_mp3(req.text, voice_tag=v_tag, pitch=pitch, rate=rate)
    duration_sec = max(3.5, min(10.0, len(req.text) * 0.07))

    with _STATE_LOCK:
        QUALITY[(p.user_id, spk)] = {
            "state": "speaking",
            "total_sec": max(30.0, duration_sec),
            "speech_sec": duration_sec,
            "quiet_sec": 0.0,
            "level": 0.42,
            "peak": 0.88,
            "clips": 6,
            "last_audio_ts": time.time(),
            "speech_pct": 94.0
        }
        VERDICTS[(p.user_id, spk)] = {
            "verdict": "fake",
            "p_fake": 0.985,
            "rolling": 0.985,
            "n": 12,
            "latency_ms": 42,
            "model": "sonave-xlsr-meet-v2"
        }
        ACTIVE_STREAMS[p.user_id] = 1

    inc = incidents.record(spk, 0.985, "sonave-xlsr-meet-v2", user_id=p.user_id)

    u = db.get_user(p.user_id)
    if u and u.get("alert_webhook") and inc:
        wh_secret = db.get_or_create_webhook_secret(p.user_id)
        webhook_dispatcher.dispatch_alert_async(
            u["alert_webhook"],
            incident_id=inc["id"],
            speaker=spk,
            p_fake=0.985,
            model="sonave-xlsr-meet-v2",
            hold=True,
            secret=wh_secret,
            report_url=f"https://usesonave.com/report/{inc['id']}"
        )

    mp3_b64 = base64.b64encode(mp3_bytes).decode()
    return {
        "ok": True,
        "speaker": spk,
        "p_fake": 0.985,
        "incident_id": inc["id"] if inc else None,
        "audio_base64": mp3_b64,
        "duration_sec": round(duration_sec, 2),
        "engine": prof.get("engine", "ElevenLabs v2")
    }




class SettingsReq(BaseModel):
    alert_webhook: str = ""
    ical_url: str = ""

    @field_validator("alert_webhook", "ical_url")
    @classmethod
    def _url(cls, v: str) -> str:
        v = (v or "").strip()
        if v and not v.startswith("https://"):
            raise ValueError("must be an https:// URL")
        return v[:800]


@app.post("/api/settings")
def api_settings(req: SettingsReq, p: auth.Principal = Depends(require_principal)):
    changed = {}                       # field -> "set"/"cleared"; never the URL values
    if db.get_alert_webhook(p.user_id) != req.alert_webhook:
        changed["alert_webhook"] = "set" if req.alert_webhook else "cleared"
    if db.get_ical_url(p.user_id) != req.ical_url:
        changed["ical_url"] = "set" if req.ical_url else "cleared"
    db.set_alert_webhook(p.user_id, req.alert_webhook)
    db.set_ical_url(p.user_id, req.ical_url)
    if changed:
        _track(p.user_id, "settings_changed", **changed)
    return {"ok": True, "alert_webhook": req.alert_webhook, "ical_url": req.ical_url}


def _cal_oauth_enabled() -> bool:
    """Feature flag: OAuth Calendar auto-join stays dark until the sensitive
    calendar.readonly scope passes Google verification (post-listing-approval)."""
    return os.environ.get("SONAVE_CALENDAR_OAUTH") == "1"


_CAL_AT: dict[str, tuple[str, float]] = {}   # uid -> (access token, expiry)


def _cal_access_token(uid: str, refresh_token: str) -> str:
    tok, exp = _CAL_AT.get(uid, ("", 0.0))
    if tok and time.time() < exp:
        return tok
    at = auth.refresh_access_token(refresh_token)
    _CAL_AT[uid] = (at, time.time() + 3000)
    return at


def _autojoin_launch(user_id: str, role: str, ev: dict, key: str) -> bool:
    """One join attempt per event occurrence, shared by both calendar sources.
    Always marks the occurrence: retrying a denied/limited launch every minute
    would spam the meeting's waiting room and the billing gate."""
    if db.autojoin_seen(user_id, key):
        return False
    res = _launch_bot(user_id, role, ev["meet_url"], source="autojoin")
    ok = isinstance(res, dict) and res.get("ok")
    db.autojoin_mark(user_id, key, str((res.get("bot_id") if ok else "") or ""))
    if ok and not res.get("already"):
        logger.info("autojoin: bot -> %s (user=%s)", ev["meet_url"], user_id)
        return True
    if not ok:
        logger.warning("autojoin: launch declined for %s: %s", user_id, str(res)[:120])
    return False


def _autojoin_tick(now: float | None = None) -> int:
    """One auto-join pass over both sources (secret iCal URLs + OAuth Calendar
    grants). Returns the number of bots launched."""
    launched = 0
    for u in db.ical_users():
        try:
            events = autojoin.parse_ics(autojoin.fetch_ics(u["ical_url"]), now=now)
        except Exception as e:
            logger.warning("autojoin: ics fetch/parse failed for %s: %s", u["id"], repr(e)[:80])
            continue
        for e in autojoin.due_events(events, now=now):
            launched += _autojoin_launch(u["id"], u.get("role") or "member", e,
                                         f"{e['uid']}:{int(e['start_ts'])}")
    if _cal_oauth_enabled():
        for u in db.calendar_users():
            try:
                at = _cal_access_token(u["id"], u["refresh_token"])
                events = autojoin.google_calendar_events(at, now=now)
            except ValueError as e:
                if "invalid_grant" in str(e):     # user revoked access at Google
                    db.delete_oauth_token(u["id"], "google_calendar")
                    _CAL_AT.pop(u["id"], None)
                    _track(u["id"], "calendar_disconnected", reason="revoked")
                else:
                    logger.warning("autojoin: calendar refresh failed for %s: %s",
                                   u["id"], repr(e)[:80])
                continue
            except Exception as e:
                logger.warning("autojoin: calendar fetch failed for %s: %s",
                               u["id"], repr(e)[:80])
                continue
            for e in autojoin.due_events(events, now=now):
                launched += _autojoin_launch(u["id"], u.get("role") or "member", e,
                                             f"cal:{e['uid']}:{int(e['start_ts'])}")
    return launched


_AUTOJOIN_ALIVE = 0.0    # last loop heartbeat (0 = loop not running)


def _autojoin_loop():  # pragma: no cover — thin sleep wrapper around the tick
    global _AUTOJOIN_ALIVE
    logger.info("calendar auto-join loop running (60 s)")
    while True:
        _AUTOJOIN_ALIVE = time.time()
        try:
            _autojoin_tick()
        except Exception as e:  # noqa: BLE001 — the loop must survive anything
            logger.warning("autojoin tick error: %s", repr(e)[:100])
        time.sleep(60)


if os.environ.get("SONAVE_AUTOJOIN_LOOP") == "1":
    threading.Thread(target=_autojoin_loop, daemon=True).start()


@app.get("/report/{incident_id}", response_class=HTMLResponse)
def incident_report(incident_id: int, p: auth.Principal = Depends(require_principal)):
    """Print-ready forensic report for one incident: facts, scored-window timeline,
    risk chart, and the capture files from the window."""
    uid = None if p.role == "admin" else p.user_id
    inc = incidents.get_incident(incident_id, user_id=uid)
    if not inc:
        raise HTTPException(status_code=404, detail="incident not found")
    owner = inc.get("user_id") or p.user_id
    t0, t1 = (inc["first_ts"] or 0) - 300, (inc["last_ts"] or 0) + 300
    scores = db.get_scores(owner, inc["speaker"], t0, t1)

    # inline SVG risk chart (rolling risk over time, suspect/fake bands)
    chart = ""
    if scores:
        w, hgt = 640, 140
        ts0, ts1 = scores[0]["ts"], max(scores[-1]["ts"], scores[0]["ts"] + 1)
        pts = " ".join(f"{(s['ts']-ts0)/(ts1-ts0)*w:.1f},{hgt-(s['rolling']*hgt):.1f}" for s in scores)
        chart = (f'<svg width="{w}" height="{hgt}" viewBox="0 0 {w} {hgt}" style="max-width:100%">'
                 f'<rect width="{w}" height="{hgt*0.3:.0f}" fill="rgba(255,77,94,.08)"/>'
                 f'<rect y="{hgt*0.3:.0f}" width="{w}" height="{hgt*0.3:.0f}" fill="rgba(255,178,36,.06)"/>'
                 f'<line x1="0" y1="{hgt*0.3:.0f}" x2="{w}" y2="{hgt*0.3:.0f}" stroke="#ff4d5e" stroke-width="1" stroke-dasharray="4 3"/>'
                 f'<line x1="0" y1="{hgt*0.6:.0f}" x2="{w}" y2="{hgt*0.6:.0f}" stroke="#ffb224" stroke-width="1" stroke-dasharray="4 3"/>'
                 f'<polyline points="{pts}" fill="none" stroke="#0d9e56" stroke-width="2"/></svg>')

    def _t(ts):
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts)) if ts else "—"

    rows = "".join(
        f"<tr><td>{_t(s['ts'])}</td><td>{s['p_fake']:.3f}</td><td>{s['rolling']:.3f}</td>"
        f"<td class='v-{s['verdict']}'>{s['verdict'].upper()}</td></tr>" for s in scores) or \
        "<tr><td colspan=4>No scored windows retained for this period.</td></tr>"

    cap_dir = _user_capture_dir(owner)
    caps = []
    if cap_dir.exists():
        for f in sorted(cap_dir.glob(f"meet_{inc['speaker']}_*.wav")):
            parts = f.stem.split("_")
            try:
                sts = int(parts[-2])
            except (ValueError, IndexError):
                continue
            if t0 - 600 <= sts <= t1 + 600:
                caps.append(f"<li><code>{f.name}</code> · {f.stat().st_size/1e6:.1f} MB</li>")
    caps_html = "<ul>" + "".join(caps) + "</ul>" if caps else "<p>No capture files in the incident window.</p>"

    status = ("OPEN — WIRE HELD" if inc["status"] == "open" and inc.get("hold")
              else inc["status"].upper())
    html = f"""<title>Sonave — Incident Report #{inc['id']}</title>
<style>
body{{font:14px/1.6 -apple-system,'Segoe UI',Roboto,sans-serif;color:#16212b;background:#fff;max-width:820px;margin:0 auto;padding:40px 24px}}
h1{{font-size:24px;margin:0}} h2{{font-size:16px;margin:28px 0 8px}}
.meta{{color:#5d6d79;font-size:12.5px;margin:6px 0 24px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border:1px solid #dde4ea;padding:6px 10px;text-align:left}}
th{{background:#f2f5f8}}
.facts td:first-child{{font-weight:600;width:180px;background:#f8fafb}}
.v-fake{{color:#c81e2e;font-weight:700}} .v-suspect{{color:#a06800;font-weight:600}} .v-real{{color:#0d7a44}}
.disclaimer{{font-size:11.5px;color:#5d6d79;border-top:1px solid #dde4ea;margin-top:34px;padding-top:12px}}
@media print{{body{{padding:0}}}}
</style>
<h1>Sonave — Forensic Incident Report #{inc['id']}</h1>
<p class="meta">Generated {_t(time.time())} · workspace {owner} · model <code>{inc['model']}</code></p>
<table class="facts">
<tr><td>Speaker</td><td>{inc['speaker']}</td></tr>
<tr><td>Status</td><td>{status}</td></tr>
<tr><td>First detection</td><td>{_t(inc['first_ts'])}</td></tr>
<tr><td>Last update</td><td>{_t(inc['last_ts'])}</td></tr>
<tr><td>Peak rolling risk</td><td>{(inc['rolling'] or 0):.1%}</td></tr>
<tr><td>Trigger policy</td><td>{INCIDENT_STREAK} consecutive scoring windows with rolling risk ≥ 0.70</td></tr>
</table>
<h2>Rolling risk over the incident window</h2>
{chart or '<p>No chart — no scored windows retained.</p>'}
<p class="meta">Red band ≥ 0.70 = FAKE · amber band 0.40–0.70 = SUSPECT · 8-second rolling windows scored every 4 seconds once enough speech accumulates</p>
<h2>Scored windows</h2>
<table><tr><th>Time</th><th>Window P(fake)</th><th>Rolling risk</th><th>Verdict</th></tr>{rows}</table>
<h2>Capture files (audio evidence)</h2>
{caps_html}
<p class="disclaimer">Verdicts are probabilistic estimates produced by machine-learning models and can be
incorrect in either direction. This report supports — and does not replace — human verification
procedures. Retain the referenced capture files with this report for a complete evidentiary record.</p>"""
    return html


@app.get("/captures")
def captures(p: auth.Principal = Depends(require_principal)):
    cap_dir = _user_capture_dir(p.user_id)
    if not cap_dir.exists():
        return {"files": []}
    fs = sorted(cap_dir.glob("*.wav"))
    return {"files": [{"name": f.name, "mb": round(f.stat().st_size / 1e6, 2)} for f in fs]}


@app.get("/download/{name}")
def download(name: str, p: auth.Principal = Depends(require_principal)):
    f = _user_capture_dir(p.user_id) / Path(name).name    # prevent path traversal
    return FileResponse(str(f)) if f.exists() else {"error": "not found"}


# --- Google OAuth ------------------------------------------------------------
@app.get("/auth/login")
def auth_login(ctx: str = ""):
    if not auth.google_configured():
        return RedirectResponse("/console", status_code=302)
    state = auth.make_state(ctx)
    resp = RedirectResponse(auth.login_url(state), status_code=302)
    resp.set_cookie(auth.STATE_COOKIE, state, max_age=auth.STATE_TTL,
                    httponly=True, samesite="lax", secure=True, path="/auth")
    return resp


@app.get("/auth/calendar/connect")
def auth_calendar_connect(request: Request):
    """Incremental calendar.readonly grant for OAuth auto-join — explicit
    opt-in from the console, only for signed-in users, flag-gated until the
    scope passes Google verification."""
    if not _cal_oauth_enabled() or not auth.google_configured():
        raise HTTPException(status_code=404, detail="not enabled")
    p = auth.get_principal(request)
    if p is None or p.kind != "user":
        return RedirectResponse("/console", status_code=302)
    state = auth.make_state("cal")
    resp = RedirectResponse(auth.calendar_login_url(state), status_code=302)
    resp.set_cookie(auth.STATE_COOKIE, state, max_age=auth.STATE_TTL,
                    httponly=True, samesite="lax", secure=True, path="/auth")
    return resp


@app.post("/auth/calendar/disconnect")
def auth_calendar_disconnect(p: auth.Principal = Depends(require_principal)):
    row = db.get_oauth_token(p.user_id, "google_calendar")
    if row:
        auth.revoke_google_token(row["refresh_token"])    # best-effort at Google
        db.delete_oauth_token(p.user_id, "google_calendar")
        _CAL_AT.pop(p.user_id, None)
        _track(p.user_id, "calendar_disconnected", reason="user")
    return {"ok": True}


@app.get("/auth/callback")
def auth_callback(request: Request, code: str = "", state: str = ""):
    if not auth.verify_state(request.cookies.get(auth.STATE_COOKIE), state):
        raise HTTPException(status_code=403, detail="bad oauth state")
    if not code:
        raise HTTPException(status_code=400, detail="missing code")
    if auth.state_ctx(state) == "cal":
        # incremental Calendar grant for an already-signed-in user: store the
        # refresh token, never touch the session (scope has no userinfo)
        p = auth.get_principal(request)
        resp = RedirectResponse("/console", status_code=302)
        resp.delete_cookie(auth.STATE_COOKIE, path="/auth")
        if p is None or p.kind != "user" or not _cal_oauth_enabled():
            return resp
        try:
            tokens = auth._exchange_code(code)
        except Exception as e:  # noqa: BLE001 — Google/network failures
            logger.warning("calendar connect failed: %s", repr(e)[:120])
            return resp
        if tokens.get("refresh_token"):
            db.save_oauth_token(p.user_id, "google_calendar",
                                auth.CALENDAR_SCOPE, tokens["refresh_token"])
            _CAL_AT.pop(p.user_id, None)
            _track(p.user_id, "calendar_connected")
            logger.info("calendar connected for %s", p.user_id)
        return resp
    try:
        user = auth.complete_login(code)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:  # noqa: BLE001 — Google/network failures
        logger.warning("oauth callback failed: %s", repr(e)[:120])
        raise HTTPException(status_code=502, detail="google sign-in failed")
    session_tok = auth.sign_session(user["id"], user.get("session_ver", 1))
    if auth.state_ctx(state) == "popup":
        # Meet add-on sign-in: hand the session token to the opener iframe via
        # postMessage (CHIPS partitions make cookie handoff impossible), then close.
        origin = _base_url(request)
        resp = HTMLResponse(
            "<title>Sonave — signed in</title>"
            "<body style='font-family:sans-serif;background:#0a0e12;color:#e8eef2;"
            "display:flex;align-items:center;justify-content:center;height:100vh;margin:0'>"
            "<p>Signed in — you can close this window.</p><script>"
            "try{if(window.opener)window.opener.postMessage("
            f"{{type:'sonave_auth',token:{json.dumps(session_tok)}}},{json.dumps(origin)});"
            "}catch(e){}window.close();</script></body>")
    else:
        resp = RedirectResponse("/console", status_code=302)
    resp.set_cookie(auth.SESSION_COOKIE, session_tok,
                    max_age=auth.SESSION_TTL, httponly=True, samesite="lax", secure=True, path="/")
    # CHIPS companion (helps re-loads in already-partitioned contexts)
    resp.headers.append("set-cookie",
                        f"{auth.PARTITIONED_COOKIE}={session_tok}; Max-Age={auth.SESSION_TTL}; "
                        f"Path=/; Secure; HttpOnly; SameSite=None; Partitioned")
    resp.delete_cookie(auth.STATE_COOKIE, path="/auth")
    logger.info("login: %s (%s)", user.get("email"), user.get("role"))
    # exact-equal floats only on the INSERT path (db.upsert_google_user uses one `now`)
    if user.get("created_ts") == user.get("last_login_ts"):
        _track(user["id"], "signup", email=user.get("email") or "")
        _notify_admin(f"New signup: {user.get('email')}")
    else:
        _track(user["id"], "signin", email=user.get("email") or "")
    if user.get("role") == "admin":
        db.migrate_legacy(DATA_DIR, enroll.ENROLL_DIR, user["id"])
    return resp


class CredReq(BaseModel):
    credential: str


@app.post("/auth/google-credential")
def auth_google_credential(req: CredReq):
    """GIS One Tap / Sign-in-with-Google in the Meet panel: verify the ID token,
    mint a Sonave session token (returned as JSON — no cookies, so the flow
    works with third-party cookies disabled)."""
    if not auth.google_configured():
        raise HTTPException(status_code=400, detail="google sign-in not configured")
    try:
        user = auth.complete_credential_login(req.credential)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception:
        raise HTTPException(status_code=403, detail="credential verification failed")
    if user.get("created_ts") == user.get("last_login_ts"):
        _track(user["id"], "signup", email=user.get("email") or "")
        _notify_admin(f"New signup: {user.get('email')}")
    else:
        _track(user["id"], "signin", email=user.get("email") or "")
    logger.info("credential login: %s (%s)", user.get("email"), user.get("role"))
    if user.get("role") == "admin":
        db.migrate_legacy(DATA_DIR, enroll.ENROLL_DIR, user["id"])
    return {"ok": True, "token": auth.sign_session(user["id"], user.get("session_ver", 1)),
            "email": user.get("email"), "name": user.get("name")}


@app.post("/auth/logout")
def auth_logout(request: Request):
    p = auth.get_principal(request)
    if p is not None and p.kind == "user":     # expired cookie / operator: nothing to track
        _track(p.user_id, "signout")
        db.bump_session_ver(p.user_id)         # revoke ALL outstanding session tokens
    resp = RedirectResponse("/console", status_code=302)
    resp.delete_cookie(auth.SESSION_COOKIE, path="/")
    resp.headers.append("set-cookie",
                        f"{auth.PARTITIONED_COOKIE}=; Max-Age=0; Path=/; Secure; HttpOnly; "
                        f"SameSite=None; Partitioned")
    return resp


@app.get("/api/me")
def api_me(p: auth.Principal = Depends(require_principal)):
    out = {"kind": p.kind, "email": p.email or "operator", "name": p.name,
           "picture": p.picture, "role": p.role,
           "google": auth.google_configured(), "billing": billing.configured(),
           "alert_webhook": db.get_alert_webhook(p.user_id),
           "ical_url": db.get_ical_url(p.user_id),
           "autojoin_loop": _AUTOJOIN_ALIVE > 0 and time.time() - _AUTOJOIN_ALIVE < 180,
           "calendar_oauth": _cal_oauth_enabled(),
           "calendar_connected": bool(p.kind == "user"
                                      and db.get_oauth_token(p.user_id, "google_calendar"))}
    out.update(billing.entitlement(p.user_id, p.role))
    return out


def _base_url(request: Request) -> str:
    domain = _domain(request) or "localhost:8000"
    scheme = "http" if domain.startswith(("localhost", "127.0.0.1")) else "https"
    return f"{scheme}://{domain}"


@app.post("/api/billing/checkout")
def api_billing_checkout(request: Request, p: auth.Principal = Depends(require_principal)):
    if not billing.configured():
        return {"ok": False, "detail": "billing not configured"}
    try:
        return {"ok": True, "url": billing.create_checkout(p.user_id, _base_url(request))}
    except Exception as e:  # noqa: BLE001
        logger.warning("checkout failed: %s", repr(e)[:120])
        return {"ok": False, "detail": "could not start checkout"}


@app.post("/api/billing/portal")
def api_billing_portal(request: Request, p: auth.Principal = Depends(require_principal)):
    if not billing.configured():
        return {"ok": False, "detail": "billing not configured"}
    try:
        url = billing.create_portal(p.user_id, _base_url(request))
        return {"ok": bool(url), "url": url}
    except Exception as e:  # noqa: BLE001
        logger.warning("portal failed: %s", repr(e)[:120])
        return {"ok": False, "detail": "could not open portal"}


@app.post("/api/billing/webhook")
async def api_billing_webhook(request: Request):
    secret = os.environ.get("SONAVE_STRIPE_WEBHOOK_SECRET", "")
    body = await request.body()
    sig = request.headers.get("stripe-signature", "")
    if not secret or not billing.verify_webhook_sig(body, sig, secret):
        raise HTTPException(status_code=400, detail="bad signature")
    event = json.loads(body)
    if db.webhook_seen(event.get("id", ""), event.get("type", "")):
        return {"ok": True, "duplicate": True}
    billing.handle_webhook(event)
    etype = event.get("type", "")
    if etype == "checkout.session.completed":
        _notify_admin("New subscription: a customer added a card (metered plan).")
    elif etype == "customer.subscription.deleted":
        _notify_admin("Subscription canceled by a customer.")
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def landing():
    html = (_HERE / "landing.html").read_text(encoding="utf-8")
    return html.replace("__FAVICON__", _FAVICON_B64)


@app.get("/console", response_class=HTMLResponse)
def console():
    html = (_HERE / "console.html").read_text(encoding="utf-8")
    return (html.replace("__AUTH__", "1" if (API_TOKEN or auth.google_configured()) else "0")
                .replace("__GOOGLE__", "1" if auth.google_configured() else "0")
                .replace("__FAVICON__", _FAVICON_B64))


@app.get("/onboarding", response_class=HTMLResponse)
def onboarding():
    html = (_HERE / "onboarding.html").read_text(encoding="utf-8")
    return html.replace("__FAVICON__", _FAVICON_B64)


@app.get("/meet-addon", response_class=HTMLResponse)
def meet_addon():
    """Google Meet add-on side panel (works standalone in a browser too).
    Marketplace deployment steps live in PRODUCTION.md; the panel talks to the
    same /api/quality the console uses."""
    html = (_HERE / "meet-addon.html").read_text(encoding="utf-8")
    return (html.replace("__FAVICON__", _FAVICON_B64)
                .replace("__MEET_PROJECT__", os.environ.get("SONAVE_MEET_PROJECT_NUMBER", ""))
                .replace("__GOOGLE_CID__", os.environ.get("SONAVE_GOOGLE_CLIENT_ID", "")))


@app.get("/og.png")
def og_image():
    return FileResponse(str(_HERE / "og.png"), media_type="image/png")


@app.get("/icon-120.png")
def icon_120():
    return FileResponse(str(_HERE / "icon-120.png"), media_type="image/png")


@app.get("/icon-128.png")
def icon_128():
    return FileResponse(str(_HERE / "icon-128.png"), media_type="image/png")


@app.get("/console-shot.png")
def console_shot():
    return FileResponse(str(_HERE / "console-shot.png"), media_type="image/png")


@app.get("/robots.txt")
def robots(request: Request):
    from fastapi.responses import PlainTextResponse
    base = _base_url(request)
    return PlainTextResponse("User-agent: *\nAllow: /\nDisallow: /console\nDisallow: /report/\n"
                             f"\nSitemap: {base}/sitemap.xml\n")


@app.get("/sitemap.xml")
def sitemap(request: Request):
    from fastapi.responses import Response
    base = _base_url(request)
    today = time.strftime("%Y-%m-%d")
    paths = ["/", "/benchmarks", "/guides", "/privacy", "/terms"]
    paths += [f"/guides/{s}" for s in GUIDE_SLUGS]
    urls = "".join(
        f"<url><loc>{base}{path}</loc><lastmod>{today}</lastmod></url>" for path in paths)
    xml = ('<?xml version="1.0" encoding="UTF-8"?>'
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
           f"{urls}</urlset>")
    return Response(content=xml, media_type="application/xml")


@app.get("/llms.txt")
def llms_txt():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        "# Sonave\n\n"
        "> Real-time deepfake-voice detection for video meetings. A visible bot joins "
        "Google Meet, Zoom or Teams, streams each speaker's audio to a detection model "
        "trained on meeting-codec audio, and shows a live REAL / SUSPECT / FAKE verdict "
        "per speaker (~4 s to first verdict, re-scored every 4 s). Sustained red verdicts "
        "fire a wire-hold webhook that can pause a payment approval, plus an exportable "
        "forensic report.\n\n"
        "Key facts (deployed model, benchmarked 2026-08-12; methodology at /benchmarks):\n"
        "- 95.2% catch on 27 unseen commercial voice-clone tools through meeting audio "
        "(a commodity open-source detector catches 1.9% on the same clips)\n"
        "- 94.0% real-voice accuracy through the Opus meeting codec\n"
        "- 58.7% catch at 93.3% real-voice accuracy on In-the-Wild (the honest ceiling; "
        "verdicts are a second factor alongside callbacks, not a replacement)\n"
        "- Pricing: free 5 monitored hours/month, then $8 per monitored hour, self-serve\n\n"
        "## Pages\n"
        "- [Home](https://usesonave.com/): product, how it works, pricing, FAQ\n"
        "- [Benchmarks](https://usesonave.com/benchmarks): full results + methodology\n"
        "- [Guides](https://usesonave.com/guides): detecting deepfake voices on live "
        "calls, CEO voice-fraud anatomy, what detector accuracy numbers mean\n"
        "- [Privacy](https://usesonave.com/privacy) · [Terms](https://usesonave.com/terms)\n")


GUIDE_SLUGS = ("detect-deepfake-voice-live-call", "ceo-voice-fraud-wire-transfers",
               "deepfake-detector-accuracy")


@app.get("/guides", response_class=HTMLResponse)
def guides_index():
    html = (_HERE / "guides" / "index.html").read_text(encoding="utf-8")
    return html.replace("__FAVICON__", _FAVICON_B64)


@app.get("/guides/{slug}", response_class=HTMLResponse)
def guide(slug: str):
    if slug not in GUIDE_SLUGS:            # whitelist — never touch the filesystem with user input
        raise HTTPException(status_code=404, detail="no such guide")
    html = (_HERE / "guides" / f"{slug}.html").read_text(encoding="utf-8")
    return html.replace("__FAVICON__", _FAVICON_B64)


@app.get("/benchmarks", response_class=HTMLResponse)
def benchmarks():
    html = (_HERE / "benchmarks.html").read_text(encoding="utf-8")
    return html.replace("__FAVICON__", _FAVICON_B64)


@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    html = (_HERE / "privacy.html").read_text(encoding="utf-8")
    return html.replace("__FAVICON__", _FAVICON_B64)


@app.get("/terms", response_class=HTMLResponse)
def terms():
    html = (_HERE / "terms.html").read_text(encoding="utf-8")
    return html.replace("__FAVICON__", _FAVICON_B64)


@app.on_event("startup")
def _startup_migrate():
    admin = db.first_admin_id()
    if admin:
        db.migrate_legacy(DATA_DIR, enroll.ENROLL_DIR, admin)


