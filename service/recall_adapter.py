"""
recall_adapter.py — Phase 3 capture: connect Recall.ai → the orchestrator.

Recall.ai runs the meeting bot (joins Meet/Zoom/Teams, streams audio). This adapter
(1) tells Recall to send a bot to a meeting, and (2) receives Recall's real-time audio
and feeds it into `Orchestrator.ingest()` — the same path the offline demo uses.

WHERE THE API KEY GOES: environment variable `SONAVE_RECALL_API_KEY` (see .env.example).
Never hardcode it. Read here via os.environ; nothing else in the repo needs it.

STATUS: ready-to-wire STUB. The bot-creation call is real; the exact real-time audio
payload shape depends on how you configure Recall's real-time endpoints, so the
webhook handler has clearly-marked TODOs for the 2–3 field names to confirm against
your Recall dashboard / docs. Everything downstream (scoring, rolling per-speaker
state, alerts, wire-hold) already works.
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np


def _load_dotenv():
    """Load repo-root .env into os.environ (without overriding already-set vars)."""
    f = Path(__file__).resolve().parent.parent / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# --- config (from env / .env) ------------------------------------------------
API_KEY = os.environ.get("SONAVE_RECALL_API_KEY")
BASE = os.environ.get("SONAVE_RECALL_BASE", "https://us-west-2.recall.ai/api/v1")
# Real-time raw audio streams over a WEBSOCKET (wss://), per Recall docs. The Svix
# webhook only carries transcript/participant events, not raw audio.
WS_URL = os.environ.get("SONAVE_RECALL_WS")          # wss://<host>/api/ws/audio
WEBHOOK = os.environ.get("SONAVE_RECALL_WEBHOOK")    # optional: async lifecycle events


def _require_key():
    if not API_KEY:
        raise RuntimeError(
            "SONAVE_RECALL_API_KEY is not set. Put it in a .env file (see "
            ".env.example) and load it into the environment before starting the service.")


# --- 1. send a bot to a meeting ---------------------------------------------
def create_bot(meeting_url: str, bot_name: str = "Sonave") -> dict:
    """Ask Recall to join a meeting and stream real-time audio to our webhook."""
    import json
    import urllib.request

    _require_key()
    if not WS_URL:
        raise RuntimeError(
            "SONAVE_RECALL_WS is not set. Real-time audio needs a public wss:// URL "
            "to your /api/ws/audio endpoint (e.g. ngrok → wss://<id>.ngrok.app/api/ws/audio). "
            "Put it in .env.")
    # Real-time raw audio → realtime_endpoints of type 'websocket'. NOTE: the
    # audio_separate_raw artifact must be ENABLED in recording_config before you can
    # subscribe to its .data events (Recall rejects the sub otherwise with 400).
    payload = {
        "meeting_url": meeting_url,
        "bot_name": bot_name,
        "recording_config": {
            "audio_separate_raw": {},                       # enable per-participant raw audio
            "realtime_endpoints": [
                {"type": "websocket", "url": WS_URL,
                 "events": ["audio_separate_raw.data"]}
            ],
        },
    }
    req = urllib.request.Request(
        f"{BASE}/bot", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Token {API_KEY}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


# --- 2. receive real-time audio → orchestrator ------------------------------
def handle_audio_event(event: dict, orch) -> dict | None:
    """
    Translate one Recall real-time `audio_separate_raw.data` event into a streaming
    ingest. Schema (from Recall docs): buffer is base64 raw S16LE 16 kHz mono at
    data.data.buffer; speaker at data.data.participant.name; time at
    data.data.timestamp.relative.
    """
    if event.get("event") not in (None, "audio_separate_raw.data",
                                  "audio_mixed_raw.data"):
        return {"status": "ignored", "event": event.get("event")}
    d = (event.get("data") or {}).get("data") or {}
    part = d.get("participant") or {}
    speaker = part.get("name") or (f"participant_{part.get('id')}"
                                   if part.get("id") is not None else "unknown")
    ts = float((d.get("timestamp") or {}).get("relative", 0.0))
    buf_b64 = d.get("buffer")
    if not buf_b64:
        return None
    # raw S16LE PCM, 16 kHz mono -> float32
    pcm = np.frombuffer(base64.b64decode(buf_b64), dtype=np.int16).astype(np.float32) / 32768.0
    return orch.ingest_stream(speaker, pcm, ts)


def _to_mono16k(raw: bytes) -> np.ndarray | None:
    """Decode a real-time audio buffer to mono 16 kHz float. Handles wav or raw PCM16."""
    import librosa
    import soundfile as sf
    try:
        wav, sr = sf.read(io.BytesIO(raw))          # if it's a wav container
    except Exception:
        # assume raw 16-bit PCM @ 16 kHz mono (Recall's raw audio default is 16k)
        wav = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        sr = 16000
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = librosa.resample(np.asarray(wav, dtype="float32"), orig_sr=sr, target_sr=16000)
    return np.asarray(wav, dtype="float32")


# --- 3. async webhook (audio_separate.done etc.) → download & analyze ---------
def handle_async_event(event: dict, orch, state: dict | None = None) -> dict:
    """
    Handle a Svix lifecycle webhook. For per-participant "audio ready" events we
    download each participant's audio and score it into the orchestrator.

    Recall's exact payload for audio_separate.done isn't hardcoded here — capture a
    real one at GET /api/recall/events and confirm these lookups:
      - event type      : event["event"]  (e.g. "audio_separate.done")
      - download URL(s)  : per-participant audio URL (may need a follow-up API GET
                           on the bot/recording id to fetch media URLs)
      - participant name : the speaker label for each track
    """
    etype = event.get("event") or event.get("type") or ""
    if "audio_separate.done" not in etype and "audio_mixed.done" not in etype:
        return {"status": "ignored", "event": etype}

    tracks = _extract_audio_tracks(event)        # [(speaker, url)]
    if not tracks:
        return {"status": "captured",
                "note": "no download URLs found — inspect /api/recall/events and "
                        "update _extract_audio_tracks()."}

    scored = 0
    for speaker, url in tracks:
        wav = _download_wav(url)
        if wav is None:
            continue
        _score_stream_into(orch, speaker, wav)
        scored += 1
    if state is not None:
        state.setdefault("reports", []).append(
            {"event": etype, "speakers_scored": scored})
    return {"status": "analyzed", "speakers_scored": scored,
            "current": orch.status()}


def _extract_audio_tracks(event: dict):
    """Return [(speaker_name, download_url)]. TODO: match Recall's real schema."""
    data = event.get("data", {})
    # common shapes to try (confirm against a captured payload):
    tracks = []
    for t in (data.get("tracks") or data.get("audio") or []):
        url = t.get("download_url") or t.get("url")
        spk = (t.get("participant", {}) or {}).get("name") or t.get("speaker") or "spk"
        if url:
            tracks.append((spk, url))
    single = data.get("download_url") or data.get("url")
    if single and not tracks:
        tracks.append((data.get("participant", {}).get("name", "mixed"), single))
    return tracks


def _download_wav(url: str):
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Token {API_KEY}"}
                                     if API_KEY else {})
        with urllib.request.urlopen(req, timeout=60) as r:
            return _to_mono16k(r.read())
    except Exception:
        return None


def _score_stream_into(orch, speaker: str, wav: np.ndarray, hop_s: float = 2.0):
    """Window a full track and feed each window to the orchestrator for this speaker."""
    import model_sls
    win = model_sls.MAX_LEN
    hop = int(hop_s * 16000)
    for s in range(0, max(1, len(wav) - win + 1), hop):
        chunk = wav[s:s + win]
        if np.sqrt(np.mean(chunk ** 2)) >= 0.005:
            orch.ingest(speaker, chunk, s / 16000)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Send a Sonave bot to a meeting")
    ap.add_argument("meeting_url")
    args = ap.parse_args()
    print(create_bot(args.meeting_url))
