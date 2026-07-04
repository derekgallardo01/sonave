"""
orchestrator.py — the live per-speaker scoring engine (Phase 4 + Phase 5 core).

This is what sits between the capture layer and the product surface. It ingests a
stream of (speaker_id, audio_chunk, timestamp) events — exactly what Recall.ai hands
us live — scores each chunk, maintains a ROLLING per-speaker confidence (so one noisy
window can't trip an alarm), and fires:
  - an ALERT callback when a speaker crosses into "fake" (Slack/SMS/email in prod)
  - a HOLD callback (the wire-fraud "pause the transaction / require re-auth" hook)

It's transport-agnostic and testable offline: `simulate_meeting()` replays a recording
+ speaker turns through the exact same path the live stream will use.

Trigger-on-suspicion (cost control): scoring every chunk is cheap (our own model), so
we score continuously; the ALERT/HOLD escalation only fires on a sustained red.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detector  # noqa: E402
import model_sls  # noqa: E402


@dataclass
class SpeakerState:
    rolling: float = 0.0
    n: int = 0
    verdict: str = "real"
    alerted: bool = False
    peak: float = 0.0
    buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    verify_buf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    voiceprint_sim: float | None = None      # speaker-verification similarity


VERIFY_MIN = 8 * model_sls.SR     # need >=8 s of a speaker before verifying (short
VERIFY_MAX = 20 * model_sls.SR    # clips give noisy embeddings); keep the last ~20 s


def _enroll_mod():
    """Lazily import enrollment; None if resemblyzer isn't installed."""
    try:
        import enroll
        return enroll
    except Exception:
        return None


STREAM_WIN = model_sls.MAX_LEN          # 4 s scoring window
STREAM_HOP = 2 * model_sls.SR           # 2 s hop


class Orchestrator:
    def __init__(self, on_alert=None, on_hold=None, alpha: float = 0.35):
        self.alpha = alpha
        self.on_alert = on_alert or _console_alert
        self.on_hold = on_hold          # optional; wired for the finance vertical
        self.speakers: dict[str, SpeakerState] = {}

    def ingest(self, speaker_id: str, wav: np.ndarray, ts: float) -> dict:
        """Score one chunk for a speaker; update rolling state; escalate if red.

        If the speaker is ENROLLED, fuse the deepfake score with a voiceprint check
        (max of 'looks synthetic' and 'not the claimed person') — so impersonation
        is caught even when the audio is acoustically clean, and real enrolled voices
        aren't falsely flagged.
        """
        p = detector.score_array(wav)["p_fake"]
        st = self.speakers.setdefault(speaker_id, SpeakerState(rolling=p))
        score = p
        en = _enroll_mod()
        if en is not None and en.is_enrolled(speaker_id):
            # accumulate audio for a RELIABLE voiceprint check (4 s is too short)
            st.verify_buf = np.concatenate([st.verify_buf, wav])[-VERIFY_MAX:]
            if len(st.verify_buf) >= VERIFY_MIN:
                fr = en.fused_risk(p, speaker_id, st.verify_buf)
                score = fr["risk"]
                st.voiceprint_sim = (fr.get("speaker_check") or {}).get("similarity")
        st.rolling = self.alpha * score + (1 - self.alpha) * st.rolling if st.n else score
        st.n += 1
        st.peak = max(st.peak, st.rolling)
        st.verdict = detector.verdict(st.rolling)

        if st.verdict == "fake" and not st.alerted:
            st.alerted = True
            event = {"speaker_id": speaker_id, "ts": ts,
                     "rolling_p_fake": round(st.rolling, 3),
                     "model_version": detector.MODEL_VERSION}
            self.on_alert(event)
            if self.on_hold:
                self.on_hold(event)      # pause the wire / require re-auth
        return {"speaker_id": speaker_id, "ts": ts, "p_fake": round(p, 3),
                "rolling_p_fake": round(st.rolling, 3), "verdict": st.verdict,
                "voiceprint_sim": st.voiceprint_sim}

    def ingest_stream(self, speaker_id: str, pcm: np.ndarray, ts: float) -> dict:
        """Accumulate streamed real-time audio per speaker; score each full window.

        Real-time events carry small buffers (tens of ms), so we buffer per speaker
        until we have a 4 s window, score it, slide by the hop, and repeat.
        """
        st = self.speakers.setdefault(speaker_id, SpeakerState())
        st.buffer = np.concatenate([st.buffer, pcm]) if st.buffer.size else pcm
        last = None
        while len(st.buffer) >= STREAM_WIN:
            win = st.buffer[:STREAM_WIN]
            st.buffer = st.buffer[STREAM_HOP:]
            if np.sqrt(np.mean(win ** 2)) >= 0.005:      # skip silence
                last = self.ingest(speaker_id, win, ts)
        return last or {"speaker_id": speaker_id, "ts": ts,
                        "verdict": st.verdict, "rolling_p_fake": round(st.rolling, 3),
                        "buffering": True}

    def status(self) -> dict:
        """Snapshot for the live dashboard (per-speaker RAG)."""
        en = _enroll_mod()
        return {spk: {"verdict": s.verdict, "rolling_p_fake": round(s.rolling, 3),
                      "peak": round(s.peak, 3), "windows": s.n,
                      "enrolled": bool(en and en.is_enrolled(spk)),
                      "voiceprint_sim": s.voiceprint_sim}
                for spk, s in self.speakers.items()}


# --- Alert / hold hooks ------------------------------------------------------
def _console_alert(event: dict):
    print(f"[ALERT] speaker '{event['speaker_id']}' flagged FAKE at "
          f"{event['ts']:.1f}s (rolling P(fake)={event['rolling_p_fake']})")


def webhook_poster(url: str, kind: str = "alert"):
    """Return a hook that POSTs the event JSON to a URL (Slack, payments API, ...)."""
    import json
    import urllib.request

    def _post(event: dict):
        body = json.dumps({"type": kind, **event}).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:  # noqa: BLE001 — never let alerting crash detection
            print(f"[{kind}] webhook failed: {e}")
    return _post


# --- Offline replay (same path as live) --------------------------------------
def simulate_meeting(audio_path: str, segments_path: str, hop: float = 2.0,
                     orch: "Orchestrator | None" = None) -> dict:
    """Replay a recording + speaker turns through the orchestrator in time order."""
    import csv
    import librosa

    orch = orch or Orchestrator()
    wav, _ = librosa.load(audio_path, sr=model_sls.SR, mono=True)
    with open(segments_path, newline="", encoding="utf-8") as f:
        segs = [(float(r["start"]), float(r["end"]), r["speaker"])
                for r in csv.DictReader(f)]

    WIN = model_sls.MAX_LEN
    hop_n = int(hop * model_sls.SR)
    events = []
    for s in range(0, max(1, len(wav) - WIN + 1), hop_n):
        t = (s + WIN // 2) / model_sls.SR
        spk = next((sp for a, b, sp in segs if a <= t < b), None)
        if spk is None:
            continue
        chunk = wav[s:s + WIN]
        if np.sqrt(np.mean(chunk ** 2)) < 0.005:     # skip silence
            continue
        events.append(orch.ingest(spk, chunk, t))
    return {"final_status": orch.status(), "events": events}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Replay a recording through the live engine")
    ap.add_argument("audio")
    ap.add_argument("segments", help="CSV: start,end,speaker")
    ap.add_argument("--hop", type=float, default=2.0)
    ap.add_argument("--alert-webhook", default=None)
    ap.add_argument("--hold-webhook", default=None)
    args = ap.parse_args()

    orch = Orchestrator(
        on_alert=webhook_poster(args.alert_webhook, "alert") if args.alert_webhook else None,
        on_hold=webhook_poster(args.hold_webhook, "hold") if args.hold_webhook else None,
    )
    out = simulate_meeting(args.audio, args.segments, args.hop, orch)
    print("\n=== final per-speaker status ===")
    for spk, s in out["final_status"].items():
        print(f"  {spk:>8}: {s['verdict'].upper():8} "
              f"rolling P(fake)={s['rolling_p_fake']} peak={s['peak']} "
              f"({s['windows']} windows)")
