"""Incident store + alerting for the capture service (Track 3 fraud workflow).

Torch-free by design: railway is CPU-only, so this layer runs on the verdicts the
service already receives from the hosted scorer (Modal). When a speaker's rolling
verdict is a sustained "fake", we open an incident (persisted to SQLite on the /data
volume, so it survives redeploys), fire an alert webhook (Slack-formatted), and flag a
wire-hold that a human must acknowledge. Full voiceprint-fusion is a GPU-side follow-up.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import urllib.request
from pathlib import Path

# incidents.db sits next to the capture volume so it persists across Railway redeploys.
DB_PATH = Path(os.environ.get("SONAVE_INCIDENT_DB",
                              str(Path(os.environ.get("SONAVE_DATA_DIR", "/data/captured")).parent
                                  / "incidents.db")))
ALERT_WEBHOOK = os.environ.get("SONAVE_ALERT_WEBHOOK", "")    # Slack (or compatible) incoming webhook
WIRE_HOLD = os.environ.get("SONAVE_WIRE_HOLD", "1") != "0"    # a sustained fake holds the wire
_LOCK = threading.Lock()
_COLS = ["id", "speaker", "first_ts", "last_ts", "rolling", "model", "status", "hold"]


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH), timeout=10)
    c.execute("""CREATE TABLE IF NOT EXISTS incidents(
        id INTEGER PRIMARY KEY AUTOINCREMENT, speaker TEXT, first_ts REAL, last_ts REAL,
        rolling REAL, model TEXT, status TEXT DEFAULT 'open', hold INTEGER DEFAULT 0)""")
    return c


def record(speaker: str, rolling: float, model: str) -> dict | None:
    """Open a fresh incident for `speaker` if none is currently open (dedup), else just
    refresh it. Returns the new incident (to alert on) or None if one was already open."""
    now = time.time()
    with _LOCK:
        c = _conn()
        try:
            if c.execute("SELECT id FROM incidents WHERE speaker=? AND status='open'",
                         (speaker,)).fetchone():
                c.execute("UPDATE incidents SET last_ts=?, rolling=? WHERE speaker=? AND status='open'",
                          (now, rolling, speaker))
                c.commit()
                return None
            cur = c.execute(
                "INSERT INTO incidents(speaker,first_ts,last_ts,rolling,model,status,hold) "
                "VALUES(?,?,?,?,?, 'open', ?)",
                (speaker, now, now, rolling, model, 1 if WIRE_HOLD else 0))
            c.commit()
            return {"id": cur.lastrowid, "speaker": speaker, "rolling": rolling,
                    "model": model, "hold": WIRE_HOLD}
        finally:
            c.close()


def list_incidents(limit: int = 50) -> list[dict]:
    c = _conn()
    try:
        rows = c.execute(f"SELECT {','.join(_COLS)} FROM incidents ORDER BY id DESC LIMIT ?",
                         (limit,)).fetchall()
    finally:
        c.close()
    return [dict(zip(_COLS, r)) for r in rows]


def acknowledge(incident_id: int) -> bool:
    with _LOCK:
        c = _conn()
        try:
            cur = c.execute("UPDATE incidents SET status='acknowledged', hold=0 WHERE id=?",
                            (incident_id,))
            c.commit()
            return cur.rowcount > 0
        finally:
            c.close()


def notify(event: dict) -> None:
    """Fire the alert webhook (Slack `{text}` block). Best-effort — never raises."""
    if not ALERT_WEBHOOK:
        return
    held = " Wire *HELD* — re-authenticate the caller before releasing funds." if event.get("hold") else ""
    text = (f":rotating_light: *Sonave — suspected deepfake voice*\n"
            f"Speaker *{event['speaker']}* · risk *{event['rolling']:.0%}* · model `{event['model']}`."
            + held)
    body = json.dumps({"text": text}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(
            ALERT_WEBHOOK, data=body, headers={"Content-Type": "application/json"}), timeout=10)
    except Exception:
        pass
