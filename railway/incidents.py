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
_COLS = ["id", "speaker", "first_ts", "last_ts", "rolling", "model", "status", "hold", "user_id"]


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH), timeout=10)
    c.execute("""CREATE TABLE IF NOT EXISTS incidents(
        id INTEGER PRIMARY KEY AUTOINCREMENT, speaker TEXT, first_ts REAL, last_ts REAL,
        rolling REAL, model TEXT, status TEXT DEFAULT 'open', hold INTEGER DEFAULT 0)""")
    cols = [r[1] for r in c.execute("PRAGMA table_info(incidents)").fetchall()]
    if "user_id" not in cols:                       # additive migration, idempotent
        c.execute("ALTER TABLE incidents ADD COLUMN user_id TEXT")
    return c


def record(speaker: str, rolling: float, model: str, user_id: str | None = None) -> dict | None:
    """Open a fresh incident for `speaker` in this workspace if none is currently open
    (dedup), else just refresh it. Returns the new incident or None if already open."""
    now = time.time()
    with _LOCK:
        c = _conn()
        try:
            if c.execute("SELECT id FROM incidents WHERE speaker=? AND status='open' "
                         "AND (user_id=? OR (user_id IS NULL AND ? IS NULL))",
                         (speaker, user_id, user_id)).fetchone():
                c.execute("UPDATE incidents SET last_ts=?, rolling=? WHERE speaker=? AND status='open' "
                          "AND (user_id=? OR (user_id IS NULL AND ? IS NULL))",
                          (now, rolling, speaker, user_id, user_id))
                c.commit()
                return None
            cur = c.execute(
                "INSERT INTO incidents(speaker,first_ts,last_ts,rolling,model,status,hold,user_id) "
                "VALUES(?,?,?,?,?, 'open', ?, ?)",
                (speaker, now, now, rolling, model, 1 if WIRE_HOLD else 0, user_id))
            c.commit()
            return {"id": cur.lastrowid, "speaker": speaker, "rolling": rolling,
                    "model": model, "hold": WIRE_HOLD, "user_id": user_id}
        finally:
            c.close()


def list_incidents(limit: int = 50, user_id: str | None = None) -> list[dict]:
    """user_id=None -> all incidents (admin / single-tenant view)."""
    c = _conn()
    try:
        if user_id is None:
            rows = c.execute(f"SELECT {','.join(_COLS)} FROM incidents ORDER BY id DESC LIMIT ?",
                             (limit,)).fetchall()
        else:
            rows = c.execute(f"SELECT {','.join(_COLS)} FROM incidents WHERE user_id=? "
                             "ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()
    finally:
        c.close()
    return [dict(zip(_COLS, r)) for r in rows]


def get_incident(incident_id: int, user_id: str | None = None) -> dict | None:
    """Fetch one incident; user_id=None -> unrestricted (admin)."""
    c = _conn()
    try:
        if user_id is None:
            r = c.execute(f"SELECT {','.join(_COLS)} FROM incidents WHERE id=?",
                          (incident_id,)).fetchone()
        else:
            r = c.execute(f"SELECT {','.join(_COLS)} FROM incidents WHERE id=? AND user_id=?",
                          (incident_id, user_id)).fetchone()
        return dict(zip(_COLS, r)) if r else None
    finally:
        c.close()


def acknowledge(incident_id: int, user_id: str | None = None) -> bool:
    """user_id=None -> unrestricted (admin); else only that workspace's incident."""
    with _LOCK:
        c = _conn()
        try:
            if user_id is None:
                cur = c.execute("UPDATE incidents SET status='acknowledged', hold=0 WHERE id=?",
                                (incident_id,))
            else:
                cur = c.execute("UPDATE incidents SET status='acknowledged', hold=0 "
                                "WHERE id=? AND user_id=?", (incident_id, user_id))
            c.commit()
            return cur.rowcount > 0
        finally:
            c.close()


def notify(event: dict, webhook: str | None = None) -> None:
    """Fire the alert webhook (Slack `{text}` block). Uses the per-workspace webhook
    when given, else the global env fallback. Best-effort — never raises."""
    url = webhook or ALERT_WEBHOOK
    if not url:
        return
    held = " Wire *HELD* — re-authenticate the caller before releasing funds." if event.get("hold") else ""
    text = (f":rotating_light: *Sonave — suspected deepfake voice*\n"
            f"Speaker *{event['speaker']}* · risk *{event['rolling']:.0%}* · model `{event['model']}`."
            + held)
    body = json.dumps({"text": text}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}), timeout=10)
    except Exception:
        pass
