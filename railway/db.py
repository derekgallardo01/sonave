"""db.py — the app database (/data/app.db): users, bots, billing tables.

Same conventions as incidents.py: stdlib sqlite3, connection per call, lazy
directory creation (no import-time I/O — tests import with no volume present).
Single-web-worker deployment assumed (also required by app.py's in-memory state).
"""
from __future__ import annotations

import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    google_sub    TEXT UNIQUE,
    email         TEXT,
    name          TEXT,
    picture       TEXT,
    role          TEXT DEFAULT 'member',
    session_ver   INTEGER DEFAULT 1,
    created_ts    REAL,
    last_login_ts REAL
);
CREATE TABLE IF NOT EXISTS oauth_tokens (
    user_id       TEXT,
    provider      TEXT,
    scopes        TEXT,
    refresh_token TEXT,
    updated_ts    REAL,
    PRIMARY KEY (user_id, provider)
);
CREATE TABLE IF NOT EXISTS bots (
    bot_id      TEXT PRIMARY KEY,
    user_id     TEXT,
    token_hash  TEXT,
    meeting_url TEXT,
    created_ts  REAL,
    started_ts  REAL,
    ended_ts    REAL,
    metered_sec REAL DEFAULT 0,
    status      TEXT DEFAULT 'created'
);
CREATE INDEX IF NOT EXISTS idx_bots_token ON bots(token_hash);
CREATE TABLE IF NOT EXISTS subscriptions (
    user_id                TEXT PRIMARY KEY,
    stripe_customer_id     TEXT,
    stripe_subscription_id TEXT,
    status                 TEXT,
    updated_ts             REAL
);
CREATE TABLE IF NOT EXISTS usage (
    user_id    TEXT,
    month      TEXT,
    minutes    REAL DEFAULT 0,
    updated_ts REAL,
    PRIMARY KEY (user_id, month)
);
CREATE TABLE IF NOT EXISTS webhook_events (
    id         TEXT PRIMARY KEY,
    type       TEXT,
    created_ts REAL
);
CREATE TABLE IF NOT EXISTS scores (
    user_id TEXT,
    speaker TEXT,
    ts      REAL,
    p_fake  REAL,
    rolling REAL,
    verdict TEXT
);
CREATE INDEX IF NOT EXISTS idx_scores_user_spk_ts ON scores(user_id, speaker, ts);
"""


def _db_path() -> Path:
    return Path(os.environ.get("SONAVE_APP_DB", "/data/app.db"))


def _conn() -> sqlite3.Connection:
    p = _db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p, timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript(_SCHEMA)
    cols = [r[1] for r in c.execute("PRAGMA table_info(users)").fetchall()]
    if "alert_webhook" not in cols:                  # additive migration, idempotent
        c.execute("ALTER TABLE users ADD COLUMN alert_webhook TEXT")
    return c


# --- users -------------------------------------------------------------------
def get_user(uid: str) -> dict | None:
    c = _conn()
    try:
        r = c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        return dict(r) if r else None
    finally:
        c.close()


def upsert_google_user(sub: str, email: str, name: str, picture: str, role: str) -> dict:
    """Create-or-update by google_sub; role only ever upgrades to admin."""
    now = time.time()
    with _LOCK:
        c = _conn()
        try:
            r = c.execute("SELECT * FROM users WHERE google_sub=?", (sub,)).fetchone()
            if r:
                new_role = "admin" if (role == "admin" or r["role"] == "admin") else r["role"]
                c.execute("UPDATE users SET email=?, name=?, picture=?, role=?, last_login_ts=? WHERE id=?",
                          (email, name, picture, new_role, now, r["id"]))
                c.commit()
                return dict(c.execute("SELECT * FROM users WHERE id=?", (r["id"],)).fetchone())
            uid = "u_" + secrets.token_hex(8)
            c.execute("INSERT INTO users (id, google_sub, email, name, picture, role, created_ts, last_login_ts) "
                      "VALUES (?,?,?,?,?,?,?,?)", (uid, sub, email, name, picture, role, now, now))
            c.commit()
            return dict(c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone())
        finally:
            c.close()


def get_user_by_sub(sub: str) -> dict | None:
    c = _conn()
    try:
        r = c.execute("SELECT * FROM users WHERE google_sub=?", (sub,)).fetchone()
        return dict(r) if r else None
    finally:
        c.close()


def first_admin_id() -> str | None:
    c = _conn()
    try:
        r = c.execute("SELECT id FROM users WHERE role='admin' ORDER BY created_ts LIMIT 1").fetchone()
        return r["id"] if r else None
    finally:
        c.close()


# --- bots --------------------------------------------------------------------
def insert_bot(bot_id: str, user_id: str, token_hash: str, meeting_url: str) -> None:
    with _LOCK:
        c = _conn()
        try:
            c.execute("INSERT OR REPLACE INTO bots (bot_id, user_id, token_hash, meeting_url, created_ts, status) "
                      "VALUES (?,?,?,?,?, 'created')", (bot_id, user_id, token_hash, meeting_url, time.time()))
            c.commit()
        finally:
            c.close()


def resolve_bot_token(token_hash: str, max_age_sec: float = 24 * 3600) -> dict | None:
    c = _conn()
    try:
        r = c.execute("SELECT * FROM bots WHERE token_hash=?", (token_hash,)).fetchone()
        if not r or time.time() - (r["created_ts"] or 0) > max_age_sec:
            return None
        return dict(r)
    finally:
        c.close()


def mark_bot(bot_id: str, **cols) -> None:
    if not cols:
        return
    sets = ", ".join(f"{k}=?" for k in cols)
    with _LOCK:
        c = _conn()
        try:
            c.execute(f"UPDATE bots SET {sets} WHERE bot_id=?", (*cols.values(), bot_id))
            c.commit()
        finally:
            c.close()


def add_bot_seconds(bot_id: str, sec: float) -> None:
    with _LOCK:
        c = _conn()
        try:
            c.execute("UPDATE bots SET metered_sec = COALESCE(metered_sec,0)+? WHERE bot_id=?", (sec, bot_id))
            c.commit()
        finally:
            c.close()


def unended_bots(user_id: str, window_sec: float = 4 * 3600) -> list[dict]:
    """Bots that look live on our books (no ended_ts) — candidates for the
    Recall status reaper, which detects kicked/denied bots whose websocket
    never closed cleanly."""
    c = _conn()
    try:
        rows = c.execute(
            "SELECT * FROM bots WHERE user_id=? AND ended_ts IS NULL AND created_ts > ?",
            (user_id, time.time() - window_sec)).fetchall()
        return [dict(r) for r in rows]
    finally:
        c.close()


def find_active_bot(user_id: str, meeting_url: str) -> dict | None:
    """An existing live-looking bot for this user+meeting, to dedupe re-deploys:
    streaming (started, not ended, <4 h old) always counts; a just-launched bot
    that never connected counts only for 5 min, so a denied/failed bot doesn't
    block a legitimate retry for long."""
    now = time.time()
    c = _conn()
    try:
        r = c.execute(
            "SELECT * FROM bots WHERE user_id=? AND meeting_url=? AND ended_ts IS NULL "
            "AND ((started_ts IS NOT NULL AND created_ts > ?) OR created_ts > ?) "
            "ORDER BY created_ts DESC LIMIT 1",
            (user_id, meeting_url, now - 4 * 3600, now - 300)).fetchone()
        return dict(r) if r else None
    finally:
        c.close()


def count_active_bots(user_id: str, window_sec: float = 4 * 3600) -> int:
    """Bots launched recently and not ended — the concurrency guard."""
    c = _conn()
    try:
        r = c.execute("SELECT COUNT(*) n FROM bots WHERE user_id=? AND ended_ts IS NULL AND created_ts>?",
                      (user_id, time.time() - window_sec)).fetchone()
        return int(r["n"])
    finally:
        c.close()


# --- billing -----------------------------------------------------------------
def get_subscription(user_id: str) -> dict | None:
    c = _conn()
    try:
        r = c.execute("SELECT * FROM subscriptions WHERE user_id=?", (user_id,)).fetchone()
        return dict(r) if r else None
    finally:
        c.close()


def upsert_subscription(user_id: str, customer_id: str | None, subscription_id: str | None, status: str) -> None:
    with _LOCK:
        c = _conn()
        try:
            c.execute("INSERT INTO subscriptions (user_id, stripe_customer_id, stripe_subscription_id, status, updated_ts) "
                      "VALUES (?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET "
                      "stripe_customer_id=COALESCE(excluded.stripe_customer_id, stripe_customer_id), "
                      "stripe_subscription_id=COALESCE(excluded.stripe_subscription_id, stripe_subscription_id), "
                      "status=excluded.status, updated_ts=excluded.updated_ts",
                      (user_id, customer_id, subscription_id, status, time.time()))
            c.commit()
        finally:
            c.close()


def add_usage_minutes(user_id: str, month: str, minutes: float) -> float:
    """Add to the month bucket; returns the new total for that month."""
    with _LOCK:
        c = _conn()
        try:
            c.execute("INSERT INTO usage (user_id, month, minutes, updated_ts) VALUES (?,?,?,?) "
                      "ON CONFLICT(user_id, month) DO UPDATE SET minutes=minutes+excluded.minutes, "
                      "updated_ts=excluded.updated_ts", (user_id, month, minutes, time.time()))
            c.commit()
            r = c.execute("SELECT minutes FROM usage WHERE user_id=? AND month=?", (user_id, month)).fetchone()
            return float(r["minutes"])
        finally:
            c.close()


def get_usage_minutes(user_id: str, month: str) -> float:
    c = _conn()
    try:
        r = c.execute("SELECT minutes FROM usage WHERE user_id=? AND month=?", (user_id, month)).fetchone()
        return float(r["minutes"]) if r else 0.0
    finally:
        c.close()


def add_score(user_id: str, speaker: str, p_fake: float, rolling: float, verdict: str) -> None:
    """Append to the scored-window history (feeds forensic reports)."""
    with _LOCK:
        c = _conn()
        try:
            c.execute("INSERT INTO scores (user_id, speaker, ts, p_fake, rolling, verdict) "
                      "VALUES (?,?,?,?,?,?)",
                      (user_id, speaker, time.time(), p_fake, rolling, verdict))
            c.commit()
        finally:
            c.close()


def get_scores(user_id: str, speaker: str, t0: float, t1: float, limit: int = 2000) -> list[dict]:
    c = _conn()
    try:
        rows = c.execute("SELECT ts, p_fake, rolling, verdict FROM scores "
                         "WHERE user_id=? AND speaker=? AND ts BETWEEN ? AND ? "
                         "ORDER BY ts LIMIT ?", (user_id, speaker, t0, t1, limit)).fetchall()
        return [dict(r) for r in rows]
    finally:
        c.close()


def get_alert_webhook(user_id: str) -> str:
    u = get_user(user_id)
    return (u or {}).get("alert_webhook") or ""


def set_alert_webhook(user_id: str, url: str) -> None:
    with _LOCK:
        c = _conn()
        try:
            c.execute("UPDATE users SET alert_webhook=? WHERE id=?", (url, user_id))
            c.commit()
        finally:
            c.close()


def migrate_legacy(data_dir: Path, enroll_dir: Path, admin_uid: str) -> None:
    """One-time move of pre-tenancy flat data into the admin's workspace:
    flat capture WAVs and voiceprints into per-user subdirs, plus anything that
    landed in the interim 'admin' machine workspace before the first sign-in.
    Same-volume renames are atomic; a marker file prevents re-runs."""
    marker = data_dir.parent / ".tenancy_migrated" if data_dir.parent != data_dir else data_dir / ".tenancy_migrated"
    if marker.exists():
        return
    moved = 0
    try:
        if data_dir.exists():
            target = data_dir / admin_uid
            target.mkdir(parents=True, exist_ok=True)
            for f in list(data_dir.glob("*.wav")):                    # flat legacy files
                f.rename(target / f.name)
                moved += 1
            interim = data_dir / "admin"
            if admin_uid != "admin" and interim.exists():             # pre-signin machine dir
                for f in list(interim.glob("*.wav")):
                    f.rename(target / f.name)
                    moved += 1
        if enroll_dir.exists():
            etarget = enroll_dir / admin_uid
            etarget.mkdir(parents=True, exist_ok=True)
            for f in list(enroll_dir.glob("*.npy")):
                f.rename(etarget / f.name)
                moved += 1
            einterim = enroll_dir / "admin"
            if admin_uid != "admin" and einterim.exists():
                for f in list(einterim.glob("*.npy")):
                    f.rename(etarget / f.name)
                    moved += 1
        try:
            import incidents
            with _LOCK:
                c = incidents._conn()
                try:
                    c.execute("UPDATE incidents SET user_id=? WHERE user_id IS NULL", (admin_uid,))
                    c.commit()
                finally:
                    c.close()
        except Exception:  # noqa: BLE001 — incidents backfill is best-effort
            pass
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{admin_uid} {time.time()}\n")
    except Exception:  # noqa: BLE001 — never break startup over migration
        pass
    if moved:
        print(f"tenancy migration: moved {moved} legacy files into workspace {admin_uid}")


def webhook_seen(event_id: str, event_type: str) -> bool:
    """True if this Stripe event was already processed (and records it if not)."""
    with _LOCK:
        c = _conn()
        try:
            if c.execute("SELECT 1 FROM webhook_events WHERE id=?", (event_id,)).fetchone():
                return True
            c.execute("INSERT INTO webhook_events (id, type, created_ts) VALUES (?,?,?)",
                      (event_id, event_type, time.time()))
            c.commit()
            return False
        finally:
            c.close()
