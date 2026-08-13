"""Multi-tenancy (Stage B): per-user workspaces, per-bot WS tokens, isolation,
legacy migration. Two users must never see each other's data."""
import base64
import hashlib
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import conftest

TOKEN = "machine-token-123"


class _Inline:
    def __init__(self, target=None, args=(), daemon=None):
        self._t, self._a = target, args

    def start(self):
        self._t(*self._a)


@pytest.fixture
def mod(monkeypatch, tmp_path):
    monkeypatch.setenv("SONAVE_SCORER_URL", "")
    monkeypatch.setenv("SONAVE_RECALL_API_KEY", "test-key")
    monkeypatch.setenv("SONAVE_API_TOKEN", TOKEN)               # secured mode
    monkeypatch.setenv("SONAVE_SESSION_SECRET", "sec")
    m = conftest.load_module("rwapp_tenancy", "railway/app.py")
    monkeypatch.setattr(m, "DATA_DIR", tmp_path / "captured")
    monkeypatch.setattr(m.enroll, "ENROLL_DIR", tmp_path / "enrollments")
    m.QUALITY.clear()
    m.VERDICTS.clear()
    m.ROLL.clear()
    return m


def _mk_user(m, sub, email, role="member"):
    return m.db.upsert_google_user(sub, email, email.split("@")[0], "", role)


def _client_as(m, uid):
    c = TestClient(m.app, base_url="https://testserver")
    c.cookies.set("sonave_session", m.auth.sign_session(uid))
    return c


def _mint_bot(m, bot_id, uid, tok):
    m.db.insert_bot(bot_id, uid, hashlib.sha256(tok.encode()).hexdigest(), "https://meet.google.com/x")


def _stream(m, tok, speaker, seconds):
    frame = base64.b64encode(b"\x40\x00" * m.SR).decode()
    url = "/api/ws/audio" + (f"?token={tok}" if tok else "")
    with TestClient(m.app).websocket_connect(url) as ws:
        for _ in range(seconds):
            ws.send_text(json.dumps({"data": {"data": {"buffer": frame,
                                                        "participant": {"name": speaker}}}}))


# --- WS bot-token binding -----------------------------------------------------
def test_bot_token_routes_stream_to_owner_workspace(mod, tmp_path):
    ua = _mk_user(mod, "s-a", "a@x.com")
    _mint_bot(mod, "bot-a", ua["id"], "tok-a")
    _stream(mod, "tok-a", "Alice", 3)
    assert (ua["id"], "Alice") in mod.QUALITY
    assert not any(k[0] != ua["id"] for k in mod.QUALITY)


def test_ws_rejects_unknown_token_in_secured_mode(mod):
    with pytest.raises(Exception):
        with TestClient(mod.app).websocket_connect("/api/ws/audio?token=bogus") as ws:
            ws.receive_text()


def test_ws_rejects_expired_bot_token(mod, monkeypatch):
    ua = _mk_user(mod, "s-a2", "a2@x.com")
    _mint_bot(mod, "bot-old", ua["id"], "tok-old")
    monkeypatch.setattr(mod.db, "resolve_bot_token", lambda h, max_age_sec=0: None)
    with pytest.raises(Exception):
        with TestClient(mod.app).websocket_connect("/api/ws/audio?token=tok-old") as ws:
            ws.receive_text()


def test_machine_token_streams_to_admin_workspace(mod):
    admin = _mk_user(mod, "s-adm", "adm@x.com", role="admin")
    _stream(mod, TOKEN, "Derek", 3)
    assert (admin["id"], "Derek") in mod.QUALITY


def test_ws_meters_bot_seconds(mod, monkeypatch):
    ua = _mk_user(mod, "s-met", "met@x.com")
    _mint_bot(mod, "bot-met", ua["id"], "tok-met")
    t = {"now": 1000.0}
    monkeypatch.setattr(mod.time, "time", lambda: t.pop("now", None) or _tick(t))

    def _tick(state):
        state.setdefault("n", 0)
        state["n"] += 61                                    # every call advances 61 s
        return 1000.0 + state["n"]

    _stream(mod, "tok-met", "Alice", 3)
    c = mod.db._conn()
    row = c.execute("SELECT metered_sec, status FROM bots WHERE bot_id='bot-met'").fetchone()
    c.close()
    assert row["metered_sec"] > 0 and row["status"] == "ended"


# --- API isolation ------------------------------------------------------------
def test_captures_quality_incidents_are_isolated(mod, tmp_path, monkeypatch):
    ua = _mk_user(mod, "s-u1", "u1@x.com")
    ub = _mk_user(mod, "s-u2", "u2@x.com")
    (mod.DATA_DIR / ua["id"]).mkdir(parents=True)
    (mod.DATA_DIR / ua["id"] / "meet_Alice_1_000.wav").write_bytes(b"RIFF0000WAVE")
    mod.VERDICTS[(ua["id"], "Alice")] = {"p_fake": 0.9, "rolling": 0.9, "verdict": "fake"}

    import incidents
    monkeypatch.setattr(incidents, "DB_PATH", tmp_path / "i.db")
    incidents.record("Alice", 0.9, "m", user_id=ua["id"])

    ca, cb = _client_as(mod, ua["id"]), _client_as(mod, ub["id"])
    assert ca.get("/captures").json()["files"]
    assert cb.get("/captures").json()["files"] == []
    assert "Alice" in ca.get("/api/quality").json()
    assert "Alice" not in cb.get("/api/quality").json()
    assert ca.get("/api/incidents").json()["incidents"]
    assert cb.get("/api/incidents").json()["incidents"] == []
    # B cannot ack A's incident; A can
    iid = ca.get("/api/incidents").json()["incidents"][0]["id"]
    assert cb.post("/api/incidents/ack", json={"id": iid}).json()["ok"] is False
    assert ca.post("/api/incidents/ack", json={"id": iid}).json()["ok"] is True


def test_download_scoped_to_workspace(mod):
    ua = _mk_user(mod, "s-d1", "d1@x.com")
    ub = _mk_user(mod, "s-d2", "d2@x.com")
    (mod.DATA_DIR / ua["id"]).mkdir(parents=True)
    (mod.DATA_DIR / ua["id"] / "meet_A_1_000.wav").write_bytes(b"RIFFxxxxWAVE")
    assert _client_as(mod, ua["id"]).get("/download/meet_A_1_000.wav").status_code == 200
    r = _client_as(mod, ub["id"]).get("/download/meet_A_1_000.wav")
    assert r.json() == {"error": "not found"}


def test_bot_launch_mints_scoped_token_and_redacts_it(mod, monkeypatch):
    ua = _mk_user(mod, "s-b1", "b1@x.com")
    sent = {}

    class _R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    def _open(req, timeout=None):
        sent["body"] = json.loads(req.data)
        return _R(json.dumps({"id": "recall-bot-1"}).encode())

    monkeypatch.setattr(mod.urllib.request, "urlopen", _open)
    r = _client_as(mod, ua["id"]).post("/bot", json={"meeting_url": "https://meet.google.com/abc"})
    j = r.json()
    assert j["ok"] is True and "ws" not in j                      # token redacted
    ws_url = sent["body"]["recording_config"]["realtime_endpoints"][0]["url"]
    tok = ws_url.split("token=")[1]
    assert tok != TOKEN                                           # per-bot, not the god token
    row = mod.db.resolve_bot_token(hashlib.sha256(tok.encode()).hexdigest())
    assert row and row["user_id"] == ua["id"] and row["bot_id"] == "recall-bot-1"


def test_bot_deploy_dedupes_same_meeting(mod, monkeypatch):
    """Pressing Protect/Deploy twice must not launch a second Recall bot."""
    ua = _mk_user(mod, "s-dd1", "dd1@x.com")
    calls = {"n": 0}

    class _R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    def _open(req, timeout=None):
        calls["n"] += 1
        return _R(json.dumps({"id": f"recall-bot-{calls['n']}"}).encode())

    monkeypatch.setattr(mod.urllib.request, "urlopen", _open)
    c = _client_as(mod, ua["id"])
    r1 = c.post("/bot", json={"meeting_url": "https://meet.google.com/dup"})
    r2 = c.post("/bot", json={"meeting_url": "https://meet.google.com/dup"})
    assert r1.json()["ok"] and "already" not in r1.json()
    assert r2.json() == {"ok": True, "bot_id": "recall-bot-1", "already": True,
                         "detail": "A Sonave bot is already in this meeting."}
    assert calls["n"] == 1                                        # one Recall dispatch only
    # a DIFFERENT meeting still deploys fresh
    r3 = c.post("/bot", json={"meeting_url": "https://meet.google.com/other"})
    assert r3.json()["ok"] and not r3.json().get("already") and calls["n"] == 2
    # once the first bot's stream has ended, the same meeting can be re-protected
    mod.db.mark_bot("recall-bot-1", ended_ts=1.0, status="ended")
    r4 = c.post("/bot", json={"meeting_url": "https://meet.google.com/dup"})
    assert r4.json()["ok"] and not r4.json().get("already") and calls["n"] == 3


def test_quality_reports_idle_seconds(mod):
    """idle_sec drives the muted/silent state in the console and Meet panel."""
    ua = _mk_user(mod, "s-idle", "idle@x.com")
    mod._quality(ua["id"], "Derek", b"\x40\x00" * mod.SR)
    q = _client_as(mod, ua["id"]).get("/api/quality").json()
    assert q["Derek"]["idle_sec"] in (0, 1)                       # just spoke
    mod.QUALITY[(ua["id"], "Derek")]["last_audio_ts"] -= 100      # went quiet/muted
    q = _client_as(mod, ua["id"]).get("/api/quality").json()
    assert q["Derek"]["idle_sec"] >= 99


# --- legacy migration ---------------------------------------------------------
def test_migrate_legacy_moves_flat_data_once(mod, tmp_path):
    admin = _mk_user(mod, "s-adm2", "boss@x.com", role="admin")
    mod.DATA_DIR.mkdir(parents=True, exist_ok=True)
    (mod.DATA_DIR / "meet_Old_1_000.wav").write_bytes(b"RIFF0000WAVE")
    (mod.DATA_DIR / "admin").mkdir()
    (mod.DATA_DIR / "admin" / "meet_Interim_2_000.wav").write_bytes(b"RIFF0000WAVE")
    mod.enroll.ENROLL_DIR.mkdir(parents=True, exist_ok=True)
    (mod.enroll.ENROLL_DIR / "Old.npy").write_bytes(b"\x00" * 8)

    mod.db.migrate_legacy(mod.DATA_DIR, mod.enroll.ENROLL_DIR, admin["id"])
    ws = mod.DATA_DIR / admin["id"]
    assert (ws / "meet_Old_1_000.wav").exists()
    assert (ws / "meet_Interim_2_000.wav").exists()
    assert (mod.enroll.ENROLL_DIR / admin["id"] / "Old.npy").exists()
    assert not list(mod.DATA_DIR.glob("*.wav"))                   # flat dir emptied

    # idempotent: second run is a no-op (marker file)
    (mod.DATA_DIR / "meet_New_3_000.wav").write_bytes(b"RIFF0000WAVE")
    mod.db.migrate_legacy(mod.DATA_DIR, mod.enroll.ENROLL_DIR, admin["id"])
    assert (mod.DATA_DIR / "meet_New_3_000.wav").exists()         # untouched after marker
