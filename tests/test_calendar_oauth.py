"""OAuth Calendar auto-join (flag-gated behind SONAVE_CALENDAR_OAUTH until the
sensitive scope passes Google verification)."""
import io
import json
import time
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

import conftest

TOKEN = "machine-token-123"


@pytest.fixture
def mod(monkeypatch, tmp_path):
    monkeypatch.setenv("SONAVE_SCORER_URL", "")
    monkeypatch.setenv("SONAVE_RECALL_API_KEY", "test-key")
    monkeypatch.setenv("SONAVE_API_TOKEN", TOKEN)
    monkeypatch.setenv("SONAVE_SESSION_SECRET", "sec")
    monkeypatch.setenv("SONAVE_GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("SONAVE_GOOGLE_CLIENT_SECRET", "cs")
    monkeypatch.setenv("SONAVE_PUBLIC_DOMAIN", "test.sonave.dev")
    monkeypatch.setenv("SONAVE_CALENDAR_OAUTH", "1")
    m = conftest.load_module("rwapp_cal", "railway/app.py")
    m._CAL_AT.clear()
    return m


def _user(m, sub="s-c1", email="c1@x.com"):
    return m.db.upsert_google_user(sub, email, "Cal User", "", "member")


def _client_as(m, uid):
    c = TestClient(m.app, base_url="https://testserver")
    c.cookies.set("sonave_session", m.auth.sign_session(uid))
    return c


def _connect(m, c):
    r = c.get("/auth/calendar/connect", follow_redirects=False)
    assert r.status_code == 302
    loc = r.headers["location"]
    q = parse_qs(urlparse(loc).query)
    assert q["scope"] == [m.auth.CALENDAR_SCOPE]
    assert q["access_type"] == ["offline"] and q["prompt"] == ["consent"]
    m.auth._exchange_code = lambda code: {"access_token": "at", "refresh_token": "rt-1"}
    return c.get(f"/auth/callback?code=abc&state={q['state'][0]}", follow_redirects=False)


def test_connect_stores_refresh_token_and_tracks(mod):
    u = _user(mod)
    c = _client_as(mod, u["id"])
    r = _connect(mod, c)
    assert r.status_code == 302 and r.headers["location"] == "/console"
    row = mod.db.get_oauth_token(u["id"], "google_calendar")
    assert row and row["refresh_token"] == "rt-1"
    assert mod.db.list_events(user_id=u["id"], kind="calendar_connected")
    me = c.get("/api/me").json()
    assert me["calendar_oauth"] is True and me["calendar_connected"] is True


def test_connect_404_when_flag_off(mod, monkeypatch):
    monkeypatch.setenv("SONAVE_CALENDAR_OAUTH", "0")
    u = _user(mod, "s-c2", "c2@x.com")
    assert _client_as(mod, u["id"]).get("/auth/calendar/connect",
                                        follow_redirects=False).status_code == 404
    assert _client_as(mod, u["id"]).get("/api/me").json()["calendar_oauth"] is False


def test_connect_requires_signed_in_user(mod):
    r = TestClient(mod.app, base_url="https://testserver").get(
        "/auth/calendar/connect", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/console"


def test_disconnect_revokes_and_deletes(mod, monkeypatch):
    u = _user(mod, "s-c3", "c3@x.com")
    c = _client_as(mod, u["id"])
    _connect(mod, c)
    revoked = {}
    monkeypatch.setattr(mod.auth, "revoke_google_token", lambda t: revoked.setdefault("t", t))
    assert c.post("/auth/calendar/disconnect").json()["ok"] is True
    assert revoked["t"] == "rt-1"
    assert mod.db.get_oauth_token(u["id"], "google_calendar") is None
    assert mod.db.list_events(user_id=u["id"], kind="calendar_disconnected")


CAL_ITEMS = {"items": [
    {"id": "ev1", "status": "confirmed",
     "hangoutLink": "https://meet.google.com/cal-test-one",
     "start": {"dateTime": "2026-08-14T15:00:30+00:00"}},
    {"id": "ev2", "status": "cancelled",
     "hangoutLink": "https://meet.google.com/cal-test-two",
     "start": {"dateTime": "2026-08-14T15:00:30+00:00"}},
    {"id": "ev3", "status": "confirmed",
     "start": {"date": "2026-08-14"}},                      # all-day, no dateTime
]}
NOW = 1786719600.0   # 2026-08-14 15:00:00 UTC


def test_google_calendar_events_parsing(mod, monkeypatch):
    class _R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    monkeypatch.setattr(mod.autojoin.urllib.request, "urlopen",
                        lambda req, timeout=None: _R(json.dumps(CAL_ITEMS).encode()))
    ev = mod.autojoin.google_calendar_events("at", now=NOW)
    assert len(ev) == 1
    assert ev[0]["meet_url"] == "https://meet.google.com/cal-test-one"
    assert ev[0]["start_ts"] == NOW + 30


def test_tick_launches_from_calendar_grant_once(mod, monkeypatch):
    u = _user(mod, "s-c4", "c4@x.com")
    mod.db.save_oauth_token(u["id"], "google_calendar", "scope", "rt-4")
    monkeypatch.setattr(mod.auth, "refresh_access_token", lambda rt: "at-4")
    monkeypatch.setattr(mod.autojoin, "google_calendar_events",
                        lambda at, now=None, horizon_sec=1800: [
                            {"uid": "ev-t", "start_ts": NOW + 30,
                             "meet_url": "https://meet.google.com/cal-tick"}])
    calls = {"n": 0}

    class _R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    def _open(req, timeout=None):
        calls["n"] += 1
        return _R(json.dumps({"id": f"cal-bot-{calls['n']}"}).encode())

    monkeypatch.setattr(mod.urllib.request, "urlopen", _open)
    assert mod._autojoin_tick(now=NOW) == 1
    assert mod._autojoin_tick(now=NOW) == 0                  # occurrence logged
    assert calls["n"] == 1
    d = json.loads(mod.db.list_events(user_id=u["id"], kind="bot_created")[0]["detail"])
    assert d["source"] == "autojoin"


def test_tick_drops_revoked_grant(mod, monkeypatch):
    u = _user(mod, "s-c5", "c5@x.com")
    mod.db.save_oauth_token(u["id"], "google_calendar", "scope", "rt-5")

    def _boom(rt):
        raise ValueError("invalid_grant")

    monkeypatch.setattr(mod.auth, "refresh_access_token", _boom)
    mod._autojoin_tick(now=NOW)
    assert mod.db.get_oauth_token(u["id"], "google_calendar") is None
    assert mod.db.list_events(user_id=u["id"], kind="calendar_disconnected")
