"""Tests for rich metadata and data acquisition attribution across Google Meet add-on and web console."""
import io
import json
import time
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

import conftest

TOKEN = "machine-token-123"

CLAIMS_USER = {
    "aud": "cid",
    "iss": "accounts.google.com",
    "email_verified": "true",
    "sub": "gsub-acq-1",
    "email": "meet_founder@acme.corp",
    "name": "Alex Mercer",
    "picture": "https://lh3.googleusercontent.com/photo.jpg"
}


@pytest.fixture
def mod(monkeypatch, tmp_path):
    monkeypatch.setenv("SONAVE_SCORER_URL", "")
    monkeypatch.setenv("SONAVE_RECALL_API_KEY", "test-key")
    monkeypatch.setenv("SONAVE_API_TOKEN", TOKEN)
    monkeypatch.setenv("SONAVE_APP_DB", str(tmp_path / "app.db"))
    monkeypatch.setenv("SONAVE_SESSION_SECRET", "test-secret")
    monkeypatch.setenv("SONAVE_GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("SONAVE_GOOGLE_CLIENT_SECRET", "cs")
    monkeypatch.setenv("SONAVE_ADMIN_EMAILS", "boss@sonave.dev")
    monkeypatch.setenv("SONAVE_PUBLIC_DOMAIN", "localhost:8000")
    m = conftest.load_module("rwapp_acq", "railway/app.py")
    m.QUALITY.clear()
    m.VERDICTS.clear()
    m.ROLL.clear()
    return m


def _client(m):
    return TestClient(m.app, base_url="https://testserver")


def _claims(**over):
    d = dict(CLAIMS_USER, exp=str(int(time.time()) + 3600))
    d.update(over)
    return d


def test_credential_login_captures_rich_metadata_and_notifies_admin(mod, monkeypatch):
    """Meet Add-on credential login sends timezone, locale, meet space, embedded flag, and UTMs."""
    monkeypatch.setattr(mod.auth, "_fetch_tokeninfo", lambda cred: _claims())
    notifications = []
    monkeypatch.setattr(mod, "_notify_admin", lambda text: notifications.append(text))

    c = _client(mod)
    meta = {
        "timezone": "America/Los_Angeles",
        "locale": "en-US",
        "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "platform": "MacIntel",
        "screen": "1920x1080 (vp: 400x600)",
        "is_embedded": True,
        "meet_space": "xyz-abcd-efg",
        "referrer": "https://meet.google.com/xyz-abcd-efg",
        "utm_source": "google_workspace_marketplace",
        "utm_medium": "in_call_drawer",
        "utm_campaign": "meet_add_on_launch",
        "channel": "Google Meet Add-on Drawer"
    }

    r = c.post("/auth/google-credential", json={
        "credential": "test_jwt_credential",
        "metadata": meta
    })
    assert r.status_code == 200
    res = r.json()
    assert res["ok"] and res["token"]
    assert res["email"] == "meet_founder@acme.corp"

    # Verify user profile in db has rich metadata
    u = mod.db.get_user_by_sub("gsub-acq-1")
    assert u is not None
    user_meta = json.loads(u["metadata"])
    assert user_meta["timezone"] == "America/Los_Angeles"
    assert user_meta["locale"] == "en-US"
    assert user_meta["meet_space"] == "xyz-abcd-efg"
    assert user_meta["utm_source"] == "google_workspace_marketplace"
    assert user_meta["is_embedded"] is True

    # Verify _notify_admin format has timezone, locale, space code, channel, and email
    assert len(notifications) == 1
    note = notifications[0]
    assert "👤 New Signup: meet_founder@acme.corp" in note
    assert "Timezone: America/Los_Angeles (en-US)" in note
    assert "Channel: Google Meet Add-on Drawer" in note
    assert "Space: xyz-abcd-efg" in note
    assert "UTM: google_workspace_marketplace/in_call_drawer (meet_add_on_launch)" in note

    # Verify signup event detail contains metadata
    events = mod.db.list_events(user_id=u["id"], kind="signup")
    assert len(events) == 1
    ev_det = json.loads(events[0]["detail"])
    assert ev_det["meet_space"] == "xyz-abcd-efg"
    assert ev_det["timezone"] == "America/Los_Angeles"


def test_admin_api_exposes_acquisition_metadata(mod, monkeypatch):
    """Admin users list and user telemetry endpoints expose acquisition data."""
    # Create admin
    admin = mod.db.upsert_google_user("gsub-boss", "boss@sonave.dev", "Boss", "", "admin")
    admin_tok = mod.auth.sign_session(admin["id"])
    c = _client(mod)
    admin_hdr = {"Authorization": f"Bearer {admin_tok}"}

    # Create user with metadata
    meta = {
        "timezone": "Europe/London",
        "locale": "en-GB",
        "channel": "Google Meet Add-on",
        "meet_space": "gb-team-sync",
        "utm_source": "newsletter",
        "platform": "Linux x86_64"
    }
    user = mod.db.upsert_google_user("gsub-uk", "uk_user@example.co.uk", "UK User", "", "member", metadata=meta)

    # 1. Admin Rollup
    users_resp = c.get("/api/admin/users", headers=admin_hdr)
    assert users_resp.status_code == 200
    rows = {u["email"]: u for u in users_resp.json()["users"]}
    uk_row = rows["uk_user@example.co.uk"]
    assert uk_row["timezone"] == "Europe/London"
    assert uk_row["locale"] == "en-GB"
    assert uk_row["channel"] == "Google Meet Add-on"
    assert uk_row["initial_space"] == "gb-team-sync"
    assert uk_row["utm_source"] == "newsletter"

    # 2. Admin User Telemetry 360 view
    telem_resp = c.get(f"/api/admin/users/{user['id']}/telemetry", headers=admin_hdr)
    assert telem_resp.status_code == 200
    telem_data = telem_resp.json()
    u_telem = telem_data["user"]
    assert u_telem["timezone"] == "Europe/London"
    assert u_telem["locale"] == "en-GB"
    assert u_telem["channel"] == "Google Meet Add-on"
    assert u_telem["initial_space"] == "gb-team-sync"
    assert u_telem["utm_source"] == "newsletter"
    assert "gb-team-sync" in u_telem["recent_spaces"]


def test_telemetry_event_updates_user_profile_and_tracks_milestones(mod, monkeypatch):
    """Telemetry endpoint records milestones and updates user metadata."""
    user = mod.db.upsert_google_user("gsub-tm", "telemetry_user@sonave.com", "Telem", "", "member")
    tok = mod.auth.sign_session(user["id"])
    c = _client(mod)
    auth_hdr = {"Authorization": f"Bearer {tok}"}

    alerts = []
    monkeypatch.setattr(mod, "_notify_admin", lambda text: alerts.append(text))

    # Send addon_opened event with rich metadata
    ev_addon = {
        "kind": "addon_opened",
        "detail": {
            "timezone": "Asia/Tokyo",
            "locale": "ja-JP",
            "is_embedded": True,
            "meet_space": "tokyo-boardroom",
            "screen": "2560x1440"
        }
    }
    r1 = c.post("/api/telemetry/event", json=ev_addon, headers=auth_hdr)
    assert r1.status_code == 204

    # Verify user profile in db got updated
    updated_u = mod.db.get_user(user["id"])
    umeta = json.loads(updated_u["metadata"])
    assert umeta["timezone"] == "Asia/Tokyo"
    assert umeta["locale"] == "ja-JP"
    assert umeta["meet_space"] == "tokyo-boardroom"

    # Verify alert was pushed
    assert any("Meet Add-on Opened" in a and "tokyo-boardroom" in a for a in alerts)

    # Send session_milestone event
    ev_milestone = {
        "kind": "session_milestone",
        "detail": {
            "minutes": 5,
            "space": "tokyo-boardroom"
        }
    }
    r2 = c.post("/api/telemetry/event", json=ev_milestone, headers=auth_hdr)
    assert r2.status_code == 204
    assert any("Active Call Milestone" in a and "5 min" in a for a in alerts)

    # Send diagnostic_run event
    ev_diag = {
        "kind": "diagnostic_run",
        "detail": {
            "test_name": "Urgent CEO Wire Probe",
            "voice_id": "ceo_impersonator"
        }
    }
    r3 = c.post("/api/telemetry/event", json=ev_diag, headers=auth_hdr)
    assert r3.status_code == 204
    assert any("Diagnostic Test Run" in a and "Urgent CEO Wire Probe" in a for a in alerts)


    # Send meet_connected event
    ev_connect = {
        "kind": "meet_connected",
        "detail": {
            "space": "tokyo-boardroom",
            "timezone": "Asia/Tokyo"
        }
    }
    r4 = c.post("/api/telemetry/event", json=ev_connect, headers=auth_hdr)
    assert r4.status_code == 204
    assert any("Audio Stream Attached" in a and "tokyo-boardroom" in a for a in alerts)

    # Verify telemetry aggregation in db
    u_telem = mod.db.get_user_telemetry(user["id"])
    assert u_telem["meets_total"] >= 1
    assert u_telem["sims_total"] >= 1
    assert "tokyo-boardroom" in u_telem["recent_spaces"]
    assert "Urgent CEO Wire Probe" in u_telem["recent_simulations"]


def test_web_login_and_callback_attaches_acquisition_cookie(mod, monkeypatch):
    """Web login captures UTM and referrer in cookie, and callback passes to user and alert."""
    class _R(io.BytesIO):
        pass

    sent = {}
    def _open(req, timeout=None):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data)
        return _R(b"ok")

    monkeypatch.setenv("SONAVE_ADMIN_WEBHOOK", "https://hooks.slack.example/test")
    monkeypatch.setattr(mod.threading, "Thread", lambda target=None, args=(), daemon=None: type("T", (), {"start": lambda self: target(*args)})())
    monkeypatch.setattr(mod.urllib.request, "urlopen", _open)

    c = _client(mod)
    # 1. User visits /auth/login with UTM and timezone query params
    r_login = c.get("/auth/login?utm_source=twitter&utm_medium=social&utm_campaign=deepfake_launch&ref=https://t.co/abc&tz=America/Chicago&loc=en-US", follow_redirects=False)
    assert r_login.status_code == 302
    assert "sonave_acq" in r_login.headers.get("set-cookie", "")

    # 2. Google OAuth callback completes
    mod.auth._exchange_code = lambda code: {"access_token": "at"}
    mod.auth._fetch_userinfo = lambda at: {
        "sub": "gsub-web-acq", "email": "twitter_fan@example.com", "email_verified": True, "name": "Twitter Fan", "picture": ""
    }
    state = parse_qs(urlparse(r_login.headers["location"]).query)["state"][0]
    r_cb = c.get(f"/auth/callback?code=mock_code&state={state}", follow_redirects=False)
    assert r_cb.status_code == 302

    # Verify user in db has acquisition data stored
    u = mod.db.get_user_by_sub("gsub-web-acq")
    assert u is not None
    user_meta = json.loads(u["metadata"])
    assert user_meta["utm_source"] == "twitter"
    assert user_meta["utm_medium"] == "social"
    assert user_meta["timezone"] == "America/Chicago"

    # Verify notification delivered to webhook
    assert "twitter_fan@example.com" in sent["body"]["text"]
    assert "Timezone: America/Chicago" in sent["body"]["text"]
    assert "UTM: twitter/social" in sent["body"]["text"]


def test_html_contains_client_acquisition_hooks(mod):
    """Verify HTML files contain required client-side metadata and attribution logic."""
    c = _client(mod)

    # 1. meet-addon.html
    r_addon = c.get("/meet-addon")
    assert r_addon.status_code == 200
    html_addon = r_addon.text
    assert "getMeetClientMeta" in html_addon
    assert "Intl.DateTimeFormat().resolvedOptions().timeZone" in html_addon
    assert "session_milestone" in html_addon
    assert "diagnostic_run" in html_addon
    assert "meet_connected" in html_addon
    assert "/auth/google-credential" in html_addon
    assert "/api/telemetry/event" in html_addon

    # 2. console.html
    r_console = c.get("/console")
    assert r_console.status_code == 200
    html_console = r_console.text
    assert "initAcquisitionTracking" in html_console
    assert "sonave_utm_source" in html_console
    assert "ACQUISITION ATTRIBUTION &amp; DEVICE METADATA" in html_console
    assert "utTzLocale" in html_console
    assert "utChannel" in html_console
    assert "utInitSpace" in html_console

    # 3. landing.html
    r_landing = c.get("/")
    assert r_landing.status_code == 200
    html_landing = r_landing.text
    assert "preserveOutboundLinks" in html_landing
    assert "sonave_utm_source" in html_landing
