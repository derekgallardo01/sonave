"""test_email_notifications.py — tests for user transactional emails and telemetry."""
import io
import json
from urllib.parse import parse_qs, urlparse
from fastapi.testclient import TestClient
import railway.db as db


USERINFO = {
    "sub": "gsub-email-test",
    "email": "notify-test@usesonave.com",
    "name": "Jane Doe",
    "picture": "",
    "email_verified": True,
}


def test_send_welcome_email_on_new_signup(railway_mod, monkeypatch):
    emails_sent = []
    def _mock_send_welcome(to_email, name=""):
        emails_sent.append({"to": to_email, "name": name})

    monkeypatch.setenv("SONAVE_GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("SONAVE_GOOGLE_CLIENT_SECRET", "cs")
    monkeypatch.setenv("SONAVE_SESSION_SECRET", "sec")
    monkeypatch.setattr(railway_mod, "_send_welcome_email", _mock_send_welcome)
    monkeypatch.setattr(railway_mod.auth, "_exchange_code", lambda code: {"access_token": "at"})
    monkeypatch.setattr(railway_mod.auth, "_fetch_userinfo", lambda at: dict(USERINFO, email="jane.doe@example.com"))

    c = TestClient(railway_mod.app, base_url="https://testserver")
    r = c.get("/auth/login", follow_redirects=False)
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    r2 = c.get(f"/auth/callback?code=abc&state={state}", follow_redirects=False)
    assert r2.status_code == 302

    # Verify welcome email was called
    assert len(emails_sent) == 1
    assert emails_sent[0]["to"] == "jane.doe@example.com"
    assert emails_sent[0]["name"] == "Jane Doe"

    # Verify event telemetry recorded
    events = db.list_events(kind="email_sent", limit=10)
    welcome_evs = [e for e in events if json.loads(e["detail"]).get("type") == "welcome"]
    assert len(welcome_evs) >= 1
    d = json.loads(welcome_evs[0]["detail"])
    assert d["to"] == "jane.doe@example.com"


def test_quota_warning_email_on_threshold(railway_mod, monkeypatch):
    emails_sent = []
    def _mock_quota_email(to_email, name, pct, used, free_m):
        emails_sent.append({"to": to_email, "pct": pct, "used": used, "free": free_m})

    monkeypatch.setattr(railway_mod, "_send_quota_warning_email", _mock_quota_email)

    # Create user with an email and get the generated user ID
    user = db.upsert_google_user("gsub-quota-1", "quota-user@example.com", "Quota Tester", "", "member")
    uid = user["id"]

    # Simulate reaching 80% threshold (240 min)
    railway_mod._meter_tick("bot_xyz", uid, 240 * 60)

    # Check if quota email was recorded
    assert any(e["pct"] == 80 and e["to"] == "quota-user@example.com" for e in emails_sent)

    events = db.list_events(kind="email_sent", limit=10)
    quota_evs = [e for e in events if json.loads(e["detail"]).get("type") == "quota_80"]
    assert len(quota_evs) >= 1
    d = json.loads(quota_evs[0]["detail"])
    assert d["to"] == "quota-user@example.com"


def test_incident_alert_email_on_sustained_deepfake(railway_mod, tmp_path, monkeypatch):
    import incidents
    monkeypatch.setattr(incidents, "DB_PATH", tmp_path / "inc_test.db")

    emails_sent = []
    def _mock_incident_email(to_email, spk, roll, model, hold):
        emails_sent.append({"to": to_email, "speaker": spk, "rolling": roll, "model": model, "hold": hold})

    monkeypatch.setattr(railway_mod, "_send_incident_alert_email", _mock_incident_email)

    user = db.upsert_google_user("gsub-inc-1", "incident-user@example.com", "Alice Smith", "", "member")
    uid = user["id"]

    railway_mod.SCORER_URL = "http://scorer.test"
    class _R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): self.close()

    monkeypatch.setattr(railway_mod.urllib.request, "urlopen",
                        lambda req, timeout=None: _R(json.dumps({"p_fake": 1.0, "model_version": "sonave-xlsr-meet-v2"}).encode()))

    spk = "Suspicious_Caller"
    railway_mod.ROLL[(uid, spk)] = 0.95
    railway_mod.FAKE_STREAK[(uid, spk)] = 0

    for _ in range(railway_mod.INCIDENT_STREAK):
        railway_mod._score_and_store(uid, spk, b"wavbytes")

    # Verify incident email was triggered
    assert len(emails_sent) >= 1
    assert emails_sent[0]["to"] == "incident-user@example.com"
    assert emails_sent[0]["speaker"] == spk
    assert emails_sent[0]["hold"] is True

    # Verify email_sent event
    events = db.list_events(kind="email_sent", limit=10)
    inc_evs = [e for e in events if json.loads(e["detail"]).get("type") == "incident"]
    assert len(inc_evs) >= 1
    d = json.loads(inc_evs[0]["detail"])
    assert d["to"] == "incident-user@example.com"
    assert d["speaker"] == spk


def test_email_template_rendering(railway_mod, monkeypatch):
    delivered = []
    def _capture_send_email(to_email, subject, text, html=None):
        delivered.append({"to": to_email, "subject": subject, "text": text, "html": html})

    monkeypatch.setattr(railway_mod, "_send_email", _capture_send_email)

    # Test Welcome email
    railway_mod._send_welcome_email("test@example.com", "Bob Jones")
    assert len(delivered) == 1
    assert "Welcome to Sonave" in delivered[0]["subject"]
    assert "Bob" in delivered[0]["text"]
    assert "meet.google.com" in delivered[0]["html"]

    # Test 80% Quota email
    railway_mod._send_quota_warning_email("test@example.com", "Bob Jones", 80, 240.0, 300.0)
    assert len(delivered) == 2
    assert "80%" in delivered[1]["subject"]
    assert "240" in delivered[1]["text"]

    # Test 100% Quota email
    railway_mod._send_quota_warning_email("test@example.com", "Bob Jones", 100, 300.0, 300.0)
    assert len(delivered) == 3
    assert "exhausted" in delivered[2]["subject"]
    assert "100%" in delivered[2]["text"]

    # Test Incident email
    railway_mod._send_incident_alert_email("test@example.com", "CEO Fake", 0.98, "sonave-xlsr-meet-v2", True)
    assert len(delivered) == 4
    assert "URGENT" in delivered[3]["subject"]
    assert "WIRE HOLD ACTIVE" in delivered[3]["html"]
    assert "CEO Fake" in delivered[3]["text"]
