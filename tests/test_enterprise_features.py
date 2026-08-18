"""Tests for Enterprise Features:
  - Forensic Audit Report Generator (HTML/PDF & HMAC Signatures)
  - Multi-Platform Rich Webhook Dispatcher (Slack, Discord, MS Teams, SIEM)
  - Native Google Meet Media API Ingest Engine
  - Enterprise API Endpoints (/report/{id}, /api/incidents/{id}/export.json, /api/webhook/test, /api/meet/sessions)
"""
import io
import json
import time
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_app(railway_mod, tmp_path, monkeypatch):
    import incidents
    monkeypatch.setattr(incidents, "DB_PATH", tmp_path / "test_enterprise_incidents.db")
    monkeypatch.setattr(incidents, "ALERT_WEBHOOK", "")
    return TestClient(railway_mod.app)


def test_forensics_signature_and_report_generation():
    import forensics
    inc = {
        "id": 101,
        "speaker": "Executive_Impersonator",
        "first_ts": 1723900000.0,
        "last_ts": 1723900060.0,
        "rolling": 0.945,
        "model": "sonave-xlsr-meet-v2",
        "status": "open",
        "hold": 1,
        "user_id": "u_test123"
    }
    secret = b"test-secret-key-123"
    digest, sig = forensics.compute_forensic_signature(inc, secret)
    assert len(digest) == 64 and len(sig) == 64
    
    html = forensics.generate_report_html(inc, domain="usesonave.com", secret=secret)
    assert "Forensic Incident Report" in html
    assert "#101" in html
    assert "Executive_Impersonator" in html
    assert "94.5% FAKE" in html
    assert "sonave-xlsr-meet-v2" in html
    assert "WIRE HELD" in html
    assert digest in html
    assert sig in html


def test_webhook_dispatcher_platform_detection_and_formatting():
    import webhook_dispatcher
    assert webhook_dispatcher.detect_platform("https://hooks.slack.com/services/T00/B00/X00") == "slack"
    assert webhook_dispatcher.detect_platform("https://discord.com/api/webhooks/123/abc") == "discord"
    assert webhook_dispatcher.detect_platform("https://company.webhook.office.com/webhookb2/123") == "teams"
    assert webhook_dispatcher.detect_platform("https://api.mycompany.com/webhook") == "generic"

    event = {
        "id": 42,
        "speaker": "Unknown Caller",
        "rolling": 0.88,
        "model": "sonave-xlsr-meet-v2",
        "hold": True
    }
    
    slack = webhook_dispatcher.format_slack_payload(event, domain="usesonave.com")
    assert "blocks" in slack and "🚨" in slack["text"]
    assert "Unknown Caller" in json.dumps(slack)
    
    discord = webhook_dispatcher.format_discord_payload(event, domain="usesonave.com")
    assert "embeds" in discord and discord["embeds"][0]["color"] == 0xFF4D5E
    
    teams = webhook_dispatcher.format_teams_payload(event, domain="usesonave.com")
    assert teams["@type"] == "MessageCard"
    
    generic = webhook_dispatcher.format_generic_payload(event, domain="usesonave.com")
    assert generic["event_type"] == "voice.authenticity.alert" and generic["verdict"] == "FAKE"


def test_webhook_dispatcher_send_test_alert(monkeypatch):
    import webhook_dispatcher
    sent = {}

    class _MockResp(io.BytesIO):
        def __init__(self):
            super().__init__(b"ok")
            self.status = 200
        def __enter__(self): return self
        def __exit__(self, *a): self.close()

    def _mock_urlopen(req, timeout=None):
        sent["url"] = req.full_url
        sent["headers"] = dict(req.headers)
        sent["data"] = json.loads(req.data.decode("utf-8"))
        return _MockResp()

    monkeypatch.setattr(webhook_dispatcher.urllib.request, "urlopen", _mock_urlopen)
    ok, code, msg = webhook_dispatcher.send_test_alert("https://hooks.slack.com/test", domain="usesonave.com")
    assert ok is True and code == 200 and msg == "ok"
    assert sent["url"] == "https://hooks.slack.com/test"
    assert "blocks" in sent["data"]


def test_meet_media_ingest_session_lifecycle():
    import meet_media_ingest
    received_frames = []
    
    def _on_frame(spk, chunk, ts):
        received_frames.append((spk, chunk, ts))

    sess = meet_media_ingest.MeetMediaSession("space_abc123", "test_oauth_token", on_audio_frame=_on_frame)
    assert sess.space_id == "space_abc123"
    assert sess.state == "idle"

    # Simulate chunk ingestion
    sess.state = "streaming"
    fake_pcm = b"\x00\x01" * 160  # 10ms of 16kHz audio
    sess.ingest_speaker_chunk("Speaker_1", fake_pcm)
    assert sess.bytes_received == len(fake_pcm)
    assert len(received_frames) == 1
    assert received_frames[0][0] == "Speaker_1"
    
    st = sess.status()
    assert st["state"] == "streaming"
    assert st["speakers_count"] == 1
    assert "Speaker_1" in st["speakers"]

    sess.close()
    assert sess.state == "closed"


def test_api_report_and_export_endpoints(client_app):
    import incidents
    inc = incidents.record("Fraudster_Target", 0.92, "sonave-xlsr-meet-v2")
    assert inc is not None
    inc_id = inc["id"]

    # Test HTML Forensic Report
    r_html = client_app.get(f"/api/incidents/{inc_id}/report")
    assert r_html.status_code == 200
    assert "text/html" in r_html.headers["content-type"]
    assert "Forensic Incident Report" in r_html.text
    assert "Fraudster_Target" in r_html.text

    # Test Report Alias
    r_alias = client_app.get(f"/report/{inc_id}")
    assert r_alias.status_code == 200
    assert "Fraudster_Target" in r_alias.text

    # Test JSON Evidence Export
    r_json = client_app.get(f"/api/incidents/{inc_id}/export.json")
    assert r_json.status_code == 200
    j = r_json.json()
    assert j["incident"]["speaker"] == "Fraudster_Target"
    assert "integrity" in j
    assert len(j["integrity"]["digest_sha256"]) == 64
    assert len(j["integrity"]["signature_hmac_sha256"]) == 64

    # Non-existent incident
    assert client_app.get("/api/incidents/99999/report").status_code == 404
    assert client_app.get("/api/incidents/99999/export.json").status_code == 404


def test_api_webhook_test_and_meet_sessions(client_app, monkeypatch):
    import webhook_dispatcher
    monkeypatch.setattr(webhook_dispatcher, "send_test_alert",
                        lambda url, domain="usesonave.com": (True, 200, "ok"))

    r = client_app.post("/api/webhook/test", json={"url": "https://hooks.slack.com/services/T/B/X"})
    assert r.status_code == 200
    assert r.json()["ok"] is True

    # Check Meet Sessions status
    r_sess = client_app.get("/api/meet/sessions")
    assert r_sess.status_code == 200
    assert "active_sessions" in r_sess.json()
