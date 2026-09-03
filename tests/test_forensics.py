"""Sprint 1 product completeness: scored-window history, forensic reports,
per-workspace alert webhooks."""
import io
import json

import pytest
from fastapi.testclient import TestClient

import conftest

TOKEN = "machine-token-123"


@pytest.fixture
def mod(monkeypatch, tmp_path):
    monkeypatch.setenv("SONAVE_SCORER_URL", "")
    monkeypatch.setenv("SONAVE_RECALL_API_KEY", "test-key")
    monkeypatch.setenv("SONAVE_API_TOKEN", TOKEN)
    m = conftest.load_module("rwapp_forensics", "railway/app.py")
    monkeypatch.setattr(m, "DATA_DIR", tmp_path / "captured")
    m.QUALITY.clear()
    m.VERDICTS.clear()
    m.ROLL.clear()
    return m


def _user(m, sub="s-f", email="f@x.com", role="member"):
    return m.db.upsert_google_user(sub, email, "F", "", role)


def _client_as(m, uid):
    c = TestClient(m.app, base_url="https://testserver")
    c.cookies.set("sonave_session", m.auth.sign_session(uid))
    return c


# --- score history ------------------------------------------------------------
def test_score_history_written_by_live_scoring(mod, monkeypatch, tmp_path):
    import incidents
    monkeypatch.setattr(incidents, "DB_PATH", tmp_path / "i.db")
    mod.SCORER_URL = "http://scorer.test"

    class _R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): self.close()

    monkeypatch.setattr(mod.urllib.request, "urlopen",
                        lambda req, timeout=None: _R(json.dumps({"p_fake": 0.9, "model_version": "m"}).encode()))
    mod._score_and_store("admin", "Derek", b"wav")
    rows = mod.db.get_scores("admin", "Derek", 0, 9e12)
    assert len(rows) == 1 and rows[0]["p_fake"] == 0.9 and rows[0]["verdict"] in ("fake", "suspect", "real")


# --- forensic report ----------------------------------------------------------
def test_report_ownership_and_content(mod, monkeypatch, tmp_path):
    import incidents
    monkeypatch.setattr(incidents, "DB_PATH", tmp_path / "i.db")
    ua = _user(mod, "s-r1", "r1@x.com")
    ub = _user(mod, "s-r2", "r2@x.com")
    adm = _user(mod, "s-radm", "radm@x.com", role="admin")
    inc = incidents.record("Alice", 0.91, "sonave-test", user_id=ua["id"])
    for p in (0.2, 0.5, 0.85, 0.95):
        mod.db.add_score(ua["id"], "Alice", p, p, "fake" if p >= 0.7 else "real")

    r = _client_as(mod, ua["id"]).get(f"/report/{inc['id']}")
    assert r.status_code == 200
    assert "Forensic Incident Report" in r.text and "Alice" in r.text and "WIRE HELD" in r.text
    assert "0.950" in r.text                          # score rows present
    assert _client_as(mod, ub["id"]).get(f"/report/{inc['id']}").status_code == 404
    assert _client_as(mod, adm["id"]).get(f"/report/{inc['id']}").status_code == 200
    assert TestClient(mod.app).get(f"/report/{inc['id']}").status_code == 401


# --- per-workspace alerts -----------------------------------------------------
def test_settings_webhook_validation_and_roundtrip(mod):
    u = _user(mod, "s-w1", "w1@x.com")
    c = _client_as(mod, u["id"])
    assert c.post("/api/settings", json={"alert_webhook": "http://insecure"}).status_code == 422
    assert c.post("/api/settings", json={"alert_webhook": "https://hooks.slack.com/services/T/X"}).json()["ok"]
    assert mod.db.get_alert_webhook(u["id"]) == "https://hooks.slack.com/services/T/X"
    assert c.get("/api/me").json()["alert_webhook"].startswith("https://hooks.slack.com")
    assert c.post("/api/settings", json={"alert_webhook": ""}).json()["ok"]   # clear
    assert mod.db.get_alert_webhook(u["id"]) == ""


def test_notify_uses_workspace_webhook_over_env(mod, monkeypatch):
    import netsafe
    monkeypatch.setattr(netsafe, 'assert_public_https', lambda u: None)
    import incidents
    sent = {}

    class _R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): self.close()

    def _open(req, timeout=None):
        sent["url"] = req.full_url
        return _R(b"ok")

    monkeypatch.setattr(incidents.urllib.request, "urlopen", _open)
    monkeypatch.setattr(incidents, "ALERT_WEBHOOK", "https://env-fallback.example")
    ev = {"speaker": "D", "rolling": 0.9, "model": "m", "hold": True}
    incidents.notify(ev, webhook="https://workspace.example/hook")
    assert sent["url"] == "https://workspace.example/hook"
    incidents.notify(ev)                                  # no override -> env fallback
    assert sent["url"] == "https://env-fallback.example"


def test_incident_alert_fires_to_workspace_webhook(mod, monkeypatch, tmp_path):
    import netsafe
    monkeypatch.setattr(netsafe, 'assert_public_https', lambda u: None)
    import incidents
    monkeypatch.setattr(incidents, "DB_PATH", tmp_path / "i2.db")
    ua = _user(mod, "s-wh", "wh@x.com")
    mod.db.set_alert_webhook(ua["id"], "https://hooks.example/mine")
    urls = []

    class _R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): self.close()

    def _open(req, timeout=None):
        urls.append(req.full_url)
        return _R(json.dumps({"p_fake": 1.0, "model_version": "m"}).encode())

    monkeypatch.setattr(mod.urllib.request, "urlopen", _open)
    monkeypatch.setattr(incidents.urllib.request, "urlopen", _open)
    mod.SCORER_URL = "http://scorer.test"
    mod.ROLL[(ua["id"], "Bad")] = 0.95
    for _ in range(mod.INCIDENT_STREAK):
        mod._score_and_store(ua["id"], "Bad", b"wav")
    assert "https://hooks.example/mine" in urls           # alert went to the workspace hook
