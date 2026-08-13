"""Track 3 fraud workflow: incident store (SQLite), dedup, acknowledge, Slack alert,
and the end-to-end 'sustained fake -> open incident' hook + the HTTP endpoints. Offline."""
import io
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def inc(tmp_path, monkeypatch):
    import incidents
    monkeypatch.setattr(incidents, "DB_PATH", tmp_path / "incidents.db")
    monkeypatch.setattr(incidents, "ALERT_WEBHOOK", "")
    monkeypatch.setattr(incidents, "WIRE_HOLD", True)
    return incidents


def test_record_opens_once_then_dedups(inc):
    a = inc.record("Derek", 0.8, "m")
    assert a and a["speaker"] == "Derek" and a["hold"] is True
    assert inc.record("Derek", 0.85, "m") is None       # already open -> no new incident
    rows = inc.list_incidents()
    assert len(rows) == 1 and rows[0]["status"] == "open" and rows[0]["hold"] == 1


def test_acknowledge_clears_hold_and_allows_reopen(inc):
    a = inc.record("Derek", 0.8, "m")
    assert inc.acknowledge(a["id"]) is True
    row = inc.list_incidents()[0]
    assert row["status"] == "acknowledged" and row["hold"] == 0
    assert inc.record("Derek", 0.9, "m") is not None     # a new sustained fake reopens
    assert len(inc.list_incidents()) == 2


def test_notify_noop_without_webhook(inc):
    inc.notify({"speaker": "D", "rolling": 0.8, "model": "m", "hold": True})   # no raise


def test_notify_posts_slack_text(inc, monkeypatch):
    monkeypatch.setattr(inc, "ALERT_WEBHOOK", "http://hook.test")
    sent = {}

    class _R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): self.close()

    def _open(req, timeout=None):
        sent["body"] = req.data
        return _R(b"ok")

    monkeypatch.setattr(inc.urllib.request, "urlopen", _open)
    inc.notify({"speaker": "Derek", "rolling": 0.82, "model": "m", "hold": True})
    payload = json.loads(sent["body"])
    assert "deepfake" in payload["text"].lower() and "Derek" in payload["text"]


# --- endpoints + the scoring hook ------------------------------------------
def test_incident_endpoints(railway_mod, tmp_path, monkeypatch):
    import incidents
    monkeypatch.setattr(incidents, "DB_PATH", tmp_path / "i.db")
    incidents.record("Derek", 0.8, "m")
    c = TestClient(railway_mod.app)
    out = c.get("/api/incidents").json()
    assert out["incidents"][0]["speaker"] == "Derek"
    iid = out["incidents"][0]["id"]
    assert c.post("/api/incidents/ack", json={"id": iid}).json()["ok"] is True
    assert c.get("/api/incidents").json()["incidents"][0]["status"] == "acknowledged"


def test_sustained_fake_opens_incident_and_alerts(railway_mod, tmp_path, monkeypatch):
    """One fake window must NOT hold a wire; INCIDENT_STREAK consecutive ones must."""
    import incidents
    monkeypatch.setattr(incidents, "DB_PATH", tmp_path / "i.db")
    fired = {"n": 0}
    monkeypatch.setattr(incidents, "notify",
                        lambda ev, webhook=None: fired.__setitem__("n", fired["n"] + 1))
    railway_mod.SCORER_URL = "http://scorer.test"

    class _R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): self.close()

    monkeypatch.setattr(railway_mod.urllib.request, "urlopen",
                        lambda req, timeout=None: _R(json.dumps({"p_fake": 1.0, "model_version": "m"}).encode()))
    railway_mod.ROLL[("admin", "Derek")] = 0.9                      # already high -> EMA lands in 'fake'
    for i in range(railway_mod.INCIDENT_STREAK - 1):
        railway_mod._score_and_store("admin", "Derek", b"wavbytes")
        assert incidents.list_incidents() == []          # not sustained yet -> no incident
    railway_mod._score_and_store("admin", "Derek", b"wavbytes")   # streak reached
    assert incidents.list_incidents()[0]["speaker"] == "Derek"
    assert fired["n"] == 1                               # alerted exactly once


def test_real_window_resets_fake_streak(railway_mod, tmp_path, monkeypatch):
    import incidents
    monkeypatch.setattr(incidents, "DB_PATH", tmp_path / "i3.db")
    railway_mod.SCORER_URL = "http://scorer.test"

    class _R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): self.close()

    payloads = iter([{"p_fake": 1.0}, {"p_fake": 1.0}, {"p_fake": 0.0}, {"p_fake": 1.0}])
    monkeypatch.setattr(railway_mod.urllib.request, "urlopen",
                        lambda req, timeout=None: _R(json.dumps(next(payloads) | {"model_version": "m"}).encode()))
    railway_mod.ROLL[("admin", "Derek")] = 0.95
    railway_mod.FAKE_STREAK.clear()
    for _ in range(4):
        railway_mod._score_and_store("admin", "Derek", b"wavbytes")
    # streak was broken by the clean window -> never reached INCIDENT_STREAK
    assert incidents.list_incidents() == []


def test_score_real_opens_no_incident(railway_mod, tmp_path, monkeypatch):
    import incidents
    monkeypatch.setattr(incidents, "DB_PATH", tmp_path / "i2.db")
    railway_mod.SCORER_URL = "http://scorer.test"

    class _R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): self.close()

    monkeypatch.setattr(railway_mod.urllib.request, "urlopen",
                        lambda req, timeout=None: _R(json.dumps({"p_fake": 0.02, "model_version": "m"}).encode()))
    railway_mod.ROLL.pop(("admin", "Derek"), None)
    railway_mod._score_and_store("admin", "Derek", b"wavbytes")
    assert incidents.list_incidents() == []
