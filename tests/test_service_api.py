"""The detection microservice HTTP contract (service/app.py). The detector is mocked,
so no model loads — this pins the API shape the orchestrator + Modal deploy depend on."""
import pytest
from fastapi.testclient import TestClient

import conftest


@pytest.fixture
def svc(monkeypatch):
    import detector
    monkeypatch.setattr(detector, "load", lambda: (object(), "cpu"))
    monkeypatch.setattr(detector, "score_bytes",
                        lambda data: {"p_fake": 0.91, "verdict": "fake",
                                      "confidence": 0.8, "model_version": "test"})
    monkeypatch.setattr(detector, "score_clip",
                        lambda data: {"p_fake": 0.05, "p_max": 0.4, "verdict": "real",
                                      "n_windows": 12, "model_version": "test"})
    monkeypatch.setattr(detector, "score_array",
                        lambda arr: {"p_fake": 0.12, "verdict": "real",
                                     "confidence": 0.5, "model_version": "test"})
    # model_sls.MAX_LEN needed by /ready probe
    class _FakeModelSLS:
        MAX_LEN = 64000
    monkeypatch.setattr(detector, "model_sls", _FakeModelSLS())
    app_mod = conftest.load_module("svcapp", "service/app.py")
    with TestClient(app_mod.app) as c:
        yield c


def test_healthz(svc):
    r = svc.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["device"] == "cpu"


def test_ready_probe(svc):
    r = svc.get("/ready")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ready" and j["device"] == "cpu"


def test_version_exposes_thresholds(svc):
    j = svc.get("/version").json()
    assert "tau_real" in j and "tau_fake" in j and "model_version" in j


def test_request_id_header(svc):
    r = svc.get("/healthz")
    assert "x-request-id" in r.headers
    r2 = svc.get("/healthz", headers={"x-request-id": "abc123"})
    assert r2.headers["x-request-id"] == "abc123"


def test_score_multipart(svc):
    r = svc.post("/score", files={"file": ("c.wav", b"RIFFxxxxWAVE", "audio/wav")})
    assert r.status_code == 200
    j = r.json()
    assert j["verdict"] == "fake" and "latency_ms" in j


def test_score_clip_windowed(svc):
    r = svc.post("/score_clip", files={"file": ("c.wav", b"RIFFxxxxWAVE", "audio/wav")})
    assert r.status_code == 200
    j = r.json()
    assert j["verdict"] == "real" and j["n_windows"] == 12 and "latency_ms" in j


def test_score_json_base64(svc):
    import base64
    body = {"audio_b64": base64.b64encode(b"RIFFxxxxWAVE").decode(), "speaker_id": "Derek"}
    j = svc.post("/score_json", json=body).json()
    assert j["verdict"] == "fake" and j["speaker_id"] == "Derek"


def test_rate_limit_hits_after_burst(svc):
    ok = 0
    blocked = 0
    for _ in range(15):
        r = svc.post("/score", files={"file": ("c.wav", b"RIFFxxxxWAVE", "audio/wav")})
        if r.status_code == 200:
            ok += 1
        elif r.status_code == 429:
            blocked += 1
    assert blocked >= 1, "expected at least one 429 after bursting"
    assert ok >= 1, "expected at least one success"
