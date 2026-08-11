"""Speaker enrollment / voiceprint verification end-to-end.

Covers:
  - Modal service accepts voiceprint_b64 and returns fused risk
  - Railway enrollment endpoints (enroll from captures, list, delete)
  - Railway live scoring includes voiceprint in Modal requests
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import conftest


def _fake_urlopen(payload):
    class _Resp(io.BytesIO):
        def __enter__(self):
            return self
        def __exit__(self, *a):
            self.close()
    def _open(req, timeout=None):
        return _Resp(json.dumps(payload).encode())
    return _open


# --- Modal service: voiceprint fusion ----------------------------------------
@pytest.fixture
def svc_vp(monkeypatch):
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
    class _FakeModelSLS:
        MAX_LEN = 64000
        SR = 16000
    monkeypatch.setattr(detector, "model_sls", _FakeModelSLS())

    # Mock enroll so we don't download ECAPA in tests
    app_mod = conftest.load_module("svcapp_vp", "service/app.py")

    def _fake_fused_risk_with_voiceprint(p_fake, speaker_id, source, voiceprint):
        return {
            "p_fake": p_fake,
            "risk": 0.85,
            "verdict": "fake",
            "speaker_check": {"similarity": 0.15, "match": False},
            "match_conf": 0.0,
            "mismatch_risk": 0.85,
        }

    monkeypatch.setattr(app_mod.enroll, "fused_risk_with_voiceprint", _fake_fused_risk_with_voiceprint)
    # _decode_audio needs librosa/soundfile — mock it for the fast test suite
    monkeypatch.setattr(app_mod, "_decode_audio", lambda data: np.zeros(16000, dtype=np.float32))
    with TestClient(app_mod.app) as c:
        yield c


def test_score_clip_with_voiceprint_fuses(svc_vp):
    vp = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    vp = vp / np.linalg.norm(vp)
    vp_b64 = base64.b64encode(vp.tobytes()).decode()
    r = svc_vp.post("/score_clip",
                    data={"speaker_id": "Derek", "voiceprint_b64": vp_b64},
                    files={"file": ("c.wav", b"RIFFxxxxWAVE", "audio/wav")})
    assert r.status_code == 200
    j = r.json()
    assert j["verdict"] == "fake"
    assert j["risk"] == pytest.approx(0.85)
    assert j["speaker_check"]["match"] is False


def test_score_without_voiceprint_unchanged(svc_vp):
    r = svc_vp.post("/score_clip",
                    files={"file": ("c.wav", b"RIFFxxxxWAVE", "audio/wav")})
    assert r.status_code == 200
    j = r.json()
    assert j["verdict"] == "real"  # no fusion -> original detector verdict


def test_score_json_with_voiceprint(svc_vp):
    vp = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    vp = vp / np.linalg.norm(vp)
    body = {
        "audio_b64": base64.b64encode(b"RIFFxxxxWAVE").decode(),
        "speaker_id": "Derek",
        "voiceprint_b64": base64.b64encode(vp.tobytes()).decode(),
    }
    j = svc_vp.post("/score_json", json=body).json()
    assert j["verdict"] == "fake"
    assert "risk" in j


# --- Railway enrollment endpoints --------------------------------------------
@pytest.fixture
def rw_enroll(monkeypatch, tmp_path):
    """Railway module with mocked enroll + temp data dirs."""
    monkeypatch.setenv("SONAVE_SCORER_URL", "")
    monkeypatch.setenv("SONAVE_RECALL_API_KEY", "test-key")
    monkeypatch.setenv("SONAVE_ENROLL_DIR", str(tmp_path / "enrollments"))
    mod = conftest.load_module("rwapp_e", "railway/app.py")
    mod.QUALITY.clear()
    mod.VERDICTS.clear()
    mod.ROLL.clear()
    # Mock enroll functions to avoid ECAPA load
    monkeypatch.setattr(mod.enroll, "is_enrolled", lambda sid: (mod.enroll.ENROLL_DIR / f"{sid}.npy").exists())
    monkeypatch.setattr(mod.enroll, "list_enrolled", lambda: [p.stem for p in mod.enroll.ENROLL_DIR.glob("*.npy")] if mod.enroll.ENROLL_DIR.exists() else [])
    monkeypatch.setattr(mod.enroll, "enroll", lambda sid, paths: _fake_enroll(mod, sid, paths))
    return mod


def _fake_enroll(mod, speaker_id, paths):
    mod.enroll.ENROLL_DIR.mkdir(parents=True, exist_ok=True)
    vp = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    np.save(mod.enroll.ENROLL_DIR / f"{speaker_id}.npy", vp)
    return vp


def test_enroll_from_captures(rw_enroll, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(rw_enroll, "DATA_DIR", tmp_path)
    (tmp_path / "meet_Derek_1_000.wav").write_bytes(b"RIFF0000WAVE")
    (tmp_path / "meet_Derek_1_001.wav").write_bytes(b"RIFF0000WAVE")
    c = TestClient(rw_enroll.app)
    r = c.post("/api/enroll", json={"speaker_id": "Derek"})
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    assert j["clips"] == 2
    # should now be listed
    enrolled = c.get("/api/enrolled").json()
    assert any(e["speaker_id"] == "Derek" for e in enrolled["enrolled"])


def test_enroll_missing_captures(rw_enroll, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(rw_enroll, "DATA_DIR", tmp_path)
    c = TestClient(rw_enroll.app)
    r = c.post("/api/enroll", json={"speaker_id": "Nobody"})
    assert r.status_code == 200
    assert r.json()["ok"] is False


def test_delete_enrollment(rw_enroll, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(rw_enroll, "DATA_DIR", tmp_path)
    rw_enroll.enroll.ENROLL_DIR.mkdir(parents=True, exist_ok=True)
    np.save(rw_enroll.enroll.ENROLL_DIR / "Derek.npy", np.array([1.0, 2.0]))
    c = TestClient(rw_enroll.app)
    r = c.delete("/api/enroll/Derek")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert c.get("/api/enrolled").json()["enrolled"] == []


def test_quality_shows_enrollment_status(rw_enroll, monkeypatch):
    from fastapi.testclient import TestClient
    rw_enroll.enroll.ENROLL_DIR.mkdir(parents=True, exist_ok=True)
    np.save(rw_enroll.enroll.ENROLL_DIR / "Derek.npy", np.array([1.0, 2.0]))
    rw_enroll.VERDICTS["Derek"] = {"p_fake": 0.1, "rolling": 0.1, "verdict": "real"}
    c = TestClient(rw_enroll.app)
    q = c.get("/api/quality").json()
    assert q["Derek"]["enrolled"] is True


# --- Railway live scoring with voiceprint ------------------------------------
def test_score_and_store_sends_voiceprint_when_enrolled(rw_enroll, monkeypatch, tmp_path):
    rw_enroll.SCORER_URL = "http://scorer.test"
    rw_enroll.enroll.ENROLL_DIR.mkdir(parents=True, exist_ok=True)
    vp = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    np.save(rw_enroll.enroll.ENROLL_DIR / "Derek.npy", vp)

    captured = {"body": None}

    def _capture_req(req, timeout=None):
        captured["body"] = req.data
        class _Resp(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): self.close()
        return _Resp(json.dumps({"p_fake": 0.2, "risk": 0.15, "verdict": "real",
                                 "speaker_check": {"similarity": 0.55, "match": True}}).encode())

    monkeypatch.setattr(rw_enroll.urllib.request, "urlopen", _capture_req)
    rw_enroll._score_and_store("Derek", b"RIFFxxxxWAVE")

    # Verify the request body contained the voiceprint
    body = captured["body"]
    assert body is not None
    assert b"voiceprint_b64" in body
    # Verify verdict stored uses risk when speaker_check present
    assert rw_enroll.VERDICTS["Derek"]["verdict"] == "real"
    assert rw_enroll.VERDICTS["Derek"]["speaker_check"]["match"] is True
