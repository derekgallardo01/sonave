"""Track 1 security hardening: opt-in shared-token auth, meeting_url allowlist, input
hardening (VerdictReq), XSS-escape presence, Modal upload cap. All offline/mocked."""
import pytest
from fastapi.testclient import TestClient

import conftest

TOKEN = "s3cr3t-test-token"


# --- Railway (capture service) with auth ENABLED ---------------------------
@pytest.fixture
def rw_auth(monkeypatch):
    monkeypatch.setenv("SONAVE_API_TOKEN", TOKEN)
    monkeypatch.setenv("SONAVE_RECALL_API_KEY", "k")
    monkeypatch.setenv("SONAVE_SCORER_URL", "")
    m = conftest.load_module("rwapp_auth", "railway/app.py")
    m.QUALITY.clear(); m.VERDICTS.clear(); m.ROLL.clear()
    return m


def _c(m):
    return TestClient(m.app)


def test_gated_endpoints_401_without_token(rw_auth):
    c = _c(rw_auth)
    assert c.get("/captures").status_code == 401
    assert c.get("/api/quality").status_code == 401
    assert c.get("/download/x.wav").status_code == 401
    assert c.post("/api/verdict", json={"speaker": "D", "p_fake": 0.1,
                                        "rolling": 0.1, "verdict": "real"}).status_code == 401
    assert c.post("/bot", json={"meeting_url": "https://meet.google.com/abc"}).status_code == 401


def test_bearer_and_cookie_grant_access(rw_auth):
    assert _c(rw_auth).get("/api/quality", headers={"X-Sonave-Token": TOKEN}).status_code == 200
    c = _c(rw_auth); c.cookies.set("sonave_token", TOKEN)
    assert c.get("/api/quality").status_code == 200


def test_wrong_token_rejected(rw_auth):
    assert _c(rw_auth).get("/api/quality", headers={"X-Sonave-Token": "nope"}).status_code == 401


def test_ws_requires_token(rw_auth):
    with pytest.raises(Exception):                     # closed with 1008 before accept
        with _c(rw_auth).websocket_connect("/api/ws/audio") as ws:
            ws.receive_text()


def test_index_injects_auth_flag_and_escape_helper(rw_auth):
    t = _c(rw_auth).get("/").text
    assert "var AUTH='1'" in t          # login overlay activates
    assert "function esc(" in t          # XSS-escape helper present in the page


def test_index_auth_off_without_token(railway_mod):
    assert "var AUTH='0'" in _c(railway_mod).get("/").text


# --- meeting_url allowlist --------------------------------------------------
def test_bot_rejects_non_meeting_url(rw_auth):
    r = _c(rw_auth).post("/bot", headers={"X-Sonave-Token": TOKEN},
                         json={"meeting_url": "https://evil.example.com/x"})
    body = r.json()
    assert body.get("ok") is False and "Meet" in body.get("detail", "")


def test_bot_accepts_google_meet_host(rw_auth, monkeypatch):
    # allowlist passes -> it proceeds to the Recall call (which fails, but not on the URL check)
    r = _c(rw_auth).post("/bot", headers={"X-Sonave-Token": TOKEN},
                         json={"meeting_url": "https://meet.google.com/abc-defg-hij"})
    assert "must be a Google Meet" not in str(r.json())


# --- VerdictReq input hardening (clamps + sanitizes) ------------------------
def test_verdict_req_clamps_and_sanitizes(rw_auth):
    _c(rw_auth).post("/api/verdict", headers={"X-Sonave-Token": TOKEN},
                     json={"speaker": "<script>evil</script>", "p_fake": 5.0,
                           "rolling": -1.0, "verdict": "hax"})
    key = next(iter(rw_auth.VERDICTS))
    v = rw_auth.VERDICTS[key]
    assert "<" not in key and ">" not in key            # speaker whitelisted
    assert v["p_fake"] == 1.0 and v["rolling"] == 0.0    # probs clamped to [0,1]
    assert v["verdict"] == "suspect"                     # invalid enum -> suspect


# --- Modal detection service with auth + upload cap ------------------------
@pytest.fixture
def svc_auth(monkeypatch):
    import detector
    monkeypatch.setattr(detector, "load", lambda: (object(), "cpu"))
    monkeypatch.setattr(detector, "score_bytes",
                        lambda d: {"p_fake": 0.9, "verdict": "fake", "confidence": 0.8, "model_version": "t"})
    monkeypatch.setattr(detector, "score_clip",
                        lambda d: {"p_fake": 0.1, "p_max": 0.2, "verdict": "real", "n_windows": 3, "model_version": "t"})
    monkeypatch.setenv("SONAVE_API_TOKEN", TOKEN)
    monkeypatch.setenv("SONAVE_MAX_UPLOAD_MB", "1")
    app_mod = conftest.load_module("svcapp_auth", "service/app.py")
    with TestClient(app_mod.app) as c:
        yield c


def test_score_requires_token(svc_auth):
    assert svc_auth.post("/score", files={"file": ("c.wav", b"x", "audio/wav")}).status_code == 401
    assert svc_auth.get("/version").status_code == 401


def test_score_with_token_ok_healthz_open(svc_auth):
    r = svc_auth.post("/score", headers={"X-Sonave-Token": TOKEN},
                      files={"file": ("c.wav", b"RIFFxxxxWAVE", "audio/wav")})
    assert r.status_code == 200
    assert svc_auth.get("/healthz").status_code == 200   # health probe stays open


def test_oversized_upload_rejected(svc_auth):
    big = b"x" * 2_000_000                                # 2 MB > 1 MB cap
    r = svc_auth.post("/score", headers={"X-Sonave-Token": TOKEN},
                      files={"file": ("c.wav", big, "audio/wav")})
    assert r.status_code == 413
