"""Auth rate limiter (ASVS 1.1.1) and the HttpOnly operator sign-in flow."""
import pytest
from fastapi.testclient import TestClient

import conftest

TOKEN = "machine-token-123"


@pytest.fixture
def mod(monkeypatch, tmp_path):
    monkeypatch.setenv("SONAVE_SCORER_URL", "")
    monkeypatch.setenv("SONAVE_API_TOKEN", TOKEN)
    monkeypatch.setenv("SONAVE_SESSION_SECRET", "sec")
    monkeypatch.setenv("SONAVE_GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("SONAVE_GOOGLE_CLIENT_SECRET", "cs")
    m = conftest.load_module("rwapp_ratelimit", "railway/app.py")
    m._AUTH_RATE.clear()          # fresh window for every test
    return m


def test_auth_endpoints_throttle_at_20_per_minute(mod):
    c = TestClient(mod.app, base_url="https://testserver")
    codes = [c.post("/auth/google-credential", json={"credential": "x"}).status_code
             for _ in range(25)]
    assert all(s in (400, 403, 422) for s in codes[:20]), codes[:20]   # through, rejected
    assert all(s == 429 for s in codes[20:]), codes[20:]               # then throttled
    r = c.post("/auth/google-credential", json={"credential": "x"})
    assert r.status_code == 429 and r.headers.get("retry-after") == "60"
    mod._AUTH_RATE.clear()


def test_operator_login_sets_httponly_cookie(mod):
    c = TestClient(mod.app, base_url="https://testserver")
    assert c.post("/auth/operator", json={"token": "wrong"}).status_code == 403
    r = c.post("/auth/operator", json={"token": TOKEN})
    assert r.status_code == 200 and r.json()["ok"] is True
    sc = r.headers.get("set-cookie", "")
    assert "sonave_token=" in sc and "HttpOnly" in sc and "Secure" in sc
    assert "SameSite=strict" in sc.lower().replace("samesite=strict", "SameSite=strict")
    # the cookie authenticates as the operator
    assert c.get("/api/me").json()["kind"] == "machine"
    mod._AUTH_RATE.clear()


def test_query_param_token_no_longer_authenticates(mod):
    """ASVS 2.1.1: tokens must not work from URL query strings on HTTP routes."""
    c = TestClient(mod.app, base_url="https://testserver")
    assert c.get(f"/api/me?token={TOKEN}").status_code == 401
    assert c.get("/api/me", headers={"X-Sonave-Token": TOKEN}).status_code == 200
