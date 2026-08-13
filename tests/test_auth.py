"""Google OAuth + session + principal behavior (Stage A of multi-user)."""
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

import conftest

TOKEN = "machine-token-123"

USERINFO = {"sub": "gsub-1", "email": "derek@example.com", "email_verified": True,
            "name": "Derek", "picture": "http://p/x.png"}


@pytest.fixture
def mod(monkeypatch, tmp_path):
    monkeypatch.setenv("SONAVE_SCORER_URL", "")
    monkeypatch.setenv("SONAVE_RECALL_API_KEY", "test-key")
    monkeypatch.setenv("SONAVE_API_TOKEN", TOKEN)
    monkeypatch.setenv("SONAVE_APP_DB", str(tmp_path / "app.db"))
    monkeypatch.setenv("SONAVE_SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("SONAVE_GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("SONAVE_GOOGLE_CLIENT_SECRET", "cs")
    monkeypatch.setenv("SONAVE_ADMIN_EMAILS", "derek@example.com")
    monkeypatch.setenv("SONAVE_PUBLIC_DOMAIN", "localhost:8000")
    m = conftest.load_module("rwapp_oauth", "railway/app.py")
    m.QUALITY.clear()
    m.VERDICTS.clear()
    m.ROLL.clear()
    return m


def client(m):
    # https base so Secure cookies round-trip in the test client
    return TestClient(m.app, base_url="https://testserver")


def _google_login(m, c, userinfo=USERINFO):
    m.auth._exchange_code = lambda code: {"access_token": "at"}
    m.auth._fetch_userinfo = lambda at: dict(userinfo)
    r = c.get("/auth/login", follow_redirects=False)
    assert r.status_code == 302
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    return c.get(f"/auth/callback?code=abc&state={state}", follow_redirects=False)


# --- session token mechanics -------------------------------------------------
def test_session_sign_verify_and_tamper(mod, monkeypatch):
    u = mod.db.upsert_google_user("s1", "a@b.c", "A", "", "member")
    tok = mod.auth.sign_session(u["id"], u["session_ver"])
    assert mod.auth.verify_session(tok) == u["id"]
    assert mod.auth.verify_session(tok[:-3] + "xxx") is None       # bad signature
    assert mod.auth.verify_session("garbage") is None
    assert mod.auth.verify_session(None) is None


def test_session_expiry_and_revocation(mod, monkeypatch):
    u = mod.db.upsert_google_user("s2", "b@b.c", "B", "", "member")
    monkeypatch.setattr(mod.auth, "SESSION_TTL", -10)              # already expired
    tok = mod.auth.sign_session(u["id"])
    assert mod.auth.verify_session(tok) is None
    monkeypatch.setattr(mod.auth, "SESSION_TTL", 3600)
    tok = mod.auth.sign_session(u["id"], session_ver=99)           # wrong version
    assert mod.auth.verify_session(tok) is None


# --- OAuth flow ---------------------------------------------------------------
def test_login_redirects_to_google_with_state(mod):
    r = client(mod).get("/auth/login", follow_redirects=False)
    assert r.status_code == 302
    loc = r.headers["location"]
    assert loc.startswith("https://accounts.google.com/")
    q = parse_qs(urlparse(loc).query)
    assert q["client_id"] == ["cid"] and q["scope"] == ["openid email profile"]
    assert "sonave_oauth_state" in r.headers.get("set-cookie", "")


def test_callback_creates_admin_and_session(mod):
    c = client(mod)
    r = _google_login(mod, c)
    assert r.status_code == 302 and r.headers["location"] == "/console"
    me = c.get("/api/me").json()
    assert me["email"] == "derek@example.com" and me["role"] == "admin" and me["kind"] == "user"
    u = mod.db.get_user_by_sub("gsub-1")
    assert u["role"] == "admin"


def test_callback_member_role_for_non_admin_email(mod):
    c = client(mod)
    info = dict(USERINFO, sub="gsub-2", email="stranger@example.com")
    r = _google_login(mod, c, info)
    assert r.status_code == 302
    assert c.get("/api/me").json()["role"] == "member"


def test_callback_rejects_bad_state(mod):
    c = client(mod)
    c.get("/auth/login", follow_redirects=False)                   # sets state cookie
    r = c.get("/auth/callback?code=abc&state=forged", follow_redirects=False)
    assert r.status_code == 403


def test_callback_rejects_unverified_email(mod):
    c = client(mod)
    info = dict(USERINFO, email_verified=False)
    r = _google_login(mod, c, info)
    assert r.status_code == 403


def test_closed_signup_rejects_strangers_allows_admin(mod, monkeypatch):
    monkeypatch.setenv("SONAVE_SIGNUP_MODE", "closed")
    c = client(mod)
    info = dict(USERINFO, sub="gsub-3", email="stranger2@example.com")
    assert _google_login(mod, c, info).status_code == 403
    c2 = client(mod)
    assert _google_login(mod, c2).status_code == 302               # admin email passes


def test_logout_clears_session(mod):
    c = client(mod)
    _google_login(mod, c)
    assert c.get("/api/me").status_code == 200
    c.post("/auth/logout", follow_redirects=False)
    assert c.get("/api/me").status_code == 401


def test_callback_sets_partitioned_companion_cookie(mod):
    c = client(mod)
    r = _google_login(mod, c)
    cookies = r.headers.get_list("set-cookie")
    part = [x for x in cookies if x.startswith("sonave_session_p=")]
    assert part and "Partitioned" in part[0] and "SameSite=None" in part[0]


def test_partitioned_cookie_authenticates(mod):
    u = mod.db.upsert_google_user("s-part", "part@x.com", "P", "", "member")
    c = TestClient(mod.app, base_url="https://testserver")
    c.cookies.set("sonave_session_p", mod.auth.sign_session(u["id"]))
    assert c.get("/api/me").json()["email"] == "part@x.com"


def test_bearer_session_token_authenticates(mod):
    u = mod.db.upsert_google_user("s-bear", "bear@x.com", "B", "", "member")
    tok = mod.auth.sign_session(u["id"])
    r = client(mod).get("/api/me", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200 and r.json()["email"] == "bear@x.com"


def test_popup_callback_posts_token_and_closes(mod):
    c = client(mod)
    mod.auth._exchange_code = lambda code: {"access_token": "at"}
    mod.auth._fetch_userinfo = lambda at: dict(USERINFO)
    r = c.get("/auth/login?ctx=popup", follow_redirects=False)
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    r2 = c.get(f"/auth/callback?code=abc&state={state}", follow_redirects=False)
    assert r2.status_code == 200                       # HTML page, not a redirect
    assert "postMessage" in r2.text and "window.close()" in r2.text
    assert "sonave_auth" in r2.text
    cookies = r2.headers.get_list("set-cookie")
    assert any(x.startswith("sonave_session=") for x in cookies)   # cookies still set
    assert c.get("/api/me").status_code == 200


def test_meet_addon_page_renders(mod):
    r = client(mod).get("/meet-addon")
    assert r.status_code == 200
    assert "createAddonSession" in r.text and "__FAVICON__" not in r.text


# --- machine token unchanged --------------------------------------------------
def test_machine_token_still_opens_everything(mod):
    c = client(mod)
    assert c.get("/api/quality").status_code == 401
    h = {"X-Sonave-Token": TOKEN}
    for path in ("/api/quality", "/captures", "/api/incidents", "/api/enrolled", "/api/me"):
        assert c.get(path, headers=h).status_code == 200, path
    assert c.get("/api/me", headers=h).json()["kind"] == "machine"
