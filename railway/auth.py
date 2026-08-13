"""auth.py — Google OAuth sign-in, HMAC session cookies, and request principals.

Zero dependencies beyond stdlib: the Authorization Code flow is two urllib POSTs;
sessions are b64url(payload).b64url(HMAC_SHA256(secret, payload)). The machine
token (SONAVE_API_TOKEN) keeps working everywhere it works today and resolves to
the admin workspace, so Recall / Modal / CI / verdict_monitor never notice auth.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets as pysecrets
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

import db

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
# Calendar auto-join later: append "https://www.googleapis.com/auth/calendar.readonly"
OAUTH_SCOPES = ["openid", "email", "profile"]

SESSION_COOKIE = "sonave_session"
# Partitioned (CHIPS) companion cookie: SameSite=None + Partitioned, so the Meet
# add-on iframe (third-party context) can authenticate. Partitioning keys it to
# the embedding site, and our APIs are JSON-only (cross-site form posts 422,
# cross-origin fetch needs CORS we don't grant), so CSRF surface stays closed.
PARTITIONED_COOKIE = "sonave_session_p"
STATE_COOKIE = "sonave_oauth_state"
SESSION_TTL = 30 * 24 * 3600
STATE_TTL = 600

# The pre-signup workspace id machine principals use until an admin user exists.
MACHINE_WORKSPACE = "admin"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def google_configured() -> bool:
    return bool(_env("SONAVE_GOOGLE_CLIENT_ID") and _env("SONAVE_GOOGLE_CLIENT_SECRET"))


def _session_secret() -> bytes:
    s = _env("SONAVE_SESSION_SECRET")
    if not s:
        # Open/dev mode fallback: sessions won't survive restarts, which is fine
        # because without Google configured there are no sessions to protect.
        s = "dev-" + hashlib.sha256(_env("SONAVE_API_TOKEN", "sonave").encode()).hexdigest()
    return s.encode()


@dataclass
class Principal:
    kind: str            # "user" | "machine" | "bot"
    user_id: str
    role: str = "member"
    email: str = ""
    name: str = ""
    picture: str = ""


# --- signed tokens (sessions + oauth state) ----------------------------------
def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(payload: dict) -> str:
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64e(hmac.new(_session_secret(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def _verify(token: str) -> dict | None:
    try:
        body, sig = token.split(".", 1)
        want = _b64e(hmac.new(_session_secret(), body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, want):
            return None
        payload = json.loads(_b64d(body))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:  # noqa: BLE001 — any malformed token is just invalid
        return None


def sign_session(uid: str, session_ver: int = 1) -> str:
    return _sign({"uid": uid, "v": session_ver, "iat": int(time.time()),
                  "exp": int(time.time() + SESSION_TTL)})


def verify_session(token: str | None) -> str | None:
    """Returns the user id, checking signature, expiry, and session_ver."""
    if not token:
        return None
    p = _verify(token)
    if not p or "uid" not in p:
        return None
    user = db.get_user(p["uid"])
    if not user or user.get("session_ver", 1) != p.get("v", 1):
        return None
    return p["uid"]


def make_state() -> str:
    return _sign({"n": pysecrets.token_urlsafe(16), "exp": int(time.time() + STATE_TTL)})


def verify_state(cookie_val: str | None, param_val: str | None) -> bool:
    if not cookie_val or not param_val or not hmac.compare_digest(cookie_val, param_val):
        return False
    return _verify(param_val) is not None


# --- Google endpoints (urllib; monkeypatched in tests) ------------------------
def redirect_uri() -> str:
    domain = _env("SONAVE_PUBLIC_DOMAIN") or _env("RAILWAY_PUBLIC_DOMAIN") or "localhost:8000"
    scheme = "http" if domain.startswith(("localhost", "127.0.0.1")) else "https"
    return f"{scheme}://{domain}/auth/callback"


def login_url(state: str) -> str:
    q = urllib.parse.urlencode({
        "client_id": _env("SONAVE_GOOGLE_CLIENT_ID"),
        "redirect_uri": redirect_uri(),
        "response_type": "code",
        "scope": " ".join(OAUTH_SCOPES),
        "state": state,
        "prompt": "select_account",
    })
    return f"{GOOGLE_AUTH_URL}?{q}"


def _exchange_code(code: str) -> dict:
    body = urllib.parse.urlencode({
        "code": code,
        "client_id": _env("SONAVE_GOOGLE_CLIENT_ID"),
        "client_secret": _env("SONAVE_GOOGLE_CLIENT_SECRET"),
        "redirect_uri": redirect_uri(),
        "grant_type": "authorization_code",
    }).encode()
    req = urllib.request.Request(GOOGLE_TOKEN_URL, data=body,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def _fetch_userinfo(access_token: str) -> dict:
    req = urllib.request.Request(GOOGLE_USERINFO_URL,
                                 headers={"Authorization": f"Bearer {access_token}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def complete_login(code: str) -> dict:
    """Code -> tokens -> userinfo -> user row. Raises ValueError on policy failures."""
    tokens = _exchange_code(code)
    info = _fetch_userinfo(tokens["access_token"])
    if not info.get("email_verified", False):
        raise ValueError("email not verified")
    email = (info.get("email") or "").lower()
    admins = [e.strip().lower() for e in _env("SONAVE_ADMIN_EMAILS").split(",") if e.strip()]
    role = "admin" if email in admins else "member"
    if _env("SONAVE_SIGNUP_MODE", "open") == "closed" and role != "admin":
        existing = db.get_user_by_sub(info["sub"]) if hasattr(db, "get_user_by_sub") else None
        if not existing:
            raise ValueError("signups closed")
    return db.upsert_google_user(info["sub"], email, info.get("name", ""),
                                 info.get("picture", ""), role)


# --- principals ---------------------------------------------------------------
def _machine_token_ok(tok: str | None) -> bool:
    api = _env("SONAVE_API_TOKEN")
    return bool(api and tok and hmac.compare_digest(tok, api))


def get_principal(request) -> Principal | None:
    auth_hdr = request.headers.get("authorization", "")
    bearer = auth_hdr[7:] if auth_hdr.lower().startswith("bearer ") else ""
    tok = (request.headers.get("x-sonave-token") or bearer
           or request.cookies.get("sonave_token") or request.query_params.get("token"))
    if _machine_token_ok(tok):
        uid = db.first_admin_id() or MACHINE_WORKSPACE
        return Principal("machine", uid, "admin", email="operator")
    uid = verify_session(request.cookies.get(SESSION_COOKIE)) \
        or verify_session(request.cookies.get(PARTITIONED_COOKIE))
    if uid:
        u = db.get_user(uid)
        if u:
            return Principal("user", uid, u.get("role", "member"),
                             email=u.get("email", ""), name=u.get("name", ""),
                             picture=u.get("picture", ""))
    if not _env("SONAVE_API_TOKEN") and not google_configured():
        # Fully-open dev posture (today's behavior when no token is set).
        return Principal("machine", db.first_admin_id() or MACHINE_WORKSPACE, "admin", email="operator")
    return None
