"""Admin observability: activity events, admin API, growth pushes."""
import base64
import hashlib
import hmac
import io
import json
import time
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

import conftest

TOKEN = "machine-token-123"
WH_SECRET = "whsec_test"

USERINFO = {"sub": "gsub-adm", "email": "derek@example.com", "email_verified": True,
            "name": "Derek", "picture": ""}


@pytest.fixture
def mod(monkeypatch, tmp_path):
    monkeypatch.setenv("SONAVE_SCORER_URL", "")
    monkeypatch.setenv("SONAVE_RECALL_API_KEY", "test-key")
    monkeypatch.setenv("SONAVE_API_TOKEN", TOKEN)
    monkeypatch.setenv("SONAVE_SESSION_SECRET", "sec")
    monkeypatch.setenv("SONAVE_GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("SONAVE_GOOGLE_CLIENT_SECRET", "cs")
    monkeypatch.setenv("SONAVE_ADMIN_EMAILS", "derek@example.com")
    monkeypatch.setenv("SONAVE_STRIPE_WEBHOOK_SECRET", WH_SECRET)
    monkeypatch.setenv("SONAVE_PUBLIC_DOMAIN", "test.sonave.dev")
    m = conftest.load_module("rwapp_admin", "railway/app.py")
    m.QUALITY.clear()
    m.VERDICTS.clear()
    m.ROLL.clear()
    return m


def _mk_user(m, sub, email, role="member"):
    return m.db.upsert_google_user(sub, email, email.split("@")[0], "", role)


def _client_as(m, uid):
    c = TestClient(m.app, base_url="https://testserver")
    c.cookies.set("sonave_session", m.auth.sign_session(uid))
    return c


def _kinds(m, uid=None):
    return [e["kind"] for e in m.db.list_events(user_id=uid)]


def _google_login(m, c, userinfo=USERINFO):
    m.auth._exchange_code = lambda code: {"access_token": "at"}
    m.auth._fetch_userinfo = lambda at: dict(userinfo)
    r = c.get("/auth/login", follow_redirects=False)
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    return c.get(f"/auth/callback?code=abc&state={state}", follow_redirects=False)


# --- db layer -----------------------------------------------------------------
def test_events_roundtrip_filters_and_cursor(mod):
    for i in range(5):
        mod.db.add_event("u1", "signin" if i % 2 else "bot_created", json.dumps({"i": i}))
    mod.db.add_event("u2", "signin")
    assert len(mod.db.list_events()) == 6
    assert all(e["user_id"] == "u1" for e in mod.db.list_events(user_id="u1"))
    assert all(e["kind"] == "signin" for e in mod.db.list_events(kind="signin"))
    top = mod.db.list_events(limit=2)
    assert len(top) == 2 and top[0]["id"] > top[1]["id"]            # newest first
    older = mod.db.list_events(before_id=top[-1]["id"], limit=100)
    assert all(e["id"] < top[-1]["id"] for e in older)


def test_add_event_never_raises(mod, monkeypatch):
    monkeypatch.setenv("SONAVE_APP_DB", "Z:\\no\\such\\dir\\x.db")
    mod.db.add_event("u1", "signin")                                 # must not raise


# --- auth events --------------------------------------------------------------
def test_signup_then_signin_then_signout(mod):
    c = _client_as(mod, "none")  # fresh client; cookies replaced by login flow
    c = TestClient(mod.app, base_url="https://testserver")
    _google_login(mod, c)
    u = mod.db.get_user_by_sub("gsub-adm")
    assert _kinds(mod, u["id"]) == ["signup"]
    c2 = TestClient(mod.app, base_url="https://testserver")
    _google_login(mod, c2)
    assert _kinds(mod, u["id"])[0] == "signin"                       # newest first
    c2.post("/auth/logout", follow_redirects=False)
    assert _kinds(mod, u["id"])[0] == "signout"
    # logout with no session: no event, no 500
    n = len(mod.db.list_events())
    r = TestClient(mod.app).post("/auth/logout", follow_redirects=False)
    assert r.status_code == 302 and len(mod.db.list_events()) == n


# --- bot / meeting events -----------------------------------------------------
class _R(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_bot_created_manual_and_autojoin_and_denied(mod, monkeypatch):
    ua = _mk_user(mod, "s-a1", "a1@x.com")
    monkeypatch.setattr(mod.urllib.request, "urlopen",
                        lambda req, timeout=None: _R(json.dumps({"id": "b-1"}).encode()))
    monkeypatch.setattr(mod, "_recall_bot_status", lambda bid: "in_call_recording")
    _client_as(mod, ua["id"]).post("/bot", json={"meeting_url": "https://meet.google.com/adm"})
    ev = mod.db.list_events(user_id=ua["id"], kind="bot_created")
    assert len(ev) == 1
    d = json.loads(ev[0]["detail"])
    assert d["source"] == "manual" and d["meeting_url"].endswith("/adm")
    # autojoin source
    ub = _mk_user(mod, "s-a2", "a2@x.com")
    mod.db.set_ical_url(ub["id"], "https://calendar.google.com/x.ics")
    now = time.time()
    ics = ("BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\nUID:e1\r\n"
           + time.strftime("DTSTART:%Y%m%dT%H%M%SZ\r\n", time.gmtime(now))
           + "X-GOOGLE-CONFERENCE:https://meet.google.com/aj-adm\r\n"
           "END:VEVENT\r\nEND:VCALENDAR")
    monkeypatch.setattr(mod.autojoin, "fetch_ics", lambda url, timeout=12: ics)
    mod._autojoin_tick(now=now)
    d2 = json.loads(mod.db.list_events(user_id=ub["id"], kind="bot_created")[0]["detail"])
    assert d2["source"] == "autojoin"
    # denied: exceed the free concurrent-bot cap
    monkeypatch.setenv("SONAVE_MAX_CONCURRENT_BOTS", "0")
    _client_as(mod, ua["id"]).post("/bot", json={"meeting_url": "https://meet.google.com/den"})
    dd = json.loads(mod.db.list_events(user_id=ua["id"], kind="bot_denied")[0]["detail"])
    assert dd["code"] == "too_many_bots"


def test_meeting_started_and_ended_events(mod, monkeypatch):
    ua = _mk_user(mod, "s-m1", "m1@x.com")
    tok = "tok-m1"
    mod.db.insert_bot("bot-m1", ua["id"], hashlib.sha256(tok.encode()).hexdigest(),
                      "https://meet.google.com/m")
    frame = base64.b64encode(b"\x00\x08" * mod.SR).decode()
    with TestClient(mod.app).websocket_connect(f"/api/ws/audio?token={tok}") as ws:
        for _ in range(2):
            ws.send_text(json.dumps({"data": {"data": {"buffer": frame,
                                                        "participant": {"name": "A"}}}}))
    kinds = _kinds(mod, ua["id"])
    assert kinds[0] == "meeting_ended" and kinds[-1] == "meeting_started"
    d = json.loads(mod.db.list_events(user_id=ua["id"], kind="meeting_ended")[0]["detail"])
    assert "duration_sec" in d and "metered_min" in d


# --- settings events ----------------------------------------------------------
def test_settings_changed_fields_only_no_urls(mod):
    ua = _mk_user(mod, "s-s1", "s1@x.com")
    c = _client_as(mod, ua["id"])
    c.post("/api/settings", json={"alert_webhook": "", "ical_url": "https://x.y/z.ics"})
    ev = mod.db.list_events(user_id=ua["id"], kind="settings_changed")
    assert len(ev) == 1
    assert json.loads(ev[0]["detail"]) == {"ical_url": "set"}
    assert "x.y" not in ev[0]["detail"]                              # never the URL itself
    c.post("/api/settings", json={"alert_webhook": "", "ical_url": "https://x.y/z.ics"})
    assert len(mod.db.list_events(user_id=ua["id"], kind="settings_changed")) == 1  # no-op


# --- billing events -----------------------------------------------------------
def _signed(body: bytes, secret=WH_SECRET):
    t = int(time.time())
    mac = hmac.new(secret.encode(), f"{t}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={t},v1={mac}"


def test_subscription_events_from_webhook(mod):
    ua = _mk_user(mod, "s-w1", "w1@x.com")
    body = json.dumps({"id": "evt_a1", "type": "checkout.session.completed",
                       "data": {"object": {"client_reference_id": ua["id"],
                                           "customer": "cus_9", "subscription": "sub_9"}}}).encode()
    c = TestClient(mod.app)
    c.post("/api/billing/webhook", content=body, headers={"Stripe-Signature": _signed(body)})
    assert _kinds(mod, ua["id"])[0] == "subscription_checkout"
    body2 = json.dumps({"id": "evt_a2", "type": "customer.subscription.deleted",
                        "data": {"object": {"customer": "cus_9", "id": "sub_9"}}}).encode()
    c.post("/api/billing/webhook", content=body2, headers={"Stripe-Signature": _signed(body2)})
    assert _kinds(mod, ua["id"])[0] == "subscription_canceled"
    # replayed event id -> no duplicate
    c.post("/api/billing/webhook", content=body2, headers={"Stripe-Signature": _signed(body2)})
    assert len(mod.db.list_events(user_id=ua["id"], kind="subscription_canceled")) == 1


# --- admin API ----------------------------------------------------------------
def test_admin_endpoints_are_admin_only(mod):
    ua = _mk_user(mod, "s-r1", "r1@x.com")
    adm = _mk_user(mod, "s-r2", "boss@x.com", role="admin")
    member, admin = _client_as(mod, ua["id"]), _client_as(mod, adm["id"])
    machine = TestClient(mod.app, base_url="https://testserver")
    for path in ("/api/admin/overview", "/api/admin/users", "/api/admin/events"):
        assert member.get(path).status_code == 403, path
        assert admin.get(path).status_code == 200, path
        assert machine.get(path, headers={"X-Sonave-Token": TOKEN}).status_code == 200, path


def test_admin_overview_and_rollup_numbers(mod, monkeypatch):
    adm = _mk_user(mod, "s-o0", "boss2@x.com", role="admin")
    ua = _mk_user(mod, "s-o1", "o1@x.com")
    mod.db.upsert_subscription(ua["id"], "cus_o", "sub_o", "active")
    mod.db.add_usage_minutes(ua["id"], mod.billing.month_key(), 42.0)
    mod.db.insert_bot("bot-o1", ua["id"], "h", "https://meet.google.com/o")
    mod.db.add_event(ua["id"], "signin")
    c = _client_as(mod, adm["id"])
    ov = c.get("/api/admin/overview").json()
    assert ov["users_total"] == 2 and ov["users_new_7d"] == 2
    assert ov["subs_active"] == 1 and ov["bots_24h"] == 1
    assert ov["minutes_month"] == 42.0
    users = {u["email"]: u for u in c.get("/api/admin/users").json()["users"]}
    row = users["o1@x.com"]
    assert row["plan"] == "metered" and row["bots_total"] == 1
    assert row["minutes_month"] == 42.0 and row["last_event_ts"] is not None
    assert users["boss2@x.com"]["plan"] == "admin"
    ev = c.get(f"/api/admin/events?kind=signin&user_id={ua['id']}").json()["events"]
    assert len(ev) == 1 and ev[0]["email"] == "o1@x.com"


# --- growth pushes ------------------------------------------------------------
def test_notify_admin_fires_on_signup(mod, monkeypatch):
    sent = {}

    def _fake_thread(target=None, args=(), daemon=None):
        class _T:
            def start(self):
                target(*args)
        return _T()

    monkeypatch.setenv("SONAVE_ADMIN_WEBHOOK", "https://hooks.slack.example/x")
    monkeypatch.setattr(mod.threading, "Thread", _fake_thread)

    def _open(req, timeout=None):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data)
        return _R(b"ok")

    monkeypatch.setattr(mod.urllib.request, "urlopen", _open)
    c = TestClient(mod.app, base_url="https://testserver")
    _google_login(mod, c, dict(USERINFO, sub="gsub-new", email="new@x.com"))
    assert "hooks.slack.example" in sent.get("url", "")
    assert "new@x.com" in sent["body"]["text"]


def test_notify_admin_silent_when_unconfigured(mod, monkeypatch):
    called = {"n": 0}

    def _open(req, timeout=None):
        called["n"] += 1
        return _R(b"ok")

    monkeypatch.setattr(mod.urllib.request, "urlopen", _open)
    mod._notify_admin("quiet test")
    time.sleep(0.05)
    assert called["n"] == 0                                          # no env -> no push
