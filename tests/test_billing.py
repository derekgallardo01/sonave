"""Stripe metered billing (Stage C): webhook signatures, entitlements, quotas,
meter events. Signatures are real HMACs computed in-test — no stripe-mock."""
import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient

import conftest

TOKEN = "machine-token-123"
WH_SECRET = "whsec_test"


@pytest.fixture
def mod(monkeypatch, tmp_path):
    monkeypatch.setenv("SONAVE_SCORER_URL", "")
    monkeypatch.setenv("SONAVE_RECALL_API_KEY", "test-key")
    monkeypatch.setenv("SONAVE_API_TOKEN", TOKEN)
    monkeypatch.setenv("SONAVE_SESSION_SECRET", "sec")
    monkeypatch.setenv("SONAVE_STRIPE_SECRET_KEY", "sk_test_x")
    monkeypatch.setenv("SONAVE_STRIPE_WEBHOOK_SECRET", WH_SECRET)
    monkeypatch.setenv("SONAVE_STRIPE_PRICE_METERED", "price_x")
    monkeypatch.setenv("SONAVE_FREE_MINUTES", "300")
    m = conftest.load_module("rwapp_billing", "railway/app.py")
    m.QUALITY.clear()
    m.VERDICTS.clear()
    m.ROLL.clear()
    return m


def _user(m, sub="s-b", email="b@x.com", role="member"):
    return m.db.upsert_google_user(sub, email, "B", "", role)


def _client_as(m, uid):
    c = TestClient(m.app, base_url="https://testserver")
    c.cookies.set("sonave_session", m.auth.sign_session(uid))
    return c


def _signed(body: bytes, secret=WH_SECRET, t=None):
    t = int(t if t is not None else time.time())
    mac = hmac.new(secret.encode(), f"{t}.".encode() + body, hashlib.sha256).hexdigest()
    return f"t={t},v1={mac}"


# --- webhook ------------------------------------------------------------------
def test_webhook_valid_signature_activates_plan(mod):
    u = _user(mod)
    body = json.dumps({"id": "evt_1", "type": "checkout.session.completed",
                       "data": {"object": {"client_reference_id": u["id"],
                                           "customer": "cus_1", "subscription": "sub_1"}}}).encode()
    r = TestClient(mod.app).post("/api/billing/webhook", content=body,
                                 headers={"Stripe-Signature": _signed(body)})
    assert r.status_code == 200
    sub = mod.db.get_subscription(u["id"])
    assert sub["status"] == "active" and sub["stripe_customer_id"] == "cus_1"
    assert mod.billing.entitlement(u["id"])["plan"] == "metered"


def test_webhook_rejects_bad_signature_and_stale_timestamp(mod):
    body = json.dumps({"id": "evt_2", "type": "x"}).encode()
    c = TestClient(mod.app)
    assert c.post("/api/billing/webhook", content=body,
                  headers={"Stripe-Signature": "t=1,v1=deadbeef"}).status_code == 400
    stale = _signed(body, t=time.time() - 3600)
    assert c.post("/api/billing/webhook", content=body,
                  headers={"Stripe-Signature": stale}).status_code == 400


def test_webhook_replay_is_noop(mod):
    u = _user(mod, sub="s-r", email="r@x.com")
    body = json.dumps({"id": "evt_dup", "type": "checkout.session.completed",
                       "data": {"object": {"client_reference_id": u["id"],
                                           "customer": "cus_r", "subscription": "sub_r"}}}).encode()
    c = TestClient(mod.app)
    h = {"Stripe-Signature": _signed(body)}
    assert c.post("/api/billing/webhook", content=body, headers=h).json().get("duplicate") is None
    r2 = c.post("/api/billing/webhook", content=body,
                headers={"Stripe-Signature": _signed(body)})
    assert r2.json().get("duplicate") is True


def test_webhook_subscription_deleted_downgrades(mod):
    u = _user(mod, sub="s-del", email="del@x.com")
    mod.db.upsert_subscription(u["id"], "cus_del", "sub_del", "active")
    body = json.dumps({"id": "evt_del", "type": "customer.subscription.deleted",
                       "data": {"object": {"customer": "cus_del", "id": "sub_del"}}}).encode()
    TestClient(mod.app).post("/api/billing/webhook", content=body,
                             headers={"Stripe-Signature": _signed(body)})
    assert mod.billing.entitlement(u["id"])["plan"] == "free"


# --- entitlement math ---------------------------------------------------------
def test_free_tier_quota_and_month_rollover(mod, monkeypatch):
    u = _user(mod, sub="s-q", email="q@x.com")
    assert mod.billing.can_launch_bot(u["id"]) is None
    mod.db.add_usage_minutes(u["id"], mod.billing.month_key(), 299)
    assert mod.billing.can_launch_bot(u["id"]) is None
    mod.db.add_usage_minutes(u["id"], mod.billing.month_key(), 2)      # over 300
    denied = mod.billing.can_launch_bot(u["id"])
    assert denied and denied["code"] == "quota_exceeded"
    # a new month resets the free allowance
    monkeypatch.setattr(mod.billing, "month_key", lambda ts=None: "2099-01")
    assert mod.billing.can_launch_bot(u["id"]) is None


def test_metered_plan_allows_past_free_until_spend_cap(mod):
    u = _user(mod, sub="s-m", email="m@x.com")
    mod.db.upsert_subscription(u["id"], "cus_m", "sub_m", "active")
    mod.db.add_usage_minutes(u["id"], mod.billing.month_key(), 400)    # past free
    assert mod.billing.can_launch_bot(u["id"]) is None
    # 300 free + 1500 billable min = $200 at $8/hr -> cap
    mod.db.add_usage_minutes(u["id"], mod.billing.month_key(), 1400)
    denied = mod.billing.can_launch_bot(u["id"])
    assert denied and denied["code"] == "spend_cap"


def test_admin_is_unlimited(mod):
    a = _user(mod, sub="s-a", email="a@x.com", role="admin")
    mod.db.add_usage_minutes(a["id"], mod.billing.month_key(), 100000)
    assert mod.billing.can_launch_bot(a["id"], "admin") is None
    assert mod.billing.entitlement(a["id"], "admin")["plan"] == "admin"


def test_bot_endpoint_returns_402_when_quota_exhausted(mod, monkeypatch):
    u = _user(mod, sub="s-402", email="x402@x.com")
    mod.db.add_usage_minutes(u["id"], mod.billing.month_key(), 301)
    r = _client_as(mod, u["id"]).post("/bot", json={"meeting_url": "https://meet.google.com/abc"})
    assert r.status_code == 402 and r.json()["code"] == "quota_exceeded"


# --- meter events -------------------------------------------------------------
def test_meter_usage_reports_only_billable_slice(mod, monkeypatch):
    u = _user(mod, sub="s-me", email="me@x.com")
    mod.db.upsert_subscription(u["id"], "cus_me", "sub_me", "active")
    sent = []
    monkeypatch.setattr(mod.billing, "_stripe_post", lambda path, form: sent.append((path, form)) or {})
    mod.billing.meter_usage(u["id"], "member", 299, "k1")     # fully inside free tier
    assert sent == []
    mod.billing.meter_usage(u["id"], "member", 10, "k2")      # 9 of these 10 are billable
    assert len(sent) == 1
    path, form = sent[0]
    assert path == "/billing/meter_events"
    assert float(form["payload[value]"]) == pytest.approx(9.0)
    assert form["identifier"] == "k2"


def test_meter_usage_never_reports_for_admin(mod, monkeypatch):
    a = _user(mod, sub="s-madm", email="madm@x.com", role="admin")
    sent = []
    monkeypatch.setattr(mod.billing, "_stripe_post", lambda path, form: sent.append(1) or {})
    mod.billing.meter_usage(a["id"], "admin", 10000, "k")
    assert sent == []


# --- checkout / portal --------------------------------------------------------
def test_checkout_and_portal_endpoints(mod, monkeypatch):
    u = _user(mod, sub="s-co", email="co@x.com")
    monkeypatch.setattr(mod.billing, "_stripe_post",
                        lambda path, form: {"url": f"https://stripe.test{path}"})
    c = _client_as(mod, u["id"])
    r = c.post("/api/billing/checkout")
    assert r.json()["ok"] and "checkout/sessions" in r.json()["url"]
    assert c.post("/api/billing/portal").json()["ok"] is False   # no customer yet
    mod.db.upsert_subscription(u["id"], "cus_co", "sub_co", "active")
    assert "billing_portal" in c.post("/api/billing/portal").json()["url"]
