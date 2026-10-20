import json
import time
import pytest
from fastapi.testclient import TestClient
import conftest

TOKEN = "machine-token-meet-meter"
WH_SECRET = "whsec_test_meter"


@pytest.fixture
def mod(monkeypatch, tmp_path):
    monkeypatch.setenv("SONAVE_SCORER_URL", "")
    monkeypatch.setenv("SONAVE_RECALL_API_KEY", "test-key")
    monkeypatch.setenv("SONAVE_API_TOKEN", TOKEN)
    monkeypatch.setenv("SONAVE_SESSION_SECRET", "sec-meet-meter")
    monkeypatch.setenv("SONAVE_STRIPE_SECRET_KEY", "sk_test_x")
    monkeypatch.setenv("SONAVE_STRIPE_WEBHOOK_SECRET", WH_SECRET)
    monkeypatch.setenv("SONAVE_STRIPE_PRICE_METERED", "price_x")
    monkeypatch.setenv("SONAVE_FREE_MINUTES", "120")
    m = conftest.load_module("rwapp_meet_meter", "railway/app.py")
    m.QUALITY.clear()
    m.VERDICTS.clear()
    m.ROLL.clear()
    m.MEET_METER_LAST_TS.clear()
    m.MEET_SESSION_START.clear()
    return m


def _user(m, sub="s-meet-user", email="meetuser@x.com", role="member"):
    return m.db.upsert_google_user(sub, email, "Meet User", "", role)


def _client_as(m, uid):
    c = TestClient(m.app, base_url="https://testserver")
    c.cookies.set("sonave_session", m.auth.sign_session(uid))
    return c


def test_meet_meter_tick_and_flush_increments_db_usage(mod):
    u = _user(mod, sub="s-tick-1", email="tick1@x.com")
    uid = u["id"]

    # Initial state: 0 minutes used
    assert mod.db.get_usage_minutes(uid, mod.billing.month_key()) == 0.0

    # First tick initializes the clock
    mod._meter_meet_tick(uid, space_id="test-space")
    assert mod.db.get_usage_minutes(uid, mod.billing.month_key()) == 0.0
    assert uid in mod.MEET_METER_LAST_TS

    # Fast forward clock by 65 seconds
    mod.MEET_METER_LAST_TS[uid] -= 65.0
    mod._meter_meet_tick(uid, space_id="test-space")
    used = mod.db.get_usage_minutes(uid, mod.billing.month_key())
    assert pytest.approx(used, 0.1) == 1.08  # ~65s / 60 = 1.08 min

    # Disconnect / flush call with 20 seconds remaining
    mod.MEET_METER_LAST_TS[uid] -= 20.0
    mod._meter_meet_flush(uid, space_id="test-space")
    used_after = mod.db.get_usage_minutes(uid, mod.billing.month_key())
    assert used_after > used
    assert pytest.approx(used_after, 0.1) == 1.41
    assert uid not in mod.MEET_METER_LAST_TS


def test_meet_meter_deduplicates_simultaneous_speakers(mod):
    u = _user(mod, sub="s-dedup", email="dedup@x.com")
    uid = u["id"]

    # First tick
    mod._meter_meet_tick(uid, space_id="space-dedup")
    t0 = mod.MEET_METER_LAST_TS[uid]

    # Multiple ticks within 10 seconds (e.g. 5 audio chunks from 3 speakers)
    for _ in range(5):
        mod._meter_meet_tick(uid, space_id="space-dedup")

    # Time didn't advance >= 60s, so usage is not double-counted
    assert mod.db.get_usage_minutes(uid, mod.billing.month_key()) == 0.0
    assert mod.MEET_METER_LAST_TS[uid] == t0


def test_api_me_reflects_metered_minutes_and_remaining_free(mod):
    u = _user(mod, sub="s-api-me-meter", email="apime@x.com")
    uid = u["id"]
    c = _client_as(mod, uid)

    # Base allowance = 120 minutes
    r0 = c.get("/api/me").json()
    assert r0["plan"] == "free"
    assert r0["free_minutes"] == 120
    assert r0["remaining_free"] == 120.0
    assert r0["minutes_used"] == 0.0

    # Add 45 minutes of monitored Meet usage
    mod.db.add_usage_minutes(uid, mod.billing.month_key(), 45.0)

    r1 = c.get("/api/me").json()
    assert r1["minutes_used"] == 45.0
    assert r1["remaining_free"] == 75.0


def test_quota_exhausted_paywall_and_connect_block(mod):
    u = _user(mod, sub="s-exhaust", email="exhaust@x.com")
    uid = u["id"]
    c = _client_as(mod, uid)

    # Use all 120 free minutes
    mod.db.add_usage_minutes(uid, mod.billing.month_key(), 125.0)

    # /api/me reports exhausted free quota
    me = c.get("/api/me").json()
    assert me["remaining_free"] == 0.0
    assert me["active"] is False or me["plan"] == "free"

    # Trying to connect a new meeting returns quota_exceeded
    r = c.post("/api/meet/sessions/connect", json={"space_id": "meet.google.com/xyz-test"})
    assert r.status_code == 200
    res = r.json()
    assert res["ok"] is False
    assert res["code"] == "quota_exceeded"
    assert res["exhausted"] is True


def test_checkout_endpoints_track_upgrade_event(mod, monkeypatch):
    u = _user(mod, sub="s-upg", email="upg@x.com")
    uid = u["id"]
    monkeypatch.setattr(mod.billing, "create_checkout", lambda uid, base: "https://checkout.stripe.test/session_123")
    c = _client_as(mod, uid)

    # POST /api/billing/checkout
    r = c.post("/api/billing/checkout")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "https://checkout.stripe.test/session_123" in data["url"]

    # Verify addon_upgrade_clicked was tracked
    evs = mod.db.list_events(user_id=uid, limit=10)
    assert any(e["kind"] == "addon_upgrade_clicked" for e in evs)

    # GET /billing/upgrade redirect route
    r_redir = c.get("/billing/upgrade", follow_redirects=False)
    assert r_redir.status_code == 303
    assert r_redir.headers["location"] == "https://checkout.stripe.test/session_123"


def test_review_bonus_unlocks_exhausted_quota_instantly(mod):
    u = _user(mod, sub="s-unlock", email="unlock@x.com")
    uid = u["id"]
    c = _client_as(mod, uid)

    # Exhaust 120 minutes base quota
    mod.db.add_usage_minutes(uid, mod.billing.month_key(), 125.0)
    assert mod.billing.can_monitor_meeting(uid) is not None

    # User clicks marketplace review prompt
    c.post("/api/telemetry/event", json={"kind": "review_prompt_clicked", "detail": {"space": "test-space"}})

    # Now total free minutes = 120 + 240 = 360 minutes
    # 125 used < 360 allowed -> call is allowed again!
    assert mod.billing.can_monitor_meeting(uid) is None
    me = c.get("/api/me").json()
    assert me["free_minutes"] == 360
    assert me["remaining_free"] == 235.0


def test_meet_addon_template_contains_paywall_and_upgrade_elements():
    with open("railway/meet-addon.html", "r", encoding="utf-8") as f:
        html = f.read()

    assert 'id="planBar"' in html
    assert 'id="planBadge"' in html
    assert 'id="planRemaining"' in html
    assert 'id="btnUpgrade"' in html
    assert 'id="paywallCard"' in html
    assert 'Monthly Free Quota Exhausted' in html
    assert 'startUpgradeCheckout' in html
    assert 'refreshEntitlement' in html
    assert 'updatePlanUI' in html
