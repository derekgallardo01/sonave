"""Tests for +4 Free Hours marketplace review bonus credit wiring into user metadata and billing."""
import json
import time
import pytest
from fastapi.testclient import TestClient

import conftest

TOKEN = "machine-token-123"


@pytest.fixture
def mod(monkeypatch, tmp_path):
    monkeypatch.setenv("SONAVE_SCORER_URL", "")
    monkeypatch.setenv("SONAVE_RECALL_API_KEY", "test-key")
    monkeypatch.setenv("SONAVE_API_TOKEN", TOKEN)
    monkeypatch.setenv("SONAVE_APP_DB", str(tmp_path / "app.db"))
    monkeypatch.setenv("SONAVE_SESSION_SECRET", "test-secret")
    monkeypatch.setenv("SONAVE_ADMIN_EMAILS", "boss@sonave.dev")
    monkeypatch.setenv("SONAVE_PUBLIC_DOMAIN", "localhost:8000")
    monkeypatch.setenv("SONAVE_FREE_MINUTES", "300")
    monkeypatch.setenv("SONAVE_STRIPE_SECRET_KEY", "sk_test_x")
    m = conftest.load_module("rwapp_bonus", "railway/app.py")
    m.QUALITY.clear()
    m.VERDICTS.clear()
    m.ROLL.clear()
    return m


def _client(m):
    return TestClient(m.app, base_url="https://testserver")


def _client_as(m, uid):
    c = TestClient(m.app, base_url="https://testserver")
    c.cookies.set("sonave_session", m.auth.sign_session(uid))
    return c


def test_free_minutes_calculation(mod):
    """free_minutes() returns base 300, and adds bonus_minutes when provided in user/dict/id."""
    # 1. Base / None
    assert mod.billing.free_minutes() == 300
    assert mod.billing.free_minutes(None) == 300

    # 2. User with no bonus
    u = mod.db.upsert_google_user("sub-u1", "u1@example.com", "U1", "", "member")
    assert mod.billing.free_minutes(u["id"]) == 300
    assert mod.billing.free_minutes(u) == 300

    # 3. User with bonus_minutes in metadata
    mod.db.update_user_metadata(u["id"], {"bonus_minutes": 240, "review_bonus_awarded": True})
    assert mod.billing.free_minutes(u["id"]) == 540  # 300 + 240 = 540 min (9 hours)

    # 4. User dict with direct bonus_minutes key
    assert mod.billing.free_minutes({"bonus_minutes": 240}) == 540
    assert mod.billing.free_minutes({"metadata": json.dumps({"bonus_minutes": 120})}) == 420


def test_review_prompt_clicked_awards_bonus(mod):
    """Clicking review prompt via /api/telemetry/event awards +4 hours (240 min) to user profile."""
    u = mod.db.upsert_google_user("sub-rev", "derek@usesonave.com", "Derek", "", "member")
    c = _client_as(mod, u["id"])

    # Before review: base 300 min
    assert mod.billing.free_minutes(u["id"]) == 300
    ent_before = mod.billing.entitlement(u["id"])
    assert ent_before["free_minutes"] == 300
    assert ent_before["bonus_minutes"] == 0

    # Send review_prompt_clicked telemetry
    resp = c.post("/api/telemetry/event", json={
        "kind": "review_prompt_clicked",
        "detail": {"space": "meet-bonus-room", "email": "derek@usesonave.com"}
    })
    assert resp.status_code == 204

    # Verify db user metadata updated
    u_after = mod.db.get_user(u["id"])
    assert u_after is not None
    meta = json.loads(u_after["metadata"])
    assert meta["bonus_minutes"] == 240
    assert meta["review_bonus_awarded"] is True
    assert "reviewed_at" in meta

    # Verify billing entitlement now reflects 540 minutes (9 free hours)
    ent_after = mod.billing.entitlement(u["id"])
    assert ent_after["free_minutes"] == 540
    assert ent_after["bonus_minutes"] == 240
    assert ent_after["remaining_free"] == 540

    # Idempotent click does not double stack
    resp2 = c.post("/api/telemetry/event", json={
        "kind": "review_prompt_clicked",
        "detail": {"space": "meet-bonus-room"}
    })
    assert resp2.status_code == 204
    u_after2 = mod.db.get_user(u["id"])
    meta2 = json.loads(u_after2["metadata"])
    assert meta2["bonus_minutes"] == 240


def test_billing_quota_allows_past_300_minutes_with_bonus(mod):
    """A user who consumed 320 minutes is blocked under base free tier, but allowed with bonus."""
    u = mod.db.upsert_google_user("sub-q", "quota_user@example.com", "Quota User", "", "member")
    mod.db.add_usage_minutes(u["id"], mod.billing.month_key(), 320)

    # 1. Without bonus: 320 > 300 -> blocked
    denied = mod.billing.can_launch_bot(u["id"])
    assert denied is not None
    assert denied["code"] == "quota_exceeded"
    assert "Free 5 monitored hours" in denied["detail"]

    # 2. Award +4 hours (240 min) review bonus -> total 540 min
    mod.db.award_review_bonus(u["id"], 240)
    allowed = mod.billing.can_launch_bot(u["id"])
    assert allowed is None  # Allowed because 320 < 540!

    # Check remaining free minutes
    ent = mod.billing.entitlement(u["id"])
    assert ent["remaining_free"] == 220.0  # 540 - 320 = 220 min remaining

    # 3. Consume beyond 540 min (e.g. 545 min) -> blocked with 9 hours in detail
    mod.db.add_usage_minutes(u["id"], mod.billing.month_key(), 225)  # 320 + 225 = 545
    denied2 = mod.billing.can_launch_bot(u["id"])
    assert denied2 is not None
    assert denied2["code"] == "quota_exceeded"
    assert "Free 9 monitored hours" in denied2["detail"]


def test_format_activity_review_prompt_clicked(mod):
    """_format_activity formats review_prompt_clicked with (+4 Free Hours Unlocked)."""
    msg = mod._format_activity("u1", "review_prompt_clicked", {"email": "derek@usesonave.com"})
    assert "⭐ Marketplace Review Clicked (+4 Free Hours Unlocked): derek@usesonave.com" in msg


def test_api_me_returns_bonus_minutes_and_flag(mod):
    """/api/me endpoint returns bonus_minutes and review_bonus_awarded."""
    u = mod.db.upsert_google_user("sub-me", "me_test@example.com", "Me Test", "", "member")
    mod.db.award_review_bonus(u["id"], 240)
    c = _client_as(mod, u["id"])

    r = c.get("/api/me")
    assert r.status_code == 200
    data = r.json()
    assert data["free_minutes"] == 540
    assert data["bonus_minutes"] == 240
    assert data["review_bonus_awarded"] is True


def test_admin_telemetry_and_rollup_exposes_bonus(mod):
    """Admin users list and user telemetry endpoints expose bonus_minutes."""
    admin = mod.db.upsert_google_user("sub-adm", "boss@sonave.dev", "Boss", "", "admin")
    admin_c = _client_as(mod, admin["id"])

    u = mod.db.upsert_google_user("sub-rev-adm", "reviewed_user@example.com", "Rev User", "", "member")
    mod.db.award_review_bonus(u["id"], 240)

    # 1. Admin users rollup
    r_users = admin_c.get("/api/admin/users")
    assert r_users.status_code == 200
    users = {row["email"]: row for row in r_users.json()["users"]}
    u_row = users["reviewed_user@example.com"]
    assert u_row["bonus_minutes"] == 240
    assert u_row["review_bonus_awarded"] is True

    # 2. Admin user telemetry 360 view
    r_telem = admin_c.get(f"/api/admin/users/{u['id']}/telemetry")
    assert r_telem.status_code == 200
    telem_user = r_telem.json()["user"]
    assert telem_user["bonus_minutes"] == 240
    assert telem_user["review_bonus_awarded"] is True
    assert telem_user["reviewed_at"] is not None


def test_templates_include_review_credit_and_admin_feed_wiring(mod):
    """Templates contain the review bonus elements and admin telemetry formatters."""
    c = _client(mod)

    # meet-addon.html
    r_addon = c.get("/meet-addon")
    assert r_addon.status_code == 200
    addon_html = r_addon.text
    assert "review_prompt_clicked" in addon_html
    assert "reviewCard" in addon_html
    assert "4 bonus free hours unlocked" in addon_html
    assert "onReviewClicked" in addon_html
    assert "review_bonus_awarded" in addon_html

    # console.html
    r_console = c.get("/console")
    assert r_console.status_code == 200
    console_html = r_console.text
    assert "admSummary" in console_html
    assert "session_milestone" in console_html
    assert "review_prompt_clicked" in console_html
    assert "web_cta_click" in console_html
    assert "support_inquiry" in console_html
    assert "diagnostic_run" in console_html
    assert "utBonusBadge" in console_html
    assert "utBonusHours" in console_html
    assert "setAdmKind('review_prompt_clicked',this)" in console_html


def test_award_review_bonus_with_email_address(mod):
    """award_review_bonus resolves user when passed email directly (e.g. derek@usesonave.com)."""
    u = mod.db.upsert_google_user("sub-derek", "derek@usesonave.com", "Derek Gallardo", "", "admin")
    res = mod.db.award_review_bonus("derek@usesonave.com", 240)
    assert res is not None
    assert res["id"] == u["id"]
    meta = json.loads(res["metadata"])
    assert meta["bonus_minutes"] == 240
    assert meta["review_bonus_awarded"] is True
    assert "reviewed_at" in meta


def test_billing_entitlement_and_free_minutes_by_email(mod):
    """billing.free_minutes and billing.entitlement accept user email address directly."""
    u = mod.db.upsert_google_user("sub-email-lookup", "derek@usesonave.com", "Derek", "", "member")
    mod.db.award_review_bonus(u["id"], 240)

    assert mod.billing.free_minutes("derek@usesonave.com") == 540
    ent = mod.billing.entitlement("derek@usesonave.com")
    assert ent["free_minutes"] == 540
    assert ent["bonus_minutes"] == 240


def test_anonymous_telemetry_event_provisions_user_and_merges_on_login(mod):
    """Anonymous review click with email provisions bonus and merges cleanly into Google login."""
    c = _client(mod)

    # 1. Anonymous user clicks review prompt with email
    resp = c.post("/api/telemetry/event", json={
        "kind": "review_prompt_clicked",
        "detail": {"email": "derek@usesonave.com", "space": "meet-founder-room"}
    })
    assert resp.status_code == 204

    # Pre-provisioned user has bonus
    u = mod.db.get_user_by_email("derek@usesonave.com")
    assert u is not None
    meta = json.loads(u["metadata"])
    assert meta["bonus_minutes"] == 240
    assert meta["review_bonus_awarded"] is True
    first_ts = meta["reviewed_at"]

    # 2. User later completes Google OAuth with their real Google sub
    u_merged = mod.db.upsert_google_user("google-sub-derek-123", "derek@usesonave.com", "Derek Gallardo", "", "admin")
    assert u_merged["id"] == u["id"]  # Same account preserved!
    assert u_merged["google_sub"] == "google-sub-derek-123"
    meta_merged = json.loads(u_merged["metadata"])
    assert meta_merged["bonus_minutes"] == 240
    assert meta_merged["review_bonus_awarded"] is True
    assert meta_merged["reviewed_at"] == first_ts


def test_reviewed_at_timestamp_preserved_on_idempotent_click(mod):
    """Original reviewed_at timestamp is preserved when user clicks review link multiple times."""
    u = mod.db.upsert_google_user("sub-ts", "timestamp_user@example.com", "TS User", "", "member")
    res1 = mod.db.award_review_bonus(u["id"], 240)
    meta1 = json.loads(res1["metadata"])
    orig_ts = meta1["reviewed_at"]
    assert orig_ts > 0

    time.sleep(0.05)
    res2 = mod.db.award_review_bonus(u["id"], 240)
    meta2 = json.loads(res2["metadata"])
    assert meta2["reviewed_at"] == orig_ts  # Not overwritten!


def test_admin_outreach_endpoint(mod):
    """Admin can dispatch direct outreach emails to target users."""
    admin = mod.db.upsert_google_user("sub-adm-outreach", "boss2@sonave.dev", "Boss", "", "admin")
    admin_c = _client_as(mod, admin["id"])
    target = mod.db.upsert_google_user("sub-target", "client@example.com", "Client Name", "", "member")

    payload = {
        "to_email": "client@example.com",
        "subject": "Sonave Voice Security Follow-up",
        "text": "Hi Client, thanks for using Sonave.",
        "from_name": "Derek Gallardo",
        "from_email": "derek@usesonave.com"
    }
    r = admin_c.post("/api/admin/outreach", json=payload)
    assert r.status_code == 200
    assert r.json()["ok"] is True

    # Check that event was recorded in audit stream
    events = mod.db.list_events(user_id=target["id"], kind="email_sent")
    assert len(events) >= 1
    assert any("founder_outreach" in ev.get("detail", "") for ev in events)


