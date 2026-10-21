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
    monkeypatch.delenv("SONAVE_FREE_MINUTES", raising=False)
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
    """free_minutes() returns base 45, and adds bonus_minutes when provided in user/dict/id."""
    # 1. Base / None
    assert mod.billing.free_minutes() == 45
    assert mod.billing.free_minutes(None) == 45

    # 2. User with no bonus
    u = mod.db.upsert_google_user("sub-u1", "u1@example.com", "U1", "", "member")
    assert mod.billing.free_minutes(u["id"]) == 45
    assert mod.billing.free_minutes(u) == 45

    # 3. User with bonus_minutes in metadata
    mod.db.update_user_metadata(u["id"], {"bonus_minutes": 240, "review_bonus_awarded": True})
    assert mod.billing.free_minutes(u["id"]) == 285  # 45 + 240 = 285 min

    # 4. User dict with direct bonus_minutes key
    assert mod.billing.free_minutes({"bonus_minutes": 240}) == 285
    assert mod.billing.free_minutes({"metadata": json.dumps({"bonus_minutes": 120})}) == 165


def test_review_prompt_clicked_awards_bonus(mod):
    """Clicking review prompt via /api/telemetry/event awards +4 hours (240 min) to user profile."""
    u = mod.db.upsert_google_user("sub-rev", "derek@usesonave.com", "Derek", "", "member")
    c = _client_as(mod, u["id"])

    # Before review: base 45 min
    assert mod.billing.free_minutes(u["id"]) == 45
    ent_before = mod.billing.entitlement(u["id"])
    assert ent_before["free_minutes"] == 45
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

    # Verify billing entitlement now reflects 285 minutes
    ent_after = mod.billing.entitlement(u["id"])
    assert ent_after["free_minutes"] == 285
    assert ent_after["bonus_minutes"] == 240
    assert ent_after["remaining_free"] == 285

    # Idempotent click does not double stack
    resp2 = c.post("/api/telemetry/event", json={
        "kind": "review_prompt_clicked",
        "detail": {"space": "meet-bonus-room"}
    })
    assert resp2.status_code == 204
    u_after2 = mod.db.get_user(u["id"])
    meta2 = json.loads(u_after2["metadata"])
    assert meta2["bonus_minutes"] == 240


def test_billing_quota_allows_past_120_minutes_with_bonus(mod):
    """A user who consumed 150 minutes is blocked under base free tier, but allowed with bonus."""
    u = mod.db.upsert_google_user("sub-q", "quota_user@example.com", "Quota User", "", "member")
    mod.db.add_usage_minutes(u["id"], mod.billing.month_key(), 150)

    # 1. Without bonus: 150 > 45 -> blocked
    denied = mod.billing.can_launch_bot(u["id"])
    assert denied is not None
    assert denied["code"] == "quota_exceeded"
    assert "Free 45 monitored minutes" in denied["detail"]

    # 2. Award +4 hours (240 min) review bonus -> total 285 min
    mod.db.award_review_bonus(u["id"], 240)
    allowed = mod.billing.can_launch_bot(u["id"])
    assert allowed is None  # Allowed because 150 < 285!

    # Check remaining free minutes
    ent = mod.billing.entitlement(u["id"])
    assert ent["remaining_free"] == 135.0  # 285 - 150 = 135 min remaining

    # 3. Consume beyond 285 min (e.g. 290 min) -> blocked with 4 monitored hours in detail
    mod.db.add_usage_minutes(u["id"], mod.billing.month_key(), 140)  # 150 + 140 = 290
    denied2 = mod.billing.can_launch_bot(u["id"])
    assert denied2 is not None
    assert denied2["code"] == "quota_exceeded"
    assert "Free 4 monitored hours" in denied2["detail"]


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
    assert data["free_minutes"] == 285
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

    assert mod.billing.free_minutes("derek@usesonave.com") == 285
    ent = mod.billing.entitlement("derek@usesonave.com")
    assert ent["free_minutes"] == 285
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


def test_default_free_quota_45_without_env_var(monkeypatch):
    """billing.free_minutes defaults to 45 minutes when SONAVE_FREE_MINUTES is unset."""
    monkeypatch.delenv("SONAVE_FREE_MINUTES", raising=False)
    import billing
    assert billing.free_minutes() == 45
    assert billing.free_minutes(None) == 45
    # Custom env var override works
    monkeypatch.setenv("SONAVE_FREE_MINUTES", "240")
    assert billing.free_minutes() == 240


def test_post_meeting_review_email_triggered_on_disconnect(mod):
    """Post-meeting review email is triggered when meeting disconnects for active call with unreviewed user."""
    u = mod.db.upsert_google_user("sub-post-email", "active_host@example.com", "Active Host", "", "member")
    c = _client_as(mod, u["id"])

    # 1. User has an active call milestone (e.g. 1 min milestone)
    resp = c.post("/api/telemetry/event", json={
        "kind": "session_milestone",
        "detail": {"space": "meet-growth-room", "minutes": 1, "elapsed_sec": 65}
    })
    assert resp.status_code == 204

    # User has not reviewed yet and has not received post-call email
    meta_before = json.loads(mod.db.get_user(u["id"])["metadata"] or "{}")
    assert not meta_before.get("review_bonus_awarded")
    assert not meta_before.get("post_call_email_sent")

    # 2. Meeting ends (session disconnect)
    r_disc = c.post("/api/meet/sessions/disconnect", json={"space_id": "meet-growth-room"})
    assert r_disc.status_code == 200

    # 3. User metadata is updated with post_call_email_sent: True
    u_after = mod.db.get_user(u["id"])
    meta_after = json.loads(u_after["metadata"])
    assert meta_after.get("post_call_email_sent") is True

    # 4. email_sent event is logged with type: "post_meeting_review"
    events = mod.db.list_events(user_id=u["id"], kind="email_sent")
    assert len(events) >= 1
    post_events = [ev for ev in events if '"post_meeting_review"' in ev.get("detail", "")]
    assert len(post_events) == 1
    post_ev_detail = json.loads(post_events[0]["detail"])
    assert post_ev_detail["type"] == "post_meeting_review"
    assert post_ev_detail["to"] == "active_host@example.com"


def test_post_meeting_review_email_skipped_if_already_reviewed(mod):
    """Users who have already claimed review bonus do not receive post-meeting review email."""
    u = mod.db.upsert_google_user("sub-already-reviewed", "reviewed_host@example.com", "Rev Host", "", "member")
    mod.db.award_review_bonus(u["id"], 240)
    c = _client_as(mod, u["id"])

    # Milestone reached
    c.post("/api/telemetry/event", json={
        "kind": "session_milestone",
        "detail": {"space": "meet-growth-room-2", "minutes": 5}
    })

    # Meeting disconnects
    c.post("/api/meet/sessions/disconnect", json={"space_id": "meet-growth-room-2"})

    meta = json.loads(mod.db.get_user(u["id"])["metadata"])
    assert not meta.get("post_call_email_sent")
    events = mod.db.list_events(user_id=u["id"], kind="email_sent")
    assert not any("post_meeting_review" in ev.get("detail", "") for ev in events)


def test_post_meeting_review_email_skipped_if_already_sent(mod):
    """If user already received post_call_email_sent, it is not sent again."""
    u = mod.db.upsert_google_user("sub-already-sent", "sent_host@example.com", "Sent Host", "", "member")
    mod.db.update_user_metadata(u["id"], {"post_call_email_sent": True})
    c = _client_as(mod, u["id"])

    # Milestone reached
    c.post("/api/telemetry/event", json={
        "kind": "session_milestone",
        "detail": {"space": "meet-growth-room-3", "minutes": 1}
    })

    # Meeting disconnects
    c.post("/api/meet/sessions/disconnect", json={"space_id": "meet-growth-room-3"})

    events = mod.db.list_events(user_id=u["id"], kind="email_sent")
    assert not any("post_meeting_review" in ev.get("detail", "") for ev in events)


def test_post_meeting_review_email_skipped_if_no_active_call(mod):
    """A user with no active call time (< 1 min, no milestone) does not get post-meeting email."""
    u = mod.db.upsert_google_user("sub-quick-drop", "quick_drop@example.com", "Quick Drop", "", "member")
    c = _client_as(mod, u["id"])

    # Connects and immediately disconnects (< 1 min)
    mod.MEET_SESSION_START[u["id"]] = time.time() - 5.0  # only 5 seconds ago
    c.post("/api/meet/sessions/disconnect", json={"space_id": "meet-quick-drop"})

    meta = json.loads(mod.db.get_user(u["id"])["metadata"] or "{}")
    assert not meta.get("post_call_email_sent")
    events = mod.db.list_events(user_id=u["id"], kind="email_sent")
    assert not any("post_meeting_review" in ev.get("detail", "") for ev in events)


def test_soundcheck_seeds_98_9_percent_real_host(mod):
    """/api/audio/soundcheck returns 98.9% Real baseline and seeds QUALITY and VERDICTS."""
    u = mod.db.upsert_google_user("sub-sc", "host_speaker@example.com", "Host Speaker", "", "member")
    c = _client_as(mod, u["id"])

    resp = c.post("/api/audio/soundcheck", json={"speaker_name": "Host Speaker", "duration_sec": 2.5})
    assert resp.status_code == 200
    d = resp.json()
    assert d["ok"] is True
    assert d["authenticity_pct"] == 98.9
    assert d["p_fake"] == 0.011
    assert d["verdict"] == "real"


def test_meet_addon_template_has_auto_verify_and_listening_pulse(mod):
    """meet-addon.html has active dynamic audio meter, listening pulse, and auto-verify baseline logic."""
    c = _client(mod)
    r = c.get("/meet-addon")
    assert r.status_code == 200
    html = r.text
    assert "emptyAudioMeter" in html
    assert "meterPulseText" in html
    assert "Listening for speech" in html
    assert "handleLiveMicAudioFrame" in html
    assert "triggerHostAutoVerify" in html
    assert "Verified Authentic Human (98.9% Real)" in html
    assert "ACOUSTIC BASELINE CALIBRATED" in html


def test_post_meeting_review_email_skipped_when_historical_event_exists(mod):
    """Historical meet_connected events from previous sessions must not cause short drops to trigger emails."""
    u = mod.db.upsert_google_user("sub-old-hist", "old_hist@example.com", "Old Hist", "", "member")
    c = _client_as(mod, u["id"])

    # Simulate an old connection event from 2 hours ago
    mod.db.add_event(u["id"], "meet_connected", json.dumps({"space": "yesterdays-call"}))
    with mod.db._LOCK:
        conn = mod.db._conn()
        conn.execute("UPDATE events SET ts=? WHERE user_id=? AND kind='meet_connected'", (time.time() - 7200, u["id"]))
        conn.commit()
        conn.close()

    # User connects today and disconnects after only 4 seconds
    mod.MEET_SESSION_START[u["id"]] = time.time() - 4.0
    c.post("/api/meet/sessions/disconnect", json={"space_id": "meet-quick-drop-2"})

    meta = json.loads(mod.db.get_user(u["id"])["metadata"] or "{}")
    assert not meta.get("post_call_email_sent")
    events = mod.db.list_events(user_id=u["id"], kind="email_sent")
    assert not any("post_meeting_review" in ev.get("detail", "") for ev in events)


def test_post_meeting_review_email_triggered_by_meeting_ended(mod):
    """Bot-monitored meeting_ended event with duration >= 60s triggers review email."""
    u = mod.db.upsert_google_user("sub-bot-ended", "bot_host@example.com", "Bot Host", "", "member")

    # Simulate meeting_ended event with 120s duration
    mod._track(u["id"], "meeting_ended", bot_id="bot-test-1", duration_sec=120.0, metered_min=2.0)

    meta = json.loads(mod.db.get_user(u["id"])["metadata"])
    assert meta.get("post_call_email_sent") is True
    events = mod.db.list_events(user_id=u["id"], kind="email_sent")
    assert any("post_meeting_review" in ev.get("detail", "") for ev in events)


def test_post_meeting_review_email_supports_user_lookup_by_email(mod):
    """_maybe_send_post_meeting_review_email resolves user by email if passed an email string."""
    u = mod.db.upsert_google_user("sub-by-email", "by_email@example.com", "By Email", "", "member")

    sent = mod._maybe_send_post_meeting_review_email("by_email@example.com", call_duration_sec=75.0)
    assert sent is True

    meta = json.loads(mod.db.get_user(u["id"])["metadata"])
    assert meta.get("post_call_email_sent") is True


def test_api_quality_exposes_baseline_and_model_flags(mod):
    """api_quality exposes baseline and model flags so the Meet Add-on can render the baseline badge."""
    u = mod.db.upsert_google_user("sub-sc-qual", "qual_host@example.com", "Qual Host", "", "member")
    c = _client_as(mod, u["id"])

    # Perform soundcheck to seed baseline
    c.post("/api/audio/soundcheck", json={"speaker_name": "Qual Host", "duration_sec": 2.5})

    r = c.get("/api/quality")
    assert r.status_code == 200
    data = r.json()
    spk_key = "Qual Host (Verified Voice)"
    assert spk_key in data
    row = data[spk_key]
    assert row.get("baseline") is True
    assert row.get("model") == "soundcheck"
    assert row.get("auth_verdict") == "real"




