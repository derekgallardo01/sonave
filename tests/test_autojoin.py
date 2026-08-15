"""Zero-scope calendar auto-join: ICS parsing and the deploy tick."""
import io
import json
import time

import pytest

import conftest

TOKEN = "machine-token-123"


@pytest.fixture
def mod(monkeypatch, tmp_path):
    monkeypatch.setenv("SONAVE_SCORER_URL", "")
    monkeypatch.setenv("SONAVE_RECALL_API_KEY", "test-key")
    monkeypatch.setenv("SONAVE_API_TOKEN", TOKEN)
    monkeypatch.setenv("SONAVE_SESSION_SECRET", "sec")
    monkeypatch.setenv("SONAVE_PUBLIC_DOMAIN", "test.sonave.dev")
    return conftest.load_module("rwapp_autojoin", "railway/app.py")


def _ics(body: str) -> str:
    return "BEGIN:VCALENDAR\r\n" + body.replace("\n", "\r\n") + "\r\nEND:VCALENDAR"


# fixed reference time: 2026-08-14 15:00:00 UTC (a Friday)
NOW = 1786719600.0
AJ = None


def _aj(mod):
    return mod.autojoin


def test_parse_concrete_event_with_meet_link(mod):
    aj = _aj(mod)
    text = _ics("""BEGIN:VEVENT
UID:abc123@google.com
DTSTART:20260814T150500Z
X-GOOGLE-CONFERENCE:https://meet.google.com/abc-defg-hij
END:VEVENT""")
    ev = aj.parse_ics(text, now=NOW)
    assert len(ev) == 1
    assert ev[0]["meet_url"] == "https://meet.google.com/abc-defg-hij"
    assert ev[0]["start_ts"] == NOW + 300


def test_parse_skips_cancelled_all_day_and_linkless(mod):
    aj = _aj(mod)
    text = _ics("""BEGIN:VEVENT
UID:c1
DTSTART:20260814T150500Z
STATUS:CANCELLED
X-GOOGLE-CONFERENCE:https://meet.google.com/aaa-aaaa-aaa
END:VEVENT
BEGIN:VEVENT
UID:c2
DTSTART;VALUE=DATE:20260814
X-GOOGLE-CONFERENCE:https://meet.google.com/bbb-bbbb-bbb
END:VEVENT
BEGIN:VEVENT
UID:c3
DTSTART:20260814T150500Z
SUMMARY:no conference link here
END:VEVENT""")
    assert aj.parse_ics(text, now=NOW) == []


def test_parse_folded_lines_and_description_link(mod):
    aj = _aj(mod)
    # ICS folds long lines with "\r\n " — the link is split mid-URL
    text = ("BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\nUID:f1\r\n"
            "DTSTART:20260814T150000Z\r\n"
            "DESCRIPTION:join here https://meet.google.com/ccc\r\n -cccc-ccc today\r\n"
            "END:VEVENT\r\nEND:VCALENDAR")
    ev = aj.parse_ics(text, now=NOW)
    assert len(ev) == 1 and ev[0]["meet_url"] == "https://meet.google.com/ccc-cccc-ccc"


def test_weekly_rrule_hits_today_and_respects_exdate(mod):
    aj = _aj(mod)
    # 2026-08-14 is a Friday; series started Fri 2026-08-07 at 15:05 UTC
    base = """BEGIN:VEVENT
UID:w1
DTSTART:20260807T150500Z
RRULE:FREQ=WEEKLY;BYDAY=FR
X-GOOGLE-CONFERENCE:https://meet.google.com/ddd-dddd-ddd
END:VEVENT"""
    ev = aj.parse_ics(_ics(base), now=NOW)
    assert any(e["start_ts"] == NOW + 300 for e in ev)      # today's occurrence
    cancelled = base.replace("RRULE:FREQ=WEEKLY;BYDAY=FR",
                             "RRULE:FREQ=WEEKLY;BYDAY=FR\nEXDATE:20260814T150500Z")
    ev2 = aj.parse_ics(_ics(cancelled), now=NOW)
    assert not any(e["start_ts"] == NOW + 300 for e in ev2)  # this instance skipped


def test_daily_rrule_until_expired(mod):
    aj = _aj(mod)
    text = _ics("""BEGIN:VEVENT
UID:d1
DTSTART:20260801T150500Z
RRULE:FREQ=DAILY;UNTIL=20260810T000000Z
X-GOOGLE-CONFERENCE:https://meet.google.com/eee-eeee-eee
END:VEVENT""")
    assert aj.parse_ics(text, now=NOW) == []                 # series already over


def test_zoom_and_teams_links_extracted(mod):
    aj = _aj(mod)
    text = _ics("""BEGIN:VEVENT
UID:z1
DTSTART:20260814T150500Z
LOCATION:https://us05web.zoom.us/j/85512345678?pwd=abC12.9
END:VEVENT
BEGIN:VEVENT
UID:t1
DTSTART:20260814T151000Z
DESCRIPTION:join: https://teams.microsoft.com/l/meetup-join/19%3ameeting_abc%40thread.v2/0?context=%7b%22Tid%22%3a%22x%22%7d
END:VEVENT""")
    ev = {e["uid"]: e["meet_url"] for e in aj.parse_ics(text, now=NOW)}
    assert ev["z1"] == "https://us05web.zoom.us/j/85512345678?pwd=abC12.9"
    assert ev["t1"].startswith("https://teams.microsoft.com/l/meetup-join/")


def test_calendar_api_zoom_in_location(mod, monkeypatch):
    import io as _io
    import json as _json

    class _R(_io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    items = {"items": [{"id": "z-ev", "status": "confirmed",
                        "location": "Zoom: https://zoom.us/j/99911122233",
                        "start": {"dateTime": "2026-08-14T15:00:30+00:00"}}]}
    monkeypatch.setattr(mod.autojoin.urllib.request, "urlopen",
                        lambda req, timeout=None: _R(_json.dumps(items).encode()))
    ev = mod.autojoin.google_calendar_events("at", now=NOW)
    assert len(ev) == 1 and ev[0]["meet_url"] == "https://zoom.us/j/99911122233"


def test_due_events_window(mod):
    aj = _aj(mod)
    evs = [{"uid": "a", "start_ts": NOW + 30, "meet_url": "m"},
           {"uid": "b", "start_ts": NOW + 3600, "meet_url": "m"},
           {"uid": "c", "start_ts": NOW - 300, "meet_url": "m"},
           {"uid": "d", "start_ts": NOW - 3000, "meet_url": "m"}]
    due = {e["uid"] for e in aj.due_events(evs, now=NOW)}
    assert due == {"a", "c"}      # 30 s early and 5 min late join; others don't


def test_tick_launches_once_and_logs(mod, monkeypatch):
    u = mod.db.upsert_google_user("s-aj", "aj@x.com", "AJ", "", "member")
    mod.db.set_ical_url(u["id"], "https://calendar.google.com/calendar/ical/x/basic.ics")
    ics = _ics("""BEGIN:VEVENT
UID:tick1
DTSTART:20260814T150000Z
X-GOOGLE-CONFERENCE:https://meet.google.com/fff-ffff-fff
END:VEVENT""")
    monkeypatch.setattr(mod.autojoin, "fetch_ics", lambda url, timeout=12: ics)
    calls = {"n": 0}

    class _R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    def _open(req, timeout=None):
        calls["n"] += 1
        return _R(json.dumps({"id": f"aj-bot-{calls['n']}"}).encode())

    monkeypatch.setattr(mod.urllib.request, "urlopen", _open)
    assert mod._autojoin_tick(now=NOW) == 1                  # deploys into the meeting
    assert calls["n"] == 1
    assert mod._autojoin_tick(now=NOW) == 0                  # logged — never twice
    assert calls["n"] == 1
    row = mod.db.find_active_bot(u["id"], "https://meet.google.com/fff-ffff-fff")
    assert row and row["bot_id"] == "aj-bot-1"
