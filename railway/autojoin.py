"""Zero-scope calendar auto-join: poll a user's secret iCal URL and deploy the
bot into Google Meet events as they start. No Google OAuth scopes involved —
the user pastes their calendar's "Secret address in iCal format" in the console.

Deliberately small ICS support, matching what Google Calendar exports:
- concrete timed events (DTSTART with Z or TZID; all-day events are skipped)
- recurring DAILY/WEEKLY RRULEs with INTERVAL / BYDAY / UNTIL, plus EXDATEs
- STATUS:CANCELLED events skipped
Limitations (documented, acceptable for meeting auto-join): COUNT-limited
series are treated as unbounded (UNTIL and the join log still bound behavior);
MONTHLY/YEARLY rules are ignored.
"""
from __future__ import annotations

import re
import time
import urllib.request
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

MEET_RE = re.compile(r"https://meet\.google\.com/[a-z0-9\-]+")
_DAYS = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}


def fetch_ics(url: str, timeout: float = 12) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Sonave-AutoJoin/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _unfold(text: str) -> str:
    """ICS folds long lines with a leading space/tab on the continuation."""
    return text.replace("\r\n", "\n").replace("\n ", "").replace("\n\t", "")


def _parse_dt(val: str, tzid: str | None) -> datetime | None:
    """ICS datetime -> aware datetime. Date-only (all-day) returns None."""
    val = val.strip()
    m = re.fullmatch(r"(\d{8})T(\d{6})(Z?)", val)
    if not m:
        return None
    naive = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    if m.group(3) == "Z":
        return naive.replace(tzinfo=timezone.utc)
    if tzid and ZoneInfo is not None:
        try:
            return naive.replace(tzinfo=ZoneInfo(tzid))
        except Exception:
            return None          # unknown zone: skipping beats joining at the wrong hour
    return naive.replace(tzinfo=timezone.utc)    # floating time: assume UTC


def _rrule_occurrences(start: datetime, rrule: str, now: float,
                       horizon_sec: float) -> list[float]:
    parts = dict(p.split("=", 1) for p in rrule.split(";") if "=" in p)
    freq = parts.get("FREQ")
    if freq not in ("DAILY", "WEEKLY"):
        return []
    interval = max(1, int(parts.get("INTERVAL") or 1))
    until = None
    if parts.get("UNTIL"):
        u = _parse_dt(parts["UNTIL"], None)
        until = u.timestamp() if u else None
    bydays = {_DAYS[d] for d in (parts.get("BYDAY") or "").split(",") if d in _DAYS}
    if freq == "WEEKLY" and not bydays:
        bydays = {start.weekday()}

    out = []
    day0 = datetime.fromtimestamp(now, tz=start.tzinfo).date()
    for d in range(-1, int(horizon_sec // 86400) + 2):
        day = day0 + timedelta(days=d)
        delta_days = (day - start.date()).days
        if delta_days < 0:
            continue
        if freq == "DAILY" and delta_days % interval:
            continue
        if freq == "WEEKLY":
            if day.weekday() not in bydays:
                continue
            if (delta_days // 7) % interval:     # approximate at week boundaries
                continue
        # tz-aware combine: ZoneInfo resolves DST for the candidate date itself
        ts = datetime.combine(day, start.timetz()).timestamp()
        if until is not None and ts > until:
            continue
        out.append(ts)
    return out


def parse_ics(text: str, now: float | None = None,
              horizon_sec: float = 48 * 3600) -> list[dict]:
    """All joinable occurrences within the horizon: {uid, start_ts, meet_url}."""
    now = time.time() if now is None else now
    events = []
    for block in re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", _unfold(text), re.S):
        if "STATUS:CANCELLED" in block:
            continue
        murl = MEET_RE.search(block)
        if not murl:
            continue
        uid_m = re.search(r"^UID:(.+)$", block, re.M)
        uid = (uid_m.group(1).strip() if uid_m else murl.group(0))[:120]
        ds = re.search(r"^DTSTART(?:;TZID=([^:;]+))?:([^\n]+)$", block, re.M)
        if not ds:
            continue
        start = _parse_dt(ds.group(2), ds.group(1))
        if start is None:
            continue
        exdates = set()
        for em in re.finditer(r"^EXDATE(?:;TZID=([^:;]+))?:([^\n]+)$", block, re.M):
            for v in em.group(2).split(","):
                dt = _parse_dt(v, em.group(1))
                if dt:
                    exdates.add(int(dt.timestamp()) // 60)
        rr = re.search(r"^RRULE:(.+)$", block, re.M)
        starts = (_rrule_occurrences(start, rr.group(1), now, horizon_sec)
                  if rr else [start.timestamp()])
        for ts in starts:
            if now - 600 <= ts <= now + horizon_sec and (int(ts) // 60) not in exdates:
                events.append({"uid": uid, "start_ts": ts, "meet_url": murl.group(0)})
    return events


def due_events(events: list[dict], now: float | None = None,
               early_sec: float = 60, late_sec: float = 600) -> list[dict]:
    """Occurrences worth joining right now: up to 1 min early, 10 min late —
    late enough to catch a poll that lands just after the hour, early enough
    that the bot is in the waiting room when people arrive."""
    now = time.time() if now is None else now
    return [e for e in events
            if e["start_ts"] - early_sec <= now <= e["start_ts"] + late_sec]
