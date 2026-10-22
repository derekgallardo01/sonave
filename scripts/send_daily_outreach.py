#!/usr/bin/env python3
"""
Sonave — Automated Daily Single-Touch Backlink Outreach Sender (Resend API)
===========================================================================
- Reads `docs/backlinks/SONAVE_OUTREACH_TRACKER_2026.csv`
- Strictly deduplicates by eTLD+1 root domain, normalized URL, and recipient email against:
    - All rows in `SONAVE_OUTREACH_TRACKER_2026.csv` with `Outreach_Status` not starting with `Not Contacted`
    - `docs/backlinks/outreach-dispatch-log.json`
    - `I:\Projects\sealed-rose\SEALED-ROSE-LINK-TRACKER.csv` & Sealed Rose dispatch logs (so cybersecurity/deepfake
      editors contacted by Sealed Rose are NEVER contacted twice)
- Sends up to `--limit` (default 8) personalized single-touch emails/day from:
    Derek Gallardo <derek@usesonave.com>
- Immediately updates `Outreach_Status` -> `Contacted (YYYY-MM-DD)` in `SONAVE_OUTREACH_TRACKER_2026.csv`
  and appends Resend receipt IDs to `docs/backlinks/outreach-dispatch-log.json`.

Usage:
  python scripts/send_daily_outreach.py --dry-run --limit 8
  python scripts/send_daily_outreach.py --send --limit 8
"""

import argparse
import csv
import datetime as dt
import json
import os
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKLINKS_DIR = REPO_ROOT / "docs" / "backlinks"
TRACKER_CSV = BACKLINKS_DIR / "SONAVE_OUTREACH_TRACKER_2026.csv"
DISPATCH_LOG = BACKLINKS_DIR / "outreach-dispatch-log.json"
SEALED_ROSE_TRACKER = Path(r"I:\Projects\sealed-rose\SEALED-ROSE-LINK-TRACKER.csv")

FROM_EMAIL = "Derek Gallardo <derek@usesonave.com>"
REPLY_TO = "derek@usesonave.com"

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,10}")
BAD_EMAIL_SUBS = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".js", ".css",
    "sentry", "example.com", "domain.com", "email.com", "yoursite",
    "wixpress", "cloudflare", "wordpress", "noreply", "no-reply",
    "donotreply", "privacy@", "legal@", "dmca@", "abuse@", "jobs@", "careers@",
)
MULTI_PART_TLDS = {
    "co.uk", "org.uk", "ac.uk", "com.au", "org.au", "edu.au", "com.br", "edu.br",
    "co.jp", "co.in", "com.mx", "com.sg", "co.za", "co.nz", "com.tr", "substack.com",
}


def extract_root_domain(val: str) -> str:
    if not val:
        return ""
    s = str(val).strip().lower().strip("<>\"' ")
    if "://" in s:
        s = urllib.parse.urlparse(s).hostname or ""
    elif "@" in s:
        s = s.split("@")[-1]
    else:
        s = s.split("/")[0]
    s = s.split(":")[0]
    s = re.sub(r"^www\d*\.", "", s).strip("<>\"' .")
    parts = [p for p in s.split(".") if p]
    if len(parts) <= 2:
        return ".".join(parts)
    last_two = ".".join(parts[-2:])
    if last_two in MULTI_PART_TLDS and len(parts) >= 3:
        return ".".join(parts[-3:])
    return last_two


def extract_email(val: str) -> str:
    if not val:
        return ""
    m = EMAIL_RE.search(str(val).strip())
    if not m:
        return ""
    return m.group(0).lower().strip(".")


def normalize_url(u: str) -> str:
    if not u:
        return ""
    s = str(u).strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\d*\.", "", s)
    s = s.split("#", 1)[0]
    return s.rstrip("/")


def resolve_resend_api_key() -> str:
    if os.environ.get("RESEND_API_KEY"):
        return os.environ["RESEND_API_KEY"].strip()
    candidates = [
        REPO_ROOT / ".env",
        Path(r"I:\Projects\sealed-rose\.env"),
        Path(r"I:\Projects\music-genie\backend\.env"),
    ]
    for p in candidates:
        if p.exists():
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line.startswith("RESEND_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def discover_email_for_domain(domain: str, article_url: str) -> str:
    root = extract_root_domain(domain or article_url)
    if not root:
        return ""
    for u in (article_url, f"https://{root}/contact", f"https://www.{root}/contact", f"https://{root}/about", f"https://{root}/"):
        if not u:
            continue
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 Chrome/131.0.0.0 Safari/537.36"})
            with urllib.request.urlopen(req, timeout=5, context=SSL_CTX) as resp:
                html = resp.read(200_000).decode("utf-8", errors="ignore")
            for m in EMAIL_RE.findall(html):
                el = m.lower().strip(".")
                if any(b in el for b in BAD_EMAIL_SUBS):
                    continue
                if extract_root_domain(el) == root:
                    return el
        except Exception:
            continue
    return ""


def send_via_resend(api_key: str, to_email: str, subject: str, text_body: str) -> str:
    payload = json.dumps({
        "from": FROM_EMAIL,
        "to": [to_email],
        "reply_to": REPLY_TO,
        "subject": subject,
        "text": text_body,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Sonave-Outreach/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("id", "")


def _persist_state(dispatch_history: list[dict], fields: list[str], rows: list[dict]) -> None:
    DISPATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    tmp_log = DISPATCH_LOG.with_suffix(".json.tmp")
    tmp_log.write_text(json.dumps(dispatch_history, indent=2), encoding="utf-8")
    os.replace(tmp_log, DISPATCH_LOG)

    TRACKER_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = TRACKER_CSV.with_suffix(".csv.tmp")
    with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp_csv, TRACKER_CSV)


def main():
    parser = argparse.ArgumentParser(description="Sonave Daily Outreach Sender")
    parser.add_argument("--send", action="store_true", help="Send live emails via Resend (default is dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="Preview batch without sending")
    parser.add_argument("--limit", type=int, default=8, help="Max emails to send today (default: 8)")
    parser.add_argument("--force", action="store_true", help="Override same-day limit guard")
    args = parser.parse_args()

    is_live = args.send and not args.dry_run
    today = dt.date.today().isoformat()

    if not TRACKER_CSV.exists():
        print(f"[ERROR] Tracker CSV not found: {TRACKER_CSV}", file=sys.stderr)
        sys.exit(1)

    dispatch_history = []
    if DISPATCH_LOG.exists():
        try:
            loaded = json.loads(DISPATCH_LOG.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                dispatch_history = loaded
        except Exception:
            dispatch_history = []

    sent_today = [d for d in dispatch_history if str(d.get("sentAt", "")).startswith(today)]
    if is_live and not args.force and len(sent_today) >= args.limit:
        print(f"[SKIP] Already dispatched {len(sent_today)} Sonave emails today ({today}).")
        return

    remaining_quota = args.limit if args.force else max(0, args.limit - len(sent_today)) if is_live else max(0, args.limit)

    contacted_domains = {extract_root_domain(d.get("domain", "")) for d in dispatch_history if d.get("domain")}
    contacted_urls = {normalize_url(d.get("url", "")) for d in dispatch_history if d.get("url")}
    contacted_emails = {extract_email(d.get("to", "")) for d in dispatch_history if extract_email(d.get("to", ""))}

    # Cross-check Sealed Rose tracker so cybersecurity/deepfake editors are never double-contacted
    if SEALED_ROSE_TRACKER.exists():
        with open(SEALED_ROSE_TRACKER, "r", encoding="utf-8-sig", errors="ignore") as f:
            for sr in csv.DictReader(f):
                pu = (sr.get("Platform URL") or "").strip()
                pl = (sr.get("Platform") or "").strip()
                em_sr = extract_email(sr.get("Contact Email") or sr.get("Email") or sr.get("Primary_Email") or "")
                if pu:
                    contacted_domains.add(extract_root_domain(pu))
                    contacted_urls.add(normalize_url(pu))
                if "." in pl:
                    contacted_domains.add(extract_root_domain(pl))
                if em_sr:
                    contacted_emails.add(em_sr)

    with open(TRACKER_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    for r in rows:
        st = (r.get("Outreach_Status") or "").strip().lower()
        if st and not st.startswith("not contacted"):
            url = (r.get("Referring_Page_URL") or "").strip()
            em = extract_email(r.get("Primary_Email") or "")
            if url:
                contacted_domains.add(extract_root_domain(url))
                contacted_urls.add(normalize_url(url))
            if em:
                contacted_emails.add(em)

    selected = []
    for idx, r in enumerate(rows):
        if len(selected) >= remaining_quota:
            break
        st = (r.get("Outreach_Status") or "").strip().lower()
        if not st.startswith("not contacted"):
            continue
        if (r.get("Dofollow") or "Yes").strip().lower() != "yes":
            continue

        subject = (r.get("Email_Subject") or "").strip()
        body = (r.get("Email_Body") or "").strip()
        if not subject or not body:
            continue

        url = (r.get("Referring_Page_URL") or "").strip()
        rd = extract_root_domain(url)
        norm_u = normalize_url(url)
        if not rd or not url or rd in contacted_domains or norm_u in contacted_urls:
            continue

        email = extract_email(r.get("Primary_Email") or "")
        if not email:
            email = discover_email_for_domain(rd, url)
            if email:
                r["Primary_Email"] = email
        if not email or email in contacted_emails:
            continue

        selected.append((idx, r, rd, norm_u, email))
        contacted_domains.add(rd)
        contacted_urls.add(norm_u)
        contacted_emails.add(email)

    print("=" * 72)
    print(f"Sonave Daily Outreach | Mode: {'LIVE SEND' if is_live else 'DRY-RUN'} | Date: {today} | Selected: {len(selected)}/{remaining_quota}")
    print("=" * 72)

    api_key = resolve_resend_api_key() if is_live else ""
    if is_live and selected and not api_key:
        print("[ERROR] RESEND_API_KEY is not configured; aborting live send.", file=sys.stderr)
        sys.exit(1)

    dispatched_now = []

    for i, (row_idx, r, rd, norm_u, email) in enumerate(selected, 1):
        subject = (r.get("Email_Subject") or "").strip()
        body = (r.get("Email_Body") or "").strip()
        if not is_live:
            print(f"  [{i:02d}] [DRY-RUN] {rd} (DR {r.get('DR')}) -> {email} | Subject: {subject}")
            continue

        try:
            resend_id = send_via_resend(api_key, email, subject, body)
        except Exception as exc:
            print(f"  [{i:02d}] [ERROR] Failed sending to {email} ({rd}): {exc}", file=sys.stderr)
            continue

        now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
        rows[row_idx]["Primary_Email"] = email
        rows[row_idx]["Outreach_Status"] = f"Contacted ({today})"

        receipt = {
            "domain": rd,
            "url": r.get("Referring_Page_URL", ""),
            "to": email,
            "subject": subject,
            "resendId": resend_id,
            "sentAt": now_iso,
        }
        dispatch_history.append(receipt)
        dispatched_now.append(receipt)
        _persist_state(dispatch_history, fields, rows)
        print(f"  [{i:02d}] [SENT] {rd} (DR {r.get('DR')}) -> {email} | Resend ID: {resend_id}")
        if i < len(selected):
            time.sleep(1.5)

    if is_live and dispatched_now:
        print(f"\n[DONE] Sent {len(dispatched_now)} emails and updated {TRACKER_CSV.name} + {DISPATCH_LOG.name}.")


if __name__ == "__main__":
    main()
