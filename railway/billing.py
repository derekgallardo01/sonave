"""billing.py — Stripe metered billing, stdlib-only.

Pricing: free SONAVE_FREE_MINUTES per calendar month (default 300 = 5 monitored
hours), then $8/monitored-hour billed through a Stripe metered subscription
(card on file via Checkout; usage reported as Billing Meter events). A monthly
spend cap blocks bot launches past SONAVE_MONTHLY_CAP_USD as a surprise-bill and
abuse guard. Admin users are unlimited and never metered to Stripe.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import db

logger = logging.getLogger("sonave.billing")

STRIPE_API = "https://api.stripe.com/v1"
RATE_USD_PER_HOUR = 8.0


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def configured() -> bool:
    return bool(_env("SONAVE_STRIPE_SECRET_KEY"))


def free_minutes() -> int:
    return int(_env("SONAVE_FREE_MINUTES", "300"))


def monthly_cap_usd() -> float:
    return float(_env("SONAVE_MONTHLY_CAP_USD", "200"))


def month_key(ts: float | None = None) -> str:
    return time.strftime("%Y-%m", time.gmtime(ts if ts is not None else time.time()))


# --- Stripe client (form-encoded POSTs, Bearer key) ---------------------------
def _stripe_post(path: str, form: dict) -> dict:
    body = urllib.parse.urlencode(form).encode()
    req = urllib.request.Request(f"{STRIPE_API}{path}", data=body, method="POST", headers={
        "Authorization": f"Bearer {_env('SONAVE_STRIPE_SECRET_KEY')}",
        "Content-Type": "application/x-www-form-urlencoded",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def verify_webhook_sig(body: bytes, sig_header: str, secret: str, tolerance: int = 300) -> bool:
    """Stripe-Signature: t=<ts>,v1=<hmac>,... — HMAC-SHA256 over f"{t}.{body}"."""
    try:
        parts = dict(p.split("=", 1) for p in sig_header.split(",") if "=" in p)
        t = int(parts.get("t", "0"))
    except Exception:  # noqa: BLE001
        return False
    if abs(time.time() - t) > tolerance:
        return False
    signed = f"{t}.".encode() + body
    want = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    v1s = [v.split("=", 1)[1] for v in sig_header.split(",") if v.strip().startswith("v1=")]
    return any(hmac.compare_digest(want, v) for v in v1s)


# --- entitlements -------------------------------------------------------------
def entitlement(user_id: str, role: str = "member") -> dict:
    used = db.get_usage_minutes(user_id, month_key())
    free = free_minutes()
    if role == "admin":
        return {"plan": "admin", "minutes_used": round(used, 1), "free_minutes": free,
                "remaining_free": None, "has_card": True, "active": True,
                "rate_usd_per_hour": RATE_USD_PER_HOUR}
    sub = db.get_subscription(user_id)
    active = bool(sub and sub.get("status") in ("active", "trialing"))
    billable_min = max(0.0, used - free)
    spend = billable_min / 60 * RATE_USD_PER_HOUR
    return {"plan": "metered" if active else "free",
            "minutes_used": round(used, 1), "free_minutes": free,
            "remaining_free": max(0.0, round(free - used, 1)),
            "has_card": active, "active": True,
            "month_spend_usd": round(spend, 2), "monthly_cap_usd": monthly_cap_usd(),
            "rate_usd_per_hour": RATE_USD_PER_HOUR}


def can_launch_bot(user_id: str, role: str = "member") -> dict | None:
    """None = allowed; else an error dict for the /bot response."""
    if role == "admin":
        return None
    ent = entitlement(user_id, role)
    if ent["plan"] == "free":
        if ent["remaining_free"] <= 0:
            return {"ok": False, "code": "quota_exceeded",
                    "detail": f"Free {free_minutes() // 60} monitored hours used this month — "
                              f"add a card to continue at ${RATE_USD_PER_HOUR:g}/hour."}
        return None
    if ent.get("month_spend_usd", 0) >= ent.get("monthly_cap_usd", 1e9):
        return {"ok": False, "code": "spend_cap",
                "detail": f"Monthly spend cap (${monthly_cap_usd():g}) reached — "
                          "raise it from the billing portal or contact us."}
    return None


# --- usage -> Stripe meter events ---------------------------------------------
def report_meter_event(user_id: str, billable_minutes: float, idempotency_key: str) -> None:
    """Push billable minutes (beyond the free tier) as a Stripe Billing Meter event.
    Best-effort: Stripe being down must never affect capture; our usage table is
    the source of truth and can be replayed."""
    if billable_minutes <= 0 or not configured():
        return
    sub = db.get_subscription(user_id)
    if not sub or not sub.get("stripe_customer_id"):
        return
    try:
        _stripe_post("/billing/meter_events", {
            "event_name": _env("SONAVE_STRIPE_METER_EVENT", "sonave_monitored_minutes"),
            "identifier": idempotency_key,
            "payload[stripe_customer_id]": sub["stripe_customer_id"],
            "payload[value]": f"{billable_minutes:.2f}",
        })
    except Exception as e:  # noqa: BLE001
        logger.warning("meter event failed for %s: %s", user_id, repr(e)[:100])


def meter_usage(user_id: str, role: str, minutes: float, idempotency_key: str) -> None:
    """Called from the WS metering tick: bucket locally, then report only the
    slice of this tick that falls beyond the free allowance."""
    month = month_key()
    total_after = db.add_usage_minutes(user_id, month, minutes)
    if role == "admin":
        return
    free = float(free_minutes())
    billable = min(minutes, max(0.0, total_after - free))
    if billable > 0:
        report_meter_event(user_id, billable, idempotency_key)


# --- checkout / portal / webhook ----------------------------------------------
def create_checkout(user_id: str, base_url: str) -> str:
    form = {
        "mode": "subscription",
        "line_items[0][price]": _env("SONAVE_STRIPE_PRICE_METERED"),
        "client_reference_id": user_id,
        "success_url": f"{base_url}/console?billing=success",
        "cancel_url": f"{base_url}/console?billing=cancelled",
    }
    sub = db.get_subscription(user_id)
    if sub and sub.get("stripe_customer_id"):
        form["customer"] = sub["stripe_customer_id"]
    return _stripe_post("/checkout/sessions", form)["url"]


def create_portal(user_id: str, base_url: str) -> str | None:
    sub = db.get_subscription(user_id)
    if not sub or not sub.get("stripe_customer_id"):
        return None
    return _stripe_post("/billing_portal/sessions", {
        "customer": sub["stripe_customer_id"],
        "return_url": f"{base_url}/console",
    })["url"]


def handle_webhook(event: dict) -> None:
    etype = event.get("type", "")
    obj = (event.get("data") or {}).get("object") or {}
    if etype == "checkout.session.completed":
        uid = obj.get("client_reference_id")
        if uid:
            db.upsert_subscription(uid, obj.get("customer"), obj.get("subscription"), "active")
            db.add_event(uid, "subscription_checkout",
                         json.dumps({"customer": obj.get("customer") or ""}))
            logger.info("billing: %s activated metered plan", uid)
    elif etype in ("customer.subscription.updated", "customer.subscription.deleted"):
        cust = obj.get("customer")
        status = "canceled" if etype.endswith("deleted") else obj.get("status", "active")
        # map customer -> user via our table
        c = db._conn()
        try:
            r = c.execute("SELECT user_id FROM subscriptions WHERE stripe_customer_id=?",
                          (cust,)).fetchone()
        finally:
            c.close()
        if r:
            db.upsert_subscription(r["user_id"], cust, obj.get("id"), status)
            kind = "subscription_canceled" if status == "canceled" else "subscription_updated"
            db.add_event(r["user_id"], kind, json.dumps({"status": status}))
            logger.info("billing: %s subscription -> %s", r["user_id"], status)
