"""webhook_dispatcher.py — Multi-platform rich alert delivery for Sonave incidents.

Formats high-priority security notifications for:
  - Slack (Block Kit with action buttons & color accents)
  - Discord (Rich embeds with 0xff4d5e red alerts)
  - Microsoft Teams (Office 365 MessageCards)
  - Generic Webhook (structured JSON for SIEM / SOAR / custom microservices)

Dispatches asynchronously in daemon threads with 10s timeouts to ensure zero latency impact
on real-time audio scoring loops.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger("sonave.webhooks")


def detect_platform(url: str) -> str:
    """Identify destination platform from the webhook URL."""
    u = (url or "").lower()
    if "hooks.slack.com" in u:
        return "slack"
    if "discord.com/api/webhooks" in u or "discordapp.com/api/webhooks" in u:
        return "discord"
    if "webhook.office.com" in u or "logic.azure.com" in u or "office365" in u:
        return "teams"
    return "generic"


def format_slack_payload(event: dict[str, Any], domain: str = "usesonave.com") -> dict:
    """Format Slack Block Kit message with alert banner and actions."""
    spk = event.get("speaker", "Unknown Speaker")
    pct = event.get("rolling", 0.0)
    model = event.get("model", "sonave-xlsr-meet-v2")
    inc_id = event.get("id", "")
    is_held = bool(event.get("hold"))
    
    status_text = "⛔ *WIRE HOLD RECOMMENDED* — Re-authenticate caller before approving payment." if is_held else "⚠️ Synthetic voice confidence above detection threshold."
    
    console_url = f"https://{domain}/console"
    report_url = f"https://{domain}/api/incidents/{inc_id}/report" if inc_id else console_url
    
    return {
        "text": f"🚨 [Sonave Alert] Deepfake Voice Detected — Speaker '{spk}' ({pct:.0%} Fake)",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🚨 Sonave — Synthetic Voice Alert",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Incident #{inc_id or 'LIVE'}*\n{status_text}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Flagged Speaker:*\n`{spk}`"},
                    {"type": "mrkdwn", "text": f"*Peak Risk Score:*\n*{pct:.0%} FAKE*"},
                    {"type": "mrkdwn", "text": f"*Detection Model:*\n`{model}`"},
                    {"type": "mrkdwn", "text": f"*Timestamp:*\n<!date^{int(time.time())}^{{date_num}} {{time_secs}}|{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}>"}
                ]
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Open Incident Console", "emoji": True},
                        "url": console_url,
                        "style": "primary"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Download Forensic Report", "emoji": True},
                        "url": report_url
                    }
                ]
            }
        ]
    }


def format_discord_payload(event: dict[str, Any], domain: str = "usesonave.com") -> dict:
    """Format Discord rich embed with red color accent."""
    spk = event.get("speaker", "Unknown Speaker")
    pct = event.get("rolling", 0.0)
    model = event.get("model", "sonave-xlsr-meet-v2")
    inc_id = event.get("id", "")
    is_held = bool(event.get("hold"))
    
    console_url = f"https://{domain}/console"
    report_url = f"https://{domain}/api/incidents/{inc_id}/report" if inc_id else console_url

    desc = "**⛔ WIRE HOLD RECOMMENDED**\nImmediate re-authentication required before authorizing transfers." if is_held else "Synthetic voice confidence crossed security threshold."

    return {
        "content": "🚨 **Sonave Voice Authenticity Alert**",
        "embeds": [
            {
                "title": f"Deepfake Detected: {spk}",
                "description": desc,
                "url": console_url,
                "color": 0xFF4D5E,  # Red
                "fields": [
                    {"name": "Speaker", "value": f"`{spk}`", "inline": True},
                    {"name": "Peak Risk", "value": f"**{pct:.0%} FAKE**", "inline": True},
                    {"name": "Model", "value": f"`{model}`", "inline": True},
                    {"name": "Forensic Report", "value": f"[View Audit PDF]({report_url})", "inline": False}
                ],
                "footer": {
                    "text": f"Sonave Fraud Prevention · Incident #{inc_id or 'LIVE'}"
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        ]
    }


def format_teams_payload(event: dict[str, Any], domain: str = "usesonave.com") -> dict:
    """Format Microsoft Teams MessageCard."""
    spk = event.get("speaker", "Unknown Speaker")
    pct = event.get("rolling", 0.0)
    model = event.get("model", "sonave-xlsr-meet-v2")
    inc_id = event.get("id", "")
    is_held = bool(event.get("hold"))
    console_url = f"https://{domain}/console"

    return {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": "FF4D5E",
        "summary": f"Sonave Alert: Synthetic Voice Detected ({spk})",
        "title": "🚨 Sonave — Synthetic Voice Alert",
        "sections": [
            {
                "activityTitle": f"Deepfake Detected: {spk}",
                "activitySubtitle": "Wire Hold Triggered" if is_held else "Security Flag Raised",
                "facts": [
                    {"name": "Speaker:", "value": spk},
                    {"name": "Risk Probability:", "value": f"{pct:.0%} FAKE"},
                    {"name": "Detection Engine:", "value": model},
                    {"name": "Incident ID:", "value": str(inc_id or "LIVE")}
                ],
                "markdown": True
            }
        ],
        "potentialAction": [
            {
                "@type": "OpenUri",
                "name": "Open Console",
                "targets": [{"os": "default", "uri": console_url}]
            }
        ]
    }


def format_generic_payload(event: dict[str, Any], domain: str = "usesonave.com") -> dict:
    """Format standard SIEM / SOAR JSON payload."""
    inc_id = event.get("id", "")
    spk = event.get("speaker", "Unknown Speaker")
    pct = event.get("rolling", 0.0)
    model = event.get("model", "sonave-xlsr-meet-v2")
    return {
        "text": f":rotating_light: *Sonave — suspected deepfake voice*\nSpeaker *{spk}* · risk *{pct:.0%}* · model `{model}`." + (" Wire *HELD* — re-authenticate caller." if event.get("hold") else ""),
        "event_type": "voice.authenticity.alert",
        "incident_id": inc_id,
        "speaker": spk,
        "confidence_fake": pct,
        "verdict": "FAKE",
        "hold_recommended": bool(event.get("hold")),
        "model": model,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "console_url": f"https://{domain}/console",
        "forensic_report_url": f"https://{domain}/api/incidents/{inc_id}/report" if inc_id else f"https://{domain}/console"
    }


def build_payload(url: str, event: dict[str, Any], domain: str = "usesonave.com") -> dict:
    """Automatically select and build the appropriate payload structure for the given URL."""
    plat = detect_platform(url)
    if plat == "slack":
        return format_slack_payload(event, domain)
    if plat == "discord":
        return format_discord_payload(event, domain)
    if plat == "teams":
        return format_teams_payload(event, domain)
    return format_generic_payload(event, domain)


def compute_webhook_signature(payload_bytes: bytes, secret: str, timestamp: int | None = None) -> tuple[str, int]:
    """Compute HMAC-SHA256 signature formatted as t={timestamp},v1={signature}."""
    import hashlib
    import hmac
    ts = timestamp or int(time.time())
    to_sign = f"t={ts}.".encode("utf-8") + payload_bytes
    sig = hmac.new(secret.encode("utf-8"), to_sign, hashlib.sha256).hexdigest()
    return f"t={ts},v1={sig}", ts


def verify_webhook_signature(payload_bytes: bytes, signature_header: str, secret: str, tolerance_sec: int = 300) -> bool:
    """Verify incoming X-Sonave-Signature header against expected HMAC signature."""
    import hashlib
    import hmac
    try:
        parts = dict(item.split("=", 1) for item in signature_header.split(",") if "=" in item)
        ts = int(parts.get("t", 0))
        v1 = parts.get("v1", "")
        if abs(time.time() - ts) > tolerance_sec:
            return False
        to_sign = f"t={ts}.".encode("utf-8") + payload_bytes
        expected = hmac.new(secret.encode("utf-8"), to_sign, hashlib.sha256).hexdigest()
        return hmac.compare_digest(v1, expected)
    except Exception:
        return False


def send_webhook_sync(url: str, payload: dict, secret: str | None = None) -> tuple[bool, int, str]:
    """Synchronous HTTP POST to the webhook endpoint with 10s timeout and optional HMAC signature."""
    if not url:
        return False, 0, "no webhook url configured"
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Sonave-Security-Alerts/1.0"
    }
    if secret:
        sig_header, ts = compute_webhook_signature(body, secret)
        headers["X-Sonave-Signature"] = sig_header
        headers["X-Sonave-Timestamp"] = str(ts)
    req = urllib.request.Request(
        url,
        data=body,
        headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            code = getattr(r, "status", getattr(r, "code", 200))
            return True, code, "ok"
    except urllib.error.HTTPError as e:
        logger.warning("Webhook dispatch failed HTTP %s: %s", e.code, url)
        return False, e.code, str(e.reason)
    except Exception as e:
        logger.warning("Webhook dispatch error: %s", e)
        return False, 0, str(e)


def dispatch_alert(event: dict[str, Any], webhook_url: str, domain: str = "usesonave.com",
                   secret: str | None = None) -> None:
    """Offload alert dispatch to a background daemon thread."""
    if not webhook_url:
        return
    payload = build_payload(webhook_url, event, domain)
    
    def _run():
        ok, code, msg = send_webhook_sync(webhook_url, payload, secret=secret)
        if not ok:
            logger.warning("Async webhook to %s failed (%s: %s)", webhook_url, code, msg)

    threading.Thread(target=_run, daemon=True).start()


def send_test_alert(webhook_url: str, domain: str = "usesonave.com",
                    secret: str | None = None) -> tuple[bool, int, str]:
    """Send an immediate test alert to verify webhook integration."""
    test_event = {
        "id": 9999,
        "speaker": "Test Speaker (Simulation)",
        "rolling": 0.94,
        "model": "sonave-xlsr-meet-v2",
        "hold": True,
        "first_ts": time.time(),
        "last_ts": time.time()
    }
    payload = build_payload(webhook_url, test_event, domain)
    return send_webhook_sync(webhook_url, payload, secret=secret)
