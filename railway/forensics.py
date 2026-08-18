"""forensics.py — Cryptographic forensic report compiler for Sonave incidents.

Generates tamper-evident, print-ready HTML / PDF audit reports with:
  - Incident telemetry & timeline breakdown
  - Speaker authenticity probability metrics & recent window score history
  - Acoustic model provenance (sonave-xlsr-meet-v2)
  - Cryptographic HMAC-SHA256 signature for SOC2 / ISO27001 compliance audit trails
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any


def compute_forensic_signature(incident: dict[str, Any], secret: bytes) -> tuple[str, str]:
    """Calculate canonical SHA256 digest and HMAC-SHA256 signature of the incident record."""
    canonical = {
        "id": incident.get("id"),
        "speaker": incident.get("speaker"),
        "first_ts": incident.get("first_ts"),
        "last_ts": incident.get("last_ts"),
        "rolling": incident.get("rolling"),
        "model": incident.get("model"),
        "user_id": incident.get("user_id"),
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    signature = hmac.new(secret, raw, hashlib.sha256).hexdigest()
    return digest, signature


def generate_report_html(incident: dict[str, Any], domain: str = "usesonave.com",
                         secret: bytes | None = None) -> str:
    """Generate a high-resolution, print-ready forensic audit report."""
    sec = secret or b"sonave-audit-default-secret"
    digest, signature = compute_forensic_signature(incident, sec)
    
    inc_id = incident.get("id", "N/A")
    speaker = incident.get("speaker", "Unknown Speaker")
    pct = incident.get("rolling", 0.0)
    model = incident.get("model", "sonave-xlsr-meet-v2")
    status = str(incident.get("status", "open")).upper()
    is_held = bool(incident.get("hold"))
    user_id = incident.get("user_id") or ""
    
    first_ts = incident.get("first_ts", time.time())
    last_ts = incident.get("last_ts", time.time())
    duration_sec = max(1, int(last_ts - first_ts))
    
    time_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(first_ts))
    generated_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    
    status_badge = '<span class="badge badge-red">WIRE HELD</span>' if is_held else '<span class="badge badge-amber">FLAGGED</span>'
    
    # Query scores from db if available
    scores_html = ""
    try:
        import db
        if user_id:
            score_rows = db.get_scores(user_id, speaker, 0, 9e12, limit=50)
            if score_rows:
                rows_tr = "".join(
                    f"<tr><td>{time.strftime('%H:%M:%S UTC', time.gmtime(s.get('ts', 0)))}</td>"
                    f"<td><code>{s.get('p_fake', 0):.3f}</code></td>"
                    f"<td><code>{s.get('rolling', 0):.3f}</code></td>"
                    f"<td><span style='font-weight:700; color:{'var(--red)' if s.get('verdict')=='fake' else 'var(--green)'}'>"
                    f"{str(s.get('verdict','')).upper()}</span></td></tr>"
                    for s in score_rows
                )
                scores_html = f"""
                <div class="section">
                  <div class="section-title">Evaluated Audio Window History</div>
                  <table class="table">
                    <tr>
                      <th>Timestamp</th>
                      <th>Window Probability</th>
                      <th>Rolling Probability</th>
                      <th>Verdict</th>
                    </tr>
                    {rows_tr}
                  </table>
                </div>
                """
    except Exception:
        pass
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sonave — Forensic Incident Report #{inc_id}</title>
<style>
  :root {{
    --bg: #ffffff;
    --card: #f8fafc;
    --line: #e2e8f0;
    --text: #0f172a;
    --muted: #64748b;
    --red: #ef4444;
    --red-bg: #fef2f2;
    --red-line: #fecaca;
    --green: #10b981;
    --green-bg: #ecfdf5;
    --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    --mono: ui-monospace, 'IBM Plex Mono', Consolas, monospace;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    padding: 48px;
    max-width: 900px;
    margin: 0 auto;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }}
  @media print {{
    body {{ padding: 24px; max-width: 100%; }}
    .no-print {{ display: none; }}
  }}
  .header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    border-bottom: 2px solid var(--line);
    padding-bottom: 24px;
    margin-bottom: 28px;
  }}
  .brand {{
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #0b0f14;
  }}
  .sub-title {{
    font-size: 14px;
    color: var(--muted);
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-top: 2px;
  }}
  .report-meta {{
    text-align: right;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
  }}
  .alert-banner {{
    background: var(--red-bg);
    border: 1px solid var(--red-line);
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .alert-content {{
    display: flex;
    align-items: center;
    gap: 16px;
  }}
  .alert-icon {{
    font-size: 28px;
  }}
  .alert-heading {{
    font-size: 16px;
    font-weight: 800;
    color: #991b1b;
  }}
  .alert-desc {{
    font-size: 13px;
    color: #7f1d1d;
    margin-top: 2px;
  }}
  .badge {{
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }}
  .badge-red {{
    background: var(--red);
    color: #ffffff;
  }}
  .badge-amber {{
    background: #f59e0b;
    color: #ffffff;
  }}
  .section {{
    margin-bottom: 28px;
  }}
  .section-title {{
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
    border-bottom: 1px solid var(--line);
    padding-bottom: 6px;
  }}
  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }}
  .metric-card {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 14px 18px;
  }}
  .metric-label {{
    font-size: 11px;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .metric-val {{
    font-size: 18px;
    font-weight: 800;
    margin-top: 4px;
    font-family: var(--mono);
  }}
  .metric-val.red {{
    color: var(--red);
  }}
  .table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 8px;
  }}
  .table th {{
    text-align: left;
    padding: 10px 14px;
    background: var(--card);
    border: 1px solid var(--line);
    font-weight: 700;
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .table td {{
    padding: 10px 14px;
    border: 1px solid var(--line);
  }}
  .integrity-box {{
    background: #0b0f14;
    color: #e2e8f0;
    border-radius: 10px;
    padding: 18px 20px;
    font-family: var(--mono);
    font-size: 11px;
    word-break: break-all;
  }}
  .integrity-label {{
    color: #94a3b8;
    font-weight: 700;
    margin-bottom: 4px;
    display: block;
  }}
  .print-bar {{
    display: flex;
    justify-content: flex-end;
    gap: 12px;
    margin-bottom: 24px;
  }}
  .btn {{
    padding: 8px 18px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 13px;
    cursor: pointer;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}
  .btn-print {{
    background: #0f172a;
    color: #ffffff;
    border: 0;
  }}
</style>
</head>
<body>

<div class="print-bar no-print">
  <button class="btn btn-print" onclick="window.print()">🖨 Print / Save as PDF</button>
</div>

<div class="header">
  <div>
    <div class="brand">Sonave</div>
    <h1 class="sub-title">Forensic Incident Report</h1>
  </div>
  <div class="report-meta">
    <div><b>INCIDENT ID:</b> #{inc_id}</div>
    <div><b>REPORT ISSUED:</b> {generated_str}</div>
    <div><b>ENVIRONMENT:</b> Production ({domain})</div>
  </div>
</div>

<div class="alert-banner">
  <div class="alert-content">
    <div class="alert-icon">⛔</div>
    <div>
      <div class="alert-heading">Synthetic Voice Activity Flagged</div>
      <div class="alert-desc">An active speaker crossed the critical authenticity risk threshold during a live video conference.</div>
    </div>
  </div>
  {status_badge}
</div>

<div class="section">
  <div class="section-title">Incident Telemetry & Verification Summary</div>
  <div class="grid-2">
    <div class="metric-card">
      <div class="metric-label">Flagged Speaker ID</div>
      <div class="metric-val">{speaker}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Peak Synthetic Probability</div>
      <div class="metric-val red">{pct:.1%} FAKE</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">First Detected (UTC)</div>
      <div class="metric-val" style="font-size:14px;">{time_str}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Duration Evaluated</div>
      <div class="metric-val" style="font-size:14px;">{duration_sec} seconds</div>
    </div>
  </div>
</div>

{scores_html}

<div class="section">
  <div class="section-title">Acoustic Model & Detection Parameters</div>
  <table class="table">
    <tr>
      <th style="width:30%;">Parameter</th>
      <th>Configuration / Measurement</th>
    </tr>
    <tr>
      <td><b>Detection Engine</b></td>
      <td><code>{model}</code> (Fine-tuned XLS-R + SLS Classifier)</td>
    </tr>
    <tr>
      <td><b>Sampling Rate & Codec</b></td>
      <td>16,000 Hz / Multi-channel Opus WebRTC stream</td>
    </tr>
    <tr>
      <td><b>Temporal Evaluation Window</b></td>
      <td>4.0 second sliding windows with exponential moving average</td>
    </tr>
    <tr>
      <td><b>Risk Classification</b></td>
      <td><span style="color:var(--red); font-weight:700;">CRITICAL RISK (&gt; 70.0% probability)</span></td>
    </tr>
    <tr>
      <td><b>Investigation Status</b></td>
      <td><code>{status}</code></td>
    </tr>
  </table>
</div>

<div class="section">
  <div class="section-title">Recommended Fraud Prevention Protocols</div>
  <table class="table">
    <tr>
      <th style="width:20%;">Action</th>
      <th>Protocol Guidelines</th>
    </tr>
    <tr>
      <td><b style="color:var(--red);">Wire Hold</b></td>
      <td>Immediately pause any wire transfers, account credential resets, or sensitive authorization requests from this meeting.</td>
    </tr>
    <tr>
      <td><b>Out-of-Band Auth</b></td>
      <td>Contact the authorized person via an established, out-of-band secondary channel (e.g. verified phone or corporate Slack).</td>
    </tr>
    <tr>
      <td><b>Incident Archival</b></td>
      <td>Retain this cryptographically signed document in compliance audit archives for SOC2/ISO27001 review.</td>
    </tr>
  </table>
</div>

<div class="section">
  <div class="section-title">Cryptographic Integrity & Tamper-Evidence</div>
  <div class="integrity-box">
    <span class="integrity-label">SHA-256 RECORD DIGEST:</span>
    {digest}
    
    <span class="integrity-label" style="margin-top:12px;">HMAC-SHA256 AUTHENTICATION SIGNATURE:</span>
    {signature}
  </div>
</div>

<div style="font-size:11px; color:var(--muted); text-align:center; margin-top:32px; border-top:1px solid var(--line); padding-top:16px;">
  Generated automatically by Sonave Voice Authenticity Engine · <a href="https://{domain}" style="color:var(--muted); text-decoration:none;">https://{domain}</a>
</div>

</body>
</html>"""
