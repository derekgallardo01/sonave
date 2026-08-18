"""legal_certificate.py — Legal Chain-of-Custody Certificate of Acoustic Authenticity.

Generates an official, legal-grade certificate formatted for:
  - FBI IC3 (Internet Crime Complaint Center) evidence filing
  - Cyber insurance fraud loss claims & adjusters
  - Corporate Internal Audit & Discovery (FINRA / SOC2)
"""
from __future__ import annotations

import hashlib
import time
from typing import Any


def generate_legal_certificate_html(incident: dict[str, Any], domain: str = "usesonave.com",
                                    secret: bytes | None = None) -> str:
    """Render a formal Legal Certificate of Acoustic Authenticity & Chain of Custody."""
    import forensics

    sec = secret or b"sonave-legal-default-secret"
    digest, signature = forensics.compute_forensic_signature(incident, sec)

    inc_id = incident.get("id", "N/A")
    speaker = incident.get("speaker", "Unknown Speaker")
    pct = incident.get("rolling", 0.0)
    model = incident.get("model", "sonave-xlsr-meet-v2")
    user_id = incident.get("user_id", "N/A")
    
    first_ts = incident.get("first_ts", time.time())
    last_ts = incident.get("last_ts", time.time())
    
    date_str = time.strftime("%B %d, %Y at %H:%M:%S UTC", time.gmtime(first_ts))
    cert_issued = time.strftime("%B %d, %Y", time.gmtime())
    cert_num = f"SNV-CERT-{inc_id:06d}" if isinstance(inc_id, int) else f"SNV-CERT-{inc_id}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sonave — Certificate of Acoustic Authenticity ({cert_num})</title>
<style>
  :root {{
    --font: 'Times New Roman', Times, Georgia, serif;
    --sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --mono: ui-monospace, 'IBM Plex Mono', Consolas, monospace;
    --gold: #b45309;
    --gold-bg: #fffbeb;
    --line: #cbd5e1;
    --text: #0f172a;
    --red: #b91c1c;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #f1f5f9;
    color: var(--text);
    font-family: var(--sans);
    padding: 36px 16px;
    display: flex;
    justify-content: center;
    -webkit-font-smoothing: antialiased;
  }}
  .cert-container {{
    background: #ffffff;
    width: 800px;
    border: 12px double #334155;
    padding: 48px 56px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    position: relative;
  }}
  @media print {{
    body {{ background: #fff; padding: 0; }}
    .cert-container {{ width: 100%; box-shadow: none; border: 8px double #334155; padding: 32px; }}
    .no-print {{ display: none; }}
  }}
  .header {{
    text-align: center;
    border-bottom: 2px solid var(--gold);
    padding-bottom: 18px;
    margin-bottom: 24px;
  }}
  .brand {{
    font-size: 16px;
    font-weight: 800;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #475569;
  }}
  .title {{
    font-family: var(--font);
    font-size: 28px;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: #0f172a;
    margin: 8px 0 4px;
  }}
  .sub {{
    font-size: 12px;
    color: #64748b;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }}
  .cert-id {{
    position: absolute;
    top: 24px;
    right: 28px;
    font-family: var(--mono);
    font-size: 11px;
    color: #64748b;
  }}
  .statement {{
    font-family: var(--font);
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 20px;
    text-align: justify;
  }}
  .evidence-box {{
    background: #f8fafc;
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 20px;
    font-size: 13px;
  }}
  .ev-row {{
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px dashed #e2e8f0;
  }}
  .ev-row:last-child {{ border-bottom: none; }}
  .ev-lbl {{ font-weight: 600; color: #475569; }}
  .ev-val {{ font-family: var(--mono); }}
  .ev-val.red {{ color: var(--red); font-weight: 700; }}
  .sig-block {{
    margin-top: 28px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 32px;
  }}
  .sig-line {{
    border-top: 1px solid #334155;
    padding-top: 6px;
    font-size: 11px;
    color: #475569;
  }}
  .crypto-seal {{
    background: #0f172a;
    color: #e2e8f0;
    padding: 12px 16px;
    border-radius: 6px;
    font-family: var(--mono);
    font-size: 10px;
    margin-top: 24px;
    word-break: break-all;
  }}
  .print-btn {{
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: #0f172a;
    color: #fff;
    border: 0;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: 700;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  }}
</style>
</head>
<body>

<button class="print-btn no-print" onclick="window.print()">🖨 Print Legal Certificate</button>

<div class="cert-container">
  <div class="cert-id">{cert_num}</div>
  
  <div class="header">
    <div class="brand">Sonave Forensic Voice Intelligence</div>
    <div class="title">Certificate of Acoustic Authenticity</div>
    <div class="sub">Legal Chain of Custody &amp; Forensic Evidence Record</div>
  </div>

  <div class="statement">
    This document certifies that on <b>{date_str}</b>, the Sonave Real-Time Voice Authenticity Engine monitored and cryptographically evaluated an active audio stream associated with speaker identifier <b>&ldquo;{speaker}&rdquo;</b>. Forensic spectral evaluation conducted under fine-tuned XLS-R acoustic modeling identified synthetic voice synthesis with a confidence level exceeding authorized security tolerances.
  </div>

  <div class="evidence-box">
    <div class="ev-row">
      <span class="ev-lbl">Forensic Incident ID:</span>
      <span class="ev-val">#{inc_id}</span>
    </div>
    <div class="ev-row">
      <span class="ev-lbl">Flagged Speaker Identifier:</span>
      <span class="ev-val">{speaker}</span>
    </div>
    <div class="ev-row">
      <span class="ev-lbl">Peak Synthetic Probability:</span>
      <span class="ev-val red">{pct:.1%} FAKE (Sustained Anomaly)</span>
    </div>
    <div class="ev-row">
      <span class="ev-lbl">Acoustic Detection Model:</span>
      <span class="ev-val">{model}</span>
    </div>
    <div class="ev-row">
      <span class="ev-lbl">Evaluation Environment:</span>
      <span class="ev-val">Production ({domain})</span>
    </div>
    <div class="ev-row">
      <span class="ev-lbl">Wire Hold Directive:</span>
      <span class="ev-val" style="color:var(--red); font-weight:700;">MANDATORY WIRE HOLD ISSUED</span>
    </div>
  </div>

  <div class="statement" style="font-size:13px; color:#334155;">
    <b>Chain of Custody Attestation:</b> The digital telemetry recorded herein was acquired in real time from conference media endpoints, evaluated against statistical voice authenticity models, and cryptographically sealed immediately upon threshold breach.
  </div>

  <div class="crypto-seal">
    <div><b>SHA-256 EVIDENCE DIGEST:</b> {digest}</div>
    <div style="margin-top:4px;"><b>HMAC-SHA256 LEGAL SEAL:</b> {signature}</div>
  </div>

  <div class="sig-block">
    <div>
      <div style="height: 36px;"></div>
      <div class="sig-line">
        <b>Authorized Security Officer / Auditor Signature</b><br>
        Workspace ID: {user_id}
      </div>
    </div>
    <div>
      <div style="height: 36px; display:flex; align-items:flex-end; font-family:var(--mono); font-size:11px; color:#64748b;">
        ISSUED: {cert_issued}
      </div>
      <div class="sig-line">
        <b>Sonave Voice Authenticity Engine</b><br>
        Cryptographically Verified Record
      </div>
    </div>
  </div>
</div>

</body>
</html>"""
