"""
report.py — generate a per-meeting forensic authenticity report (the sellable artifact).

Runs in the detector env (.venv):

    python service/report.py meeting.wav                     # single stream
    python service/report.py meeting.wav --segments turns.csv  # per-speaker

Turns the analyzer's output into a clean, self-contained, printable HTML report
(open in a browser -> Print -> Save as PDF). This is the compliance/audit artifact a
finance team needs: who spoke, when, per-speaker authenticity verdict, flagged
stretches, a score timeline, and an auditable methodology footer (model + thresholds).
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_meeting  # noqa: E402
import detector  # noqa: E402

_COL = {"real": "#2f9e5f", "suspect": "#d9911f", "fake": "#d64545", "no_speech": "#8a94a6"}
FAVICON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
           '<rect width="32" height="32" rx="7" fill="#2f6df6"/><g fill="#fff">'
           '<rect x="6" y="13" width="3" height="6" rx="1.5"/><rect x="11" y="9" width="3" height="14" rx="1.5"/>'
           '<rect x="16" y="5" width="3" height="22" rx="1.5"/><rect x="21" y="10" width="3" height="12" rx="1.5"/>'
           '<rect x="26" y="14" width="3" height="4" rx="1.5"/></g></svg>')


def _svg_timeline(series: dict, dur: float, flagged: dict, w=820, h=200) -> str:
    """series: {speaker: [(t, rolling)]}. Shade flagged regions, draw threshold lines."""
    pl, pt, pr, pb = 46, 14, 14, 26
    iw, ih = w - pl - pr, h - pt - pb
    dur = max(dur, 1.0)

    def X(t):
        return pl + iw * min(max(t / dur, 0), 1)

    def Y(v):
        return pt + ih * (1 - min(max(v, 0), 1))

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px">']
    parts.append(f'<rect x="{pl}" y="{pt}" width="{iw}" height="{ih}" fill="#f7f9fc" stroke="#e2e7f0"/>')
    # flagged regions (shaded)
    for spk, ranges in flagged.items():
        for r in ranges:
            x0, x1 = X(r["start_s"]), X(r["end_s"])
            parts.append(f'<rect x="{x0:.1f}" y="{pt}" width="{max(x1-x0,1):.1f}" height="{ih}" '
                         f'fill="#d64545" opacity="0.08"/>')
    # threshold lines
    for val, lab, col in ((detector.TAU_FAKE, "fake", "#d64545"),
                          (detector.TAU_REAL, "suspect", "#d9911f")):
        y = Y(val)
        parts.append(f'<line x1="{pl}" y1="{y:.1f}" x2="{pl+iw}" y2="{y:.1f}" stroke="{col}" '
                     f'stroke-dasharray="4 3" opacity="0.5"/>')
        parts.append(f'<text x="{pl-6}" y="{y+3:.1f}" text-anchor="end" font-size="9" fill="{col}">{val:.2f}</text>')
    # y labels
    for v in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{pl-6}" y="{Y(v)+3:.1f}" text-anchor="end" font-size="9" fill="#8a94a6">{v:.1f}</text>')
    # speaker lines
    palette = ["#2f6df6", "#7b5cff", "#e0559b", "#18a999", "#e8853a"]
    for i, (spk, pts) in enumerate(series.items()):
        if not pts:
            continue
        col = palette[i % len(palette)]
        d = "M" + " L".join(f"{X(t):.1f} {Y(v):.1f}" for t, v in pts)
        parts.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="1.8"/>')
    parts.append(f'<text x="{pl}" y="{h-8}" font-size="9" fill="#8a94a6">0s</text>')
    parts.append(f'<text x="{pl+iw}" y="{h-8}" text-anchor="end" font-size="9" fill="#8a94a6">{dur:.0f}s</text>')
    parts.append("</svg>")
    return "".join(parts)


def _speaker_rows(rep) -> tuple[str, dict, dict]:
    """Return (rows_html, series, flagged) for either mode."""
    rows, series, flagged = [], {}, {}
    if rep.get("mode") == "per_speaker":
        items = rep["speakers"].items()
    else:
        items = [(rep.get("file", "meeting"), rep)]
    for spk, v in items:
        vd = v.get("verdict", "no_speech")
        series[spk] = [(p["t"], p["rolling"]) for p in v.get("timeline", [])]
        flagged[spk] = v.get("flagged_stretches", [])
        fl = "".join(f"<div>{f['start_s']}s–{f['end_s']}s (peak {f['peak']})</div>"
                     for f in v.get("flagged_stretches", [])) or "<span class=mut>none</span>"
        rows.append(
            f"<tr><td><b>{html.escape(str(spk))}</b></td>"
            f"<td><span class='badge' style='background:{_COL.get(vd)}'>{vd.upper()}</span></td>"
            f"<td>{v.get('peak_rolling','—')}</td><td>{v.get('windows','—')}</td><td>{fl}</td></tr>")
    return "".join(rows), series, flagged


def render_html(rep: dict) -> str:
    overall = rep.get("overall_verdict") or rep.get("verdict", "no_speech")
    rows, series, flagged = _speaker_rows(rep)
    chart = _svg_timeline(series, rep.get("duration_s", 1), flagged)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Sonave Report — {html.escape(rep.get('file',''))}</title>
<link rel="icon" href="data:image/svg+xml;base64,{__import__('base64').b64encode(FAVICON.encode()).decode()}">
<style>
*{{box-sizing:border-box}}body{{font:14px/1.5 system-ui,Segoe UI,sans-serif;color:#1a2230;background:#fff;max-width:900px;margin:0 auto;padding:32px}}
.hd{{display:flex;align-items:center;gap:12px;border-bottom:2px solid #eef1f6;padding-bottom:14px}}
.hd .logo{{width:34px;height:34px}} h1{{font-size:20px;margin:0}} .sub{{color:#6a7688;font-size:13px}}
.banner{{margin:18px 0;padding:14px 18px;border-radius:10px;color:#fff;font-weight:700;letter-spacing:.03em}}
.meta{{display:flex;flex-wrap:wrap;gap:10px 26px;color:#6a7688;font-size:13px;margin:10px 0 18px}}
table{{width:100%;border-collapse:collapse;margin:8px 0 20px}} th,td{{text-align:left;padding:9px 10px;border-bottom:1px solid #eef1f6;vertical-align:top}}
th{{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#8a94a6}}
.badge{{color:#fff;font-size:11px;font-weight:700;padding:3px 9px;border-radius:5px}}
.card{{border:1px solid #eef1f6;border-radius:10px;padding:16px 18px;margin:14px 0}}
.mut{{color:#9aa4b4}} h3{{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:#8a94a6;margin:22px 0 6px}}
.foot{{margin-top:26px;border-top:1px solid #eef1f6;padding-top:12px;color:#8a94a6;font-size:12px}}
@media print{{body{{padding:0}} .banner{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}}}
</style></head><body>
<div class="hd"><span class="logo">{FAVICON}</span>
<div><h1>Meeting Authenticity Report</h1><div class="sub">Sonave · deepfake &amp; impersonation analysis</div></div></div>
<div class="banner" style="background:{_COL.get(overall)}">OVERALL: {overall.upper()}</div>
<div class="meta">
<div><b>Recording:</b> {html.escape(rep.get('file',''))}</div>
<div><b>Duration:</b> {rep.get('duration_s','—')} s</div>
<div><b>Windows scored:</b> {rep.get('windows_scored', sum(v.get('windows',0) for v in rep.get('speakers',{}).values()) if rep.get('mode')=='per_speaker' else rep.get('windows','—'))}</div>
<div><b>Model:</b> {html.escape(rep.get('model_version',''))}</div>
</div>
<h3>Per-speaker findings</h3>
<table><tr><th>Speaker</th><th>Verdict</th><th>Peak P(fake)</th><th>Windows</th><th>Flagged stretches</th></tr>{rows}</table>
<h3>Authenticity timeline</h3>
<div class="card">{chart}
<div class="sub" style="margin-top:8px">Rolling P(fake) per speaker. Dashed lines = suspect/fake thresholds; shaded bands = flagged stretches.</div></div>
<div class="foot">
Methodology: audio windowed at {rep.get('hop_s','—')} s hop; each window scored by the Sonave detector
(<b>{html.escape(rep.get('model_version',''))}</b>); a rolling confidence is thresholded at
suspect ≥ {detector.TAU_REAL}, fake ≥ {detector.TAU_FAKE}. Enrolled speakers additionally verified by voiceprint.
This report is an automated risk assessment, not a definitive determination; corroborate flagged findings before acting.
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--hop", type=float, default=2.0)
    ap.add_argument("--segments", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rep = analyze_meeting.analyze(Path(args.audio), args.hop, args.segments)
    out = Path(args.out) if args.out else Path(args.audio).with_suffix(".report.html")
    out.write_text(render_html(rep), encoding="utf-8")
    ov = rep.get("overall_verdict") or rep.get("verdict")
    print(f"OVERALL: {str(ov).upper()}  ->  report written to {out}")


if __name__ == "__main__":
    main()
