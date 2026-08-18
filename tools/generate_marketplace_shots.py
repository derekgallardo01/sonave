"""
Generate 1280x800 Google Workspace Marketplace screenshots using Playwright.
Produces:
  1. designs/marketplace/shot-meet-panel.png  (Live Meet in-call monitoring)
  2. designs/marketplace/shot-wire-hold.png   (Incident / Wire-hold red alert)
  3. designs/marketplace/shot-protect.png     (Native botless readiness screen)
  4. designs/marketplace/shot-console.png     (Console dashboard & forensics)
  5. designs/marketplace/shot-landing.png     (Product landing overview)
"""

import os
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
ROOT = HERE.parent
OUT_DIR = ROOT / "designs" / "marketplace"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Common SVG Mark
LOGO_SVG = '''<svg viewBox="0 0 24 24" fill="none" style="width:100%;height:100%"><circle cx="12" cy="12" r="8.6" stroke="#2ee584" stroke-width="2.2"></circle><path d="M6.9 12h.9l.6-1.4.7 2.9.8-4.1.9 5.4.9-7.2.9 8.3.8-6.9.7 4.8.7-2.9.6 1.6.5-.5h1.2" stroke="#2ee584" stroke-width="1.15" stroke-linejoin="round" stroke-linecap="round"></path><circle cx="18.1" cy="5.9" r="2.6" fill="#0b0d10"></circle><circle cx="18.1" cy="5.9" r="1.8" fill="#2ee584"></circle></svg>'''

BASE_CSS = """
:root {
  --bg: #0b0d10;
  --panel: #101418;
  --card: #13181d;
  --line: #1b2128;
  --line2: #242c35;
  --ink: #e8eef2;
  --ink2: #9aa9b5;
  --muted: #718695;
  --green: #2ee584;
  --green-dark: #0d9e56;
  --amber: #ffb224;
  --red: #ff4d5e;
  --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  --mono: ui-monospace, 'IBM Plex Mono', Consolas, monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  width: 1280px;
  height: 800px;
  overflow: hidden;
  background: #080a0c;
  color: var(--ink);
  font-family: var(--font);
  -webkit-font-smoothing: antialiased;
}
"""

def make_meet_frame(side_panel_html, is_fake_call=False):
    red_border = "border: 2px solid #ff4d5e; box-shadow: 0 0 24px rgba(255,77,94,0.35);" if is_fake_call else "border: 1px solid #1b2128;"
    fake_badge = '<div style="position:absolute; top:12px; right:12px; background:rgba(255,77,94,0.9); color:#fff; font-size:10px; font-weight:800; padding:4px 8px; border-radius:6px; letter-spacing:0.05em;">SYNTHETIC VOICE DETECTED</div>' if is_fake_call else ''
    
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
{BASE_CSS}
.meet-container {{
  display: flex;
  height: 800px;
  width: 1280px;
  background: #080a0c;
}}
.meet-main {{
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 16px;
  gap: 14px;
}}
.meet-grid {{
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  gap: 12px;
}}
.tile {{
  border-radius: 12px;
  background: #12161b;
  border: 1px solid #1b2128;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}}
.avatar {{
  width: 72px;
  height: 72px;
  border-radius: 50%;
  background: #1b222a;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 26px;
  font-weight: 700;
  color: var(--ink2);
}}
.tile-name {{
  position: absolute;
  left: 12px;
  bottom: 12px;
  background: rgba(0,0,0,0.65);
  backdrop-filter: blur(8px);
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 6px;
}}
.meet-bar {{
  height: 52px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 16px;
}}
.meet-ctrls {{
  display: flex;
  gap: 10px;
}}
.ctrl-btn {{
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: #1a2027;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--ink);
  font-size: 14px;
}}
.ctrl-leave {{
  width: 52px;
  height: 40px;
  border-radius: 20px;
  background: #ea4335;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-size: 13px;
  font-weight: bold;
}}
.side-panel {{
  width: 360px;
  background: var(--bg);
  border-left: 1px solid var(--line);
  display: flex;
  flex-direction: column;
  height: 100%;
}}
</style>
</head>
<body>
<div class="meet-container">
  <div class="meet-main">
    <div class="meet-grid">
      <div class="tile">
        <div class="avatar" style="background:#1e293b; color:#38bdf8;">D</div>
        <div class="tile-name">Derek Gallardo (You)</div>
      </div>
      <div class="tile">
        <div class="avatar" style="background:#2e1065; color:#c084fc;">M</div>
        <div class="tile-name">Maya Lin</div>
      </div>
      <div class="tile">
        <div class="avatar" style="background:#064e3b; color:#34d399;">A</div>
        <div class="tile-name">Alex Chen</div>
      </div>
      <div class="tile" style="{red_border}">
        <div class="avatar" style="background:{'#450a0a; color:#f87171;' if is_fake_call else '#1e293b; color:#94a3b8;'}">{'U' if is_fake_call else 'S'}</div>
        {fake_badge}
        <div class="tile-name">{'Unknown caller' if is_fake_call else 'Sarah Jenkins'}</div>
      </div>
    </div>
    <div class="meet-bar">
      <div style="font-size:13px; color:var(--muted); font-family:var(--mono);">11:42 AM | wire-approval-q3</div>
      <div class="meet-ctrls">
        <div class="ctrl-btn">🎙</div>
        <div class="ctrl-btn">🎥</div>
        <div class="ctrl-btn">✋</div>
        <div class="ctrl-btn">⛶</div>
        <div class="ctrl-leave">⏻</div>
      </div>
      <div style="display:flex; gap:12px; align-items:center;">
        <div class="ctrl-btn" style="background:#2ee584; color:#06130b; font-weight:bold;">S</div>
      </div>
    </div>
  </div>
  <div class="side-panel">
    {side_panel_html}
  </div>
</div>
</body>
</html>"""

def get_shot_protect_html():
    panel_body = f"""
    <div style="padding:14px 16px; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between;">
      <div style="display:flex; align-items:center; gap:8px;">
        <div style="width:24px; height:24px;">{LOGO_SVG}</div>
        <div>
          <b style="font-size:13px;">Sonave</b>
          <small style="display:block; font-size:9px; color:var(--muted); letter-spacing:0.05em;">VOICE AUTHENTICITY · THIS CALL</small>
        </div>
      </div>
    </div>
    <div style="padding:24px 18px; display:flex; flex-direction:column; align-items:center; text-align:center; flex:1;">
      <div style="position:relative; width:80px; height:80px; margin-bottom:18px;">
        <div style="position:absolute; inset:0; border:2px solid rgba(46,229,132,0.3); border-radius:50%;"></div>
        <div style="position:absolute; inset:12px; border:2px solid rgba(46,229,132,0.6); border-radius:50%;"></div>
        <div style="position:absolute; inset:24px; width:32px; height:32px;">{LOGO_SVG}</div>
      </div>
      <div style="font-size:17px; font-weight:800; margin-bottom:6px;">Voice verification ready</div>
      <p style="font-size:12px; color:var(--ink2); line-height:1.55; max-width:280px; margin-bottom:20px;">
        Every voice in this meeting is verified live, while speaking.
      </p>
      <button style="width:100%; padding:11px; background:var(--green); color:#06130b; border:0; border-radius:10px; font-weight:700; font-size:13px; margin-bottom:18px; box-shadow:0 6px 20px rgba(46,229,132,0.25);">
        Start voice verification
      </button>
      <div style="width:100%; display:flex; flex-direction:column; gap:9px; text-align:left;">
        <div style="display:flex; gap:10px; align-items:flex-start; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px 12px;">
          <b style="width:18px; height:18px; border-radius:50%; background:rgba(46,229,132,0.15); color:var(--green); font-size:10px; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:2px;">1</b>
          <span style="font-size:11.5px; color:var(--ink2); line-height:1.45;"><strong style="color:var(--ink);">Real-time verification</strong> — voices verified as participants talk</span>
        </div>
        <div style="display:flex; gap:10px; align-items:flex-start; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px 12px;">
          <b style="width:18px; height:18px; border-radius:50%; background:rgba(46,229,132,0.15); color:var(--green); font-size:10px; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:2px;">2</b>
          <span style="font-size:11.5px; color:var(--ink2); line-height:1.45;"><strong style="color:var(--ink);">4-second verdicts</strong> — refreshed continuously throughout call</span>
        </div>
        <div style="display:flex; gap:10px; align-items:flex-start; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px 12px;">
          <b style="width:18px; height:18px; border-radius:50%; background:rgba(46,229,132,0.15); color:var(--green); font-size:10px; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:2px;">3</b>
          <span style="font-size:11.5px; color:var(--ink2); line-height:1.45;"><strong style="color:var(--ink);">Synthetic alerts</strong> — red wire-hold alert & forensic report</span>
        </div>
      </div>
      <div style="margin-top:16px;">
        <a href="#" style="color:var(--green); font-size:11.5px; font-weight:600; text-decoration:none;">&#9654; Watch a 15-second simulated detection</a>
      </div>
    </div>
    <div style="padding:10px 14px; border-top:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; font-size:10.5px; color:var(--muted);">
      <span><span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:var(--green); margin-right:6px;"></span>detection engine online</span>
      <a href="#" style="color:var(--green); font-weight:600; text-decoration:none;">Open console ↗</a>
    </div>
    """
    return make_meet_frame(panel_body, is_fake_call=False)

def get_shot_meet_panel_html():
    panel_body = f"""
    <div style="padding:14px 16px; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between;">
      <div style="display:flex; align-items:center; gap:8px;">
        <div style="width:24px; height:24px;">{LOGO_SVG}</div>
        <div>
          <b style="font-size:13px;">Sonave</b>
          <small style="display:block; font-size:9px; color:var(--muted); letter-spacing:0.05em;">VOICE AUTHENTICITY · THIS CALL</small>
        </div>
      </div>
    </div>
    <div style="padding:12px 14px; border-bottom:1px solid var(--line); background:rgba(46,229,132,0.05);">
      <div style="font-size:18px; font-weight:800; color:var(--green); letter-spacing:-0.01em;">REAL</div>
      <div style="font-size:11px; color:var(--ink2); margin-top:2px;">All speakers within normal authenticity range.</div>
    </div>
    <div style="padding:12px 14px; display:flex; flex-direction:column; gap:9px; flex:1;">
      <!-- Speaker 1 -->
      <div style="border:1px solid var(--line); border-radius:12px; padding:12px 13px; background:var(--panel);">
        <div style="display:flex; align-items:center; gap:10px;">
          <div style="width:30px; height:30px; border-radius:50%; background:#161c22; border:1px solid var(--line2); display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; color:var(--ink2);">D</div>
          <div style="flex:1; min-width:0;">
            <div style="font-size:13px; font-weight:700;">Derek Gallardo <span style="color:var(--muted); font-weight:normal;">· you</span></div>
            <div style="font-size:10.5px; color:var(--green); margin-top:1px;">speaking</div>
          </div>
          <span style="font-family:var(--mono); font-size:15px; font-weight:700; color:var(--green);">4%</span>
        </div>
        <div style="height:4px; background:#161c22; border-radius:2px; margin-top:9px; overflow:hidden;">
          <div style="height:100%; width:4%; background:var(--green); border-radius:2px;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--muted); margin-top:7px; font-family:var(--mono);">
          <span>94% speech</span><span>14:20</span><span style="font-weight:700; color:var(--green);">REAL</span>
        </div>
      </div>
      <!-- Speaker 2 -->
      <div style="border:1px solid var(--line); border-radius:12px; padding:12px 13px; background:var(--panel);">
        <div style="display:flex; align-items:center; gap:10px;">
          <div style="width:30px; height:30px; border-radius:50%; background:#161c22; border:1px solid var(--line2); display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; color:var(--ink2);">M</div>
          <div style="flex:1; min-width:0;">
            <div style="font-size:13px; font-weight:700;">Maya Lin</div>
            <div style="font-size:10.5px; color:var(--muted); margin-top:1px;">muted / quiet · 12s</div>
          </div>
          <span style="font-family:var(--mono); font-size:15px; font-weight:700; color:var(--green);">8%</span>
        </div>
        <div style="height:4px; background:#161c22; border-radius:2px; margin-top:9px; overflow:hidden;">
          <div style="height:100%; width:8%; background:var(--green); border-radius:2px;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--muted); margin-top:7px; font-family:var(--mono);">
          <span>68% speech</span><span>12:45</span><span style="font-weight:700; color:var(--green);">REAL</span>
        </div>
      </div>
      <!-- Speaker 3 -->
      <div style="border:1px solid var(--line); border-radius:12px; padding:12px 13px; background:var(--panel);">
        <div style="display:flex; align-items:center; gap:10px;">
          <div style="width:30px; height:30px; border-radius:50%; background:#161c22; border:1px solid var(--line2); display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; color:var(--ink2);">A</div>
          <div style="flex:1; min-width:0;">
            <div style="font-size:13px; font-weight:700;">Alex Chen</div>
            <div style="font-size:10.5px; color:var(--muted); margin-top:1px;">monitoring</div>
          </div>
          <span style="font-family:var(--mono); font-size:15px; font-weight:700; color:var(--green);">12%</span>
        </div>
        <div style="height:4px; background:#161c22; border-radius:2px; margin-top:9px; overflow:hidden;">
          <div style="height:100%; width:12%; background:var(--green); border-radius:2px;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--muted); margin-top:7px; font-family:var(--mono);">
          <span>45% speech</span><span>08:10</span><span style="font-weight:700; color:var(--green);">REAL</span>
        </div>
      </div>
    </div>
    <!-- Stats -->
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; padding:2px 14px 12px;">
      <div style="border:1px solid var(--line); background:var(--panel); border-radius:10px; padding:8px 10px;">
        <b style="display:block; font-size:8.5px; letter-spacing:0.08em; color:var(--muted); font-weight:600; margin-bottom:2px;">CHECKS</b>
        <span style="font-family:var(--mono); font-size:13.5px; font-weight:600;">42</span>
      </div>
      <div style="border:1px solid var(--line); background:var(--panel); border-radius:10px; padding:8px 10px;">
        <b style="display:block; font-size:8.5px; letter-spacing:0.08em; color:var(--muted); font-weight:600; margin-bottom:2px;">SESSION</b>
        <span style="font-family:var(--mono); font-size:13.5px; font-weight:600;">14:20</span>
      </div>
      <div style="border:1px solid var(--line); background:var(--panel); border-radius:10px; padding:8px 10px;">
        <b style="display:block; font-size:8.5px; letter-spacing:0.08em; color:var(--muted); font-weight:600; margin-bottom:2px;">PEAK RISK</b>
        <span style="font-family:var(--mono); font-size:13.5px; font-weight:600; color:var(--green);">12%</span>
      </div>
    </div>
    <div style="padding:10px 14px; border-top:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; font-size:10.5px; color:var(--muted);">
      <span><span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:var(--green); margin-right:6px;"></span>detection engine online</span>
      <a href="#" style="color:var(--green); font-weight:600; text-decoration:none;">Open console ↗</a>
    </div>
    """
    return make_meet_frame(panel_body, is_fake_call=False)

def get_shot_wire_hold_html():
    panel_body = f"""
    <div style="padding:14px 16px; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between;">
      <div style="display:flex; align-items:center; gap:8px;">
        <div style="width:24px; height:24px;">{LOGO_SVG}</div>
        <div>
          <b style="font-size:13px;">Sonave</b>
          <small style="display:block; font-size:9px; color:var(--muted); letter-spacing:0.05em;">VOICE AUTHENTICITY · THIS CALL</small>
        </div>
      </div>
    </div>
    <div style="padding:12px 14px; border-bottom:1px solid var(--line); background:rgba(255,77,94,0.12);">
      <div style="font-size:18px; font-weight:800; color:var(--red); letter-spacing:-0.01em;">FAKE</div>
      <div style="font-size:11px; color:#ff8a95; margin-top:2px;">A speaker is scoring in the red band (94% confidence).</div>
    </div>
    <!-- Wire Hold Alert Card -->
    <div style="margin:10px 14px 0; padding:12px 14px; border-radius:11px; border:1px solid rgba(255,77,94,0.4); background:rgba(255,77,94,0.08);">
      <div style="font-size:12px; font-weight:700; color:#ff8a95;">⛔ Wire hold recommended</div>
      <div style="font-size:11px; color:var(--ink2); margin-top:4px; line-height:1.45;">"Unknown caller" crossed the synthetic threshold. Webhook paused payment approval.</div>
      <div style="display:flex; gap:8px; margin-top:10px;">
        <button style="flex:1; padding:7px; border-radius:7px; border:0; background:var(--red); color:#fff; font-weight:700; font-size:11px; cursor:pointer;">Hold the wire</button>
        <button style="padding:7px 10px; border-radius:7px; border:1px solid var(--line2); background:transparent; color:var(--ink2); font-weight:600; font-size:11px; cursor:pointer;">Export report</button>
      </div>
    </div>
    <div style="padding:10px 14px; display:flex; flex-direction:column; gap:9px; flex:1;">
      <!-- Speaker Fake -->
      <div style="border:1px solid rgba(255,77,94,0.5); border-radius:12px; padding:12px 13px; background:rgba(255,77,94,0.04);">
        <div style="display:flex; align-items:center; gap:10px;">
          <div style="width:30px; height:30px; border-radius:50%; background:#2a1014; border:1px solid rgba(255,77,94,0.4); display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; color:var(--red);">U</div>
          <div style="flex:1; min-width:0;">
            <div style="font-size:13px; font-weight:700; color:#ff8a95;">Unknown caller</div>
            <div style="font-size:10.5px; color:var(--red); margin-top:1px;">speaking · synthetic voice</div>
          </div>
          <span style="font-family:var(--mono); font-size:15px; font-weight:700; color:var(--red);">94%</span>
        </div>
        <div style="height:4px; background:#161c22; border-radius:2px; margin-top:9px; overflow:hidden;">
          <div style="height:100%; width:94%; background:var(--red); border-radius:2px;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--muted); margin-top:7px; font-family:var(--mono);">
          <span>52% speech</span><span>06:18</span><span style="font-weight:700; color:var(--red);">FAKE</span>
        </div>
      </div>
      <!-- Speaker Derek -->
      <div style="border:1px solid var(--line); border-radius:12px; padding:12px 13px; background:var(--panel);">
        <div style="display:flex; align-items:center; gap:10px;">
          <div style="width:30px; height:30px; border-radius:50%; background:#161c22; border:1px solid var(--line2); display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; color:var(--ink2);">D</div>
          <div style="flex:1; min-width:0;">
            <div style="font-size:13px; font-weight:700;">Derek Gallardo <span style="color:var(--muted); font-weight:normal;">· you</span></div>
            <div style="font-size:10.5px; color:var(--muted); margin-top:1px;">monitoring</div>
          </div>
          <span style="font-family:var(--mono); font-size:15px; font-weight:700; color:var(--green);">4%</span>
        </div>
        <div style="height:4px; background:#161c22; border-radius:2px; margin-top:9px; overflow:hidden;">
          <div style="height:100%; width:4%; background:var(--green); border-radius:2px;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--muted); margin-top:7px; font-family:var(--mono);">
          <span>88% speech</span><span>18:40</span><span style="font-weight:700; color:var(--green);">REAL</span>
        </div>
      </div>
    </div>
    <!-- Stats -->
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; padding:2px 14px 12px;">
      <div style="border:1px solid var(--line); background:var(--panel); border-radius:10px; padding:8px 10px;">
        <b style="display:block; font-size:8.5px; letter-spacing:0.08em; color:var(--muted); font-weight:600; margin-bottom:2px;">CHECKS</b>
        <span style="font-family:var(--mono); font-size:13.5px; font-weight:600;">68</span>
      </div>
      <div style="border:1px solid var(--line); background:var(--panel); border-radius:10px; padding:8px 10px;">
        <b style="display:block; font-size:8.5px; letter-spacing:0.08em; color:var(--muted); font-weight:600; margin-bottom:2px;">SESSION</b>
        <span style="font-family:var(--mono); font-size:13.5px; font-weight:600;">18:40</span>
      </div>
      <div style="border:1px solid var(--line); background:var(--panel); border-radius:10px; padding:8px 10px;">
        <b style="display:block; font-size:8.5px; letter-spacing:0.08em; color:var(--muted); font-weight:600; margin-bottom:2px;">PEAK RISK</b>
        <span style="font-family:var(--mono); font-size:13.5px; font-weight:600; color:var(--red);">94%</span>
      </div>
    </div>
    <div style="padding:10px 14px; border-top:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; font-size:10.5px; color:var(--muted);">
      <span><span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:var(--red); margin-right:6px;"></span>incident recorded #8402</span>
      <a href="#" style="color:var(--green); font-weight:600; text-decoration:none;">Open console ↗</a>
    </div>
    """
    return make_meet_frame(panel_body, is_fake_call=True)

def get_shot_console_html():
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
{BASE_CSS}
.console-layout {{
  display: flex;
  height: 800px;
  width: 1280px;
  background: #090c0f;
}}
.sidebar {{
  width: 230px;
  border-right: 1px solid var(--line);
  padding: 20px 16px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}}
.nav-item {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 600;
  color: var(--ink2);
  cursor: pointer;
}}
.nav-item.active {{
  background: #141b22;
  color: var(--green);
}}
.content {{
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 24px 32px;
  gap: 20px;
  overflow: hidden;
}}
.stat-card {{
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 12px;
  padding: 16px 18px;
  flex: 1;
}}
</style>
</head>
<body>
<div class="console-layout">
  <div class="sidebar">
    <div style="display:flex; align-items:center; gap:10px; padding:4px 6px 20px;">
      <div style="width:30px; height:30px;">{LOGO_SVG}</div>
      <div>
        <b style="font-size:15px;">Sonave</b>
        <small style="display:block; font-size:9px; color:var(--muted); letter-spacing:0.05em;">ENTERPRISE CONSOLE</small>
      </div>
    </div>
    <div class="nav-item active">📊 Live Monitor</div>
    <div class="nav-item">📁 Session History</div>
    <div class="nav-item">🚨 Incidents & Alerts</div>
    <div class="nav-item">🎙 Voiceprints</div>
    <div class="nav-item">⚡ Webhooks & API</div>
    <div class="nav-item">⚙ Settings & Billing</div>
    <div style="flex:1;"></div>
    <div style="border:1px solid var(--line); border-radius:10px; padding:12px; background:#0e1318;">
      <div style="font-size:9.5px; color:var(--muted); letter-spacing:0.08em; font-weight:700;">DETECTION ENGINE</div>
      <div style="font-family:var(--mono); font-size:12px; color:var(--green); margin-top:4px;">sonave-xlsr-meet-v2</div>
      <div style="font-size:11px; color:var(--ink2); margin-top:4px;">● 95.2% catch rate</div>
    </div>
  </div>
  <div class="content">
    <div style="display:flex; align-items:center; justify-content:space-between;">
      <div>
        <h1 style="font-size:20px; font-weight:800;">Realtime Call Forensics & Incident Log</h1>
        <p style="font-size:12.5px; color:var(--muted); margin-top:2px;">Live monitoring across active Google Meet conferences.</p>
      </div>
      <button style="padding:9px 16px; background:var(--green); color:#06130b; font-weight:700; font-size:12px; border:0; border-radius:8px;">Export Audit Report</button>
    </div>
    <!-- Stats Row -->
    <div style="display:flex; gap:14px;">
      <div class="stat-card">
        <div style="font-size:11px; color:var(--muted); font-weight:600;">ACTIVE CALLS</div>
        <div style="font-size:24px; font-weight:800; font-family:var(--mono); margin-top:6px;">1</div>
        <div style="font-size:11px; color:var(--green); margin-top:4px;">● Live streaming WebRTC</div>
      </div>
      <div class="stat-card">
        <div style="font-size:11px; color:var(--muted); font-weight:600;">AUTHENTICITY VERDICTS</div>
        <div style="font-size:24px; font-weight:800; font-family:var(--mono); margin-top:6px;">1,420</div>
        <div style="font-size:11px; color:var(--ink2); margin-top:4px;">4.2s median latency</div>
      </div>
      <div class="stat-card">
        <div style="font-size:11px; color:var(--muted); font-weight:600;">INCIDENTS FLAGGED</div>
        <div style="font-size:24px; font-weight:800; font-family:var(--mono); color:var(--red); margin-top:6px;">1</div>
        <div style="font-size:11px; color:#ff8a95; margin-top:4px;">Wire hold auto-triggered</div>
      </div>
      <div class="stat-card">
        <div style="font-size:11px; color:var(--muted); font-weight:600;">AVAILABLE MONITORING</div>
        <div style="font-size:24px; font-weight:800; font-family:var(--mono); margin-top:6px;">5.0 <span style="font-size:14px; color:var(--muted);">hrs</span></div>
        <div style="font-size:11px; color:var(--green); margin-top:4px;">Free tier active</div>
      </div>
    </div>
    <!-- Recent Session Table -->
    <div style="border:1px solid var(--line); background:var(--panel); border-radius:12px; overflow:hidden; flex:1; display:flex; flex-direction:column;">
      <div style="padding:14px 18px; border-bottom:1px solid var(--line); font-size:13px; font-weight:700; display:flex; justify-content:space-between;">
        <span>Recent Meeting Activity & Risk Analysis</span>
        <span style="color:var(--green); font-size:12px;">Auto-refreshing</span>
      </div>
      <div style="display:grid; grid-template-columns:1.5fr 1fr 1fr 1fr 1fr; padding:10px 18px; border-bottom:1px solid var(--line); font-size:11px; color:var(--muted); font-weight:700; letter-spacing:0.05em;">
        <span>MEETING / TOPIC</span><span>DURATION</span><span>SPEAKERS</span><span>PEAK RISK</span><span>VERDICT</span>
      </div>
      <div style="display:grid; grid-template-columns:1.5fr 1fr 1fr 1fr 1fr; padding:14px 18px; border-bottom:1px solid var(--line); font-size:12.5px; align-items:center; background:rgba(255,77,94,0.03);">
        <div><b style="color:var(--ink);">wire-approval-q3</b><small style="display:block; color:var(--muted); font-size:10.5px;">Google Meet · today 11:42 AM</small></div>
        <span style="font-family:var(--mono);">18m 40s</span>
        <span>4 participants</span>
        <span style="font-family:var(--mono); color:var(--red); font-weight:700;">94%</span>
        <span style="padding:3px 8px; border-radius:5px; background:rgba(255,77,94,0.15); color:var(--red); font-weight:800; font-size:10.5px; width:fit-content;">FAKE (HELD)</span>
      </div>
      <div style="display:grid; grid-template-columns:1.5fr 1fr 1fr 1fr 1fr; padding:14px 18px; border-bottom:1px solid var(--line); font-size:12.5px; align-items:center;">
        <div><b style="color:var(--ink);">weekly-exec-standup</b><small style="display:block; color:var(--muted); font-size:10.5px;">Google Meet · yesterday 2:00 PM</small></div>
        <span style="font-family:var(--mono);">45m 12s</span>
        <span>6 participants</span>
        <span style="font-family:var(--mono); color:var(--green); font-weight:700;">6%</span>
        <span style="padding:3px 8px; border-radius:5px; background:rgba(46,229,132,0.15); color:var(--green); font-weight:800; font-size:10.5px; width:fit-content;">REAL</span>
      </div>
      <div style="display:grid; grid-template-columns:1.5fr 1fr 1fr 1fr 1fr; padding:14px 18px; font-size:12.5px; align-items:center;">
        <div><b style="color:var(--ink);">client-vendor-onboarding</b><small style="display:block; color:var(--muted); font-size:10.5px;">Google Meet · Aug 16 10:30 AM</small></div>
        <span style="font-family:var(--mono);">32m 05s</span>
        <span>3 participants</span>
        <span style="font-family:var(--mono); color:var(--green); font-weight:700;">9%</span>
        <span style="padding:3px 8px; border-radius:5px; background:rgba(46,229,132,0.15); color:var(--green); font-weight:800; font-size:10.5px; width:fit-content;">REAL</span>
      </div>
    </div>
  </div>
</div>
</body>
</html>"""

def get_shot_landing_html():
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
{BASE_CSS}
.landing-layout {{
  display: flex;
  flex-direction: column;
  height: 800px;
  width: 1280px;
  background: radial-gradient(circle at 50% 10%, #15221b 0%, #080a0c 55%);
  padding: 24px 48px;
  gap: 32px;
}}
.nav {{
  display: flex;
  align-items: center;
  justify-content: space-between;
}}
.hero {{
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  gap: 16px;
  margin-top: 10px;
}}
.badge {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 14px;
  border-radius: 20px;
  border: 1px solid rgba(46,229,132,0.3);
  background: rgba(46,229,132,0.08);
  font-size: 12px;
  font-weight: 700;
  color: var(--green);
}}
.title {{
  font-size: 38px;
  font-weight: 900;
  letter-spacing: -0.02em;
  line-height: 1.15;
  max-width: 760px;
}}
.desc {{
  font-size: 15px;
  color: var(--ink2);
  line-height: 1.6;
  max-width: 620px;
}}
.cards {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 20px;
  margin-top: 12px;
}}
.feat-card {{
  border: 1px solid var(--line);
  background: var(--panel);
  border-radius: 14px;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}}
</style>
</head>
<body>
<div class="landing-layout">
  <div class="nav">
    <div style="display:flex; align-items:center; gap:10px;">
      <div style="width:28px; height:28px;">{LOGO_SVG}</div>
      <b style="font-size:17px; letter-spacing:-0.01em;">Sonave</b>
    </div>
    <div style="display:flex; gap:12px;">
      <button style="padding:8px 18px; border-radius:8px; border:1px solid var(--line2); background:transparent; color:var(--ink); font-weight:600; font-size:12.5px;">Sign In</button>
      <button style="padding:8px 18px; border-radius:8px; border:0; background:var(--green); color:#06130b; font-weight:700; font-size:12.5px;">Get Started Free</button>
    </div>
  </div>
  <div class="hero">
    <div class="badge"><span style="width:6px; height:6px; border-radius:50%; background:var(--green);"></span> LIVE VOICE AUTHENTICITY FOR MEETINGS</div>
    <div class="title">Stop deepfake voice fraud before the money moves.</div>
    <div class="desc">Sonave verifies the authenticity of every speaker in real time directly inside Google Meet. Catches AI voice clones in 4 seconds with 95% detection accuracy.</div>
  </div>
  <div class="cards">
    <div class="feat-card">
      <div style="font-size:18px;">⚡</div>
      <div style="font-size:14.5px; font-weight:700;">Sub-4s Realtime Latency</div>
      <div style="font-size:12px; color:var(--muted); line-height:1.5;">Evaluates 4-second audio windows on fine-tuned meeting codecs. Rolling scores refresh continuously.</div>
    </div>
    <div class="feat-card">
      <div style="font-size:18px;">🎯</div>
      <div style="font-size:14.5px; font-weight:700;">95% Catch Rate</div>
      <div style="font-size:12px; color:var(--muted); line-height:1.5;">Trained against 27 unseen voice generators, neural TTS, diffusion models, and real-time voice changers.</div>
    </div>
    <div class="feat-card">
      <div style="font-size:18px;">🛡</div>
      <div style="font-size:14.5px; font-weight:700;">Instant Wire-Hold Webhook</div>
      <div style="font-size:12px; color:var(--muted); line-height:1.5;">Automatically triggers payout holds and generates compliance-ready forensic PDF evidence for audits.</div>
    </div>
  </div>
</div>
</body>
</html>"""

def generate_all_screenshots():
    shots = [
        ("shot-protect.png", get_shot_protect_html()),
        ("shot-meet-panel.png", get_shot_meet_panel_html()),
        ("shot-wire-hold.png", get_shot_wire_hold_html()),
        ("shot-console.png", get_shot_console_html()),
        ("shot-landing.png", get_shot_landing_html()),
    ]
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800}, device_scale_factor=1)
        
        for name, html in shots:
            out_file = OUT_DIR / name
            page.set_content(html)
            page.wait_for_timeout(300)
            page.screenshot(path=str(out_file))
            print(f"Generated: {out_file.name} ({out_file.stat().st_size} bytes)")
            
        browser.close()

if __name__ == "__main__":
    generate_all_screenshots()
