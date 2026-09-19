"""
Generate 1280x800 Google Workspace Marketplace screenshots using Playwright.
Produces:
  1. shot-protect.png     (One-Click Voice Verification Ready Screen in Meet)
  2. shot-meet-panel.png  (Live Meet In-Call Real-Time Speaker Monitoring)
  3. shot-wire-hold.png   (Red Alert: Synthetic Voice Detected & Wire Hold Triggered)
  4. shot-console.png     (Sonave Enterprise Console & Forensics Dashboard)
  5. shot-landing.png     (Product Landing Page & Core Value Overview)

Saves to designs/marketplace/ and mirrors to railway/.
"""

import os
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
ROOT = HERE.parent
OUT_DIR = ROOT / "designs" / "marketplace"
RAILWAY_DIR = ROOT / "railway"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Official Sonave Radar Scope SVG Mark
LOGO_SVG = '''<svg viewBox="0 0 24 24" fill="none" style="width:100%;height:100%"><circle cx="12" cy="12" r="8.6" stroke="#2ee584" stroke-width="2.2"></circle><path d="M6.9 12h.9l.6-1.4.7 2.9.8-4.1.9 5.4.9-7.2.9 8.3.8-6.9.7 4.8.7-2.9.6 1.6.5-.5h1.2" stroke="#2ee584" stroke-width="1.15" stroke-linejoin="round" stroke-linecap="round"></path><circle cx="18.1" cy="5.9" r="2.6" fill="#0b0d10"></circle><circle cx="18.1" cy="5.9" r="1.8" fill="#2ee584"></circle></svg>'''

# Reusable Vector SVG Icons
ICONS = {
    "mic": '''<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg>''',
    "camera": '''<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m22 8-6 4 6 4V8Z"></path><rect width="14" height="12" x="2" y="6" rx="2" ry="2"></rect></svg>''',
    "hand": '''<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 11V6a2 2 0 0 0-4 0v5"></path><path d="M14 10V4a2 2 0 0 0-4 0v7"></path><path d="M10 10.5V6a2 2 0 0 0-4 0v8"></path><path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"></path></svg>''',
    "screen": '''<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 3H4a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-3"></path><path d="M8 21h8"></path><path d="M12 17v4"></path><path d="m17 8 5-5"></path><path d="M17 3h5v5"></path></svg>''',
    "dots": '''<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="1.5"></circle><circle cx="12" cy="5" r="1.5"></circle><circle cx="12" cy="19" r="1.5"></circle></svg>''',
    "phone_down": '''<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 9c-1.6 0-3.15.25-4.6.72v3.1c0 .39-.23.74-.56.9-.98.49-1.87 1.12-2.66 1.85-.18.18-.43.28-.7.28-.28 0-.53-.11-.71-.29L.29 13.08a.99.99 0 0 1 0-1.41C3.36 8.78 7.46 7 12 7s8.64 1.78 11.71 4.67c.39.39.39 1.02 0 1.41l-2.48 2.48c-.18.18-.43.29-.71.29-.27 0-.52-.1-.7-.28-.79-.74-1.69-1.36-2.67-1.85-.33-.16-.56-.5-.56-.9v-3.1C15.15 9.25 13.6 9 12 9z"/></svg>''',
    "info": '''<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M12 16v-4"></path><path d="M12 8h.01"></path></svg>''',
    "people": '''<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M22 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>''',
    "chat": '''<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>''',
    "shapes": '''<svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2 6 12h12L12 2zM5 14a4 4 0 1 0 0 8 4 4 0 0 0 0-8zm10 0h6v8h-6v-8z"/></svg>''',
    "shield_alert": '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ff4d5e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>''',
    "shield_check": '''<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2ee584" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><polyline points="9 12 11 14 15 10"></polyline></svg>''',
    "zap": '''<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2ee584" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>''',
    "target": '''<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2ee584" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="22" y1="12" x2="18" y2="12"></line><line x1="6" y1="12" x2="2" y2="12"></line><line x1="12" y1="6" x2="12" y2="2"></line><line x1="12" y1="22" x2="12" y2="18"></line></svg>''',
    "monitor": '''<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>''',
    "clock": '''<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>''',
    "alert_triangle": '''<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>''',
    "mic_nav": '''<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg>''',
    "zap_nav": '''<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>''',
    "settings_nav": '''<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>''',
}

BASE_CSS = """
:root {
  --bg: #090c0f;
  --panel: #0d1217;
  --card: #121820;
  --line: #18222c;
  --line2: #22303e;
  --ink: #e8eef2;
  --ink2: #9aa9b5;
  --muted: #6f8291;
  --dim: #435260;
  --green: #2ee584;
  --green-glow: rgba(46, 229, 132, 0.35);
  --amber: #ffb224;
  --red: #ff4d5e;
  --red-glow: rgba(255, 77, 94, 0.35);
  --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  --mono: ui-monospace, 'IBM Plex Mono', Consolas, monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  width: 1280px;
  height: 800px;
  overflow: hidden;
  background: #06080a;
  color: var(--ink);
  font-family: var(--font);
  -webkit-font-smoothing: antialiased;
}
"""

def make_meet_frame(side_panel_html, is_fake_call=False):
    red_border = "border: 2px solid #ff4d5e; box-shadow: 0 0 30px rgba(255,77,94,0.4);" if is_fake_call else "border: 1px solid #1a232c;"
    fake_badge = '''<div style="position:absolute; top:12px; right:12px; background:#ff4d5e; color:#fff; font-size:10px; font-weight:800; padding:5px 9px; border-radius:6px; letter-spacing:0.06em; box-shadow:0 2px 10px rgba(255,77,94,0.5); display:flex; align-items:center; gap:5px;"><span style="width:6px;height:6px;border-radius:50%;background:#fff;"></span>SYNTHETIC VOICE DETECTED</div>''' if is_fake_call else ''
    
    avatar_fake = f"""
      <div class="avatar" style="background:{'#3b0a0e' if is_fake_call else '#16222f'}; color:{'#ff6b77' if is_fake_call else '#93c5fd'}; border: 2px solid {'#ff4d5e' if is_fake_call else 'transparent'};">
        {'U' if is_fake_call else 'S'}
      </div>
    """

    wave_derek = """
      <span style="display:inline-flex; align-items:center; gap:2px; height:12px; margin-left:6px;">
        <span style="width:2.5px; height:6px; background:#2ee584; border-radius:1px;"></span>
        <span style="width:2.5px; height:12px; background:#2ee584; border-radius:1px;"></span>
        <span style="width:2.5px; height:8px; background:#2ee584; border-radius:1px;"></span>
      </span>
    """

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
  gap: 14px;
}}
.tile {{
  border-radius: 12px;
  background: #0f141a;
  border: 1px solid #1a232c;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}}
.avatar {{
  width: 76px;
  height: 76px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 28px;
  font-weight: 700;
}}
.tile-name {{
  position: absolute;
  left: 14px;
  bottom: 14px;
  background: rgba(10, 14, 18, 0.82);
  backdrop-filter: blur(10px);
  padding: 5px 12px;
  border-radius: 8px;
  font-size: 12.5px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 6px;
  border: 1px solid rgba(255,255,255,0.06);
}}
.meet-bar {{
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 16px;
}}
.meet-ctrls {{
  display: flex;
  gap: 10px;
  align-items: center;
}}
.ctrl-btn {{
  width: 42px;
  height: 42px;
  border-radius: 50%;
  background: #19212a;
  border: 1px solid #23303d;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--ink);
  cursor: pointer;
  transition: all .2s;
}}
.ctrl-leave {{
  width: 54px;
  height: 42px;
  border-radius: 21px;
  background: #ea4335;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  cursor: pointer;
  box-shadow: 0 3px 12px rgba(234,67,53,0.35);
}}
.side-panel {{
  width: 368px;
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
      <!-- Tile 1 -->
      <div class="tile">
        <div class="avatar" style="background:#132d4a; color:#38bdf8; border: 2px solid rgba(46,229,132,0.6); box-shadow:0 0 18px rgba(46,229,132,0.25);">D</div>
        <div class="tile-name">Derek Gallardo (You) {wave_derek}</div>
      </div>
      <!-- Tile 2 -->
      <div class="tile">
        <div class="avatar" style="background:#311b58; color:#c084fc;">M</div>
        <div class="tile-name">Maya Lin</div>
      </div>
      <!-- Tile 3 -->
      <div class="tile">
        <div class="avatar" style="background:#0c382b; color:#34d399;">A</div>
        <div class="tile-name">Alex Chen</div>
      </div>
      <!-- Tile 4 -->
      <div class="tile" style="{red_border}">
        {avatar_fake}
        {fake_badge}
        <div class="tile-name">{'Unknown caller' if is_fake_call else 'Sarah Jenkins'}</div>
      </div>
    </div>
    <div class="meet-bar">
      <div style="display:flex; align-items:center; gap:8px;">
        <span style="font-size:13.5px; font-weight:700; font-family:var(--mono); color:var(--ink);">11:42 AM</span>
        <span style="color:var(--dim);">|</span>
        <span style="font-size:13px; color:var(--muted); font-family:var(--mono);">wire-approval-q3</span>
      </div>
      <div class="meet-ctrls">
        <div class="ctrl-btn">{ICONS['mic']}</div>
        <div class="ctrl-btn">{ICONS['camera']}</div>
        <div class="ctrl-btn">{ICONS['hand']}</div>
        <div class="ctrl-btn">{ICONS['screen']}</div>
        <div class="ctrl-btn">{ICONS['dots']}</div>
        <div class="ctrl-leave">{ICONS['phone_down']}</div>
      </div>
      <div style="display:flex; gap:10px; align-items:center;">
        <div class="ctrl-btn" style="width:36px; height:36px;">{ICONS['info']}</div>
        <div class="ctrl-btn" style="width:36px; height:36px; gap:4px; font-size:11px; font-weight:700;">{ICONS['people']} <span style="font-size:10px; margin-left:1px;">4</span></div>
        <div class="ctrl-btn" style="width:36px; height:36px;">{ICONS['chat']}</div>
        <div class="ctrl-btn" style="width:36px; height:36px; background:#1b2c22; border-color:rgba(46,229,132,0.4); color:var(--green);">{ICONS['shapes']}</div>
        <div style="display:flex; align-items:center; gap:6px; padding:6px 10px; border-radius:20px; background:rgba(46,229,132,0.12); border:1px solid rgba(46,229,132,0.35);">
          <div style="width:16px; height:16px;">{LOGO_SVG}</div>
          <span style="font-size:11px; font-weight:700; color:var(--green); letter-spacing:0.02em;">Sonave</span>
        </div>
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
    <div style="padding:14px 18px; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; background:#0b0f14;">
      <div style="display:flex; align-items:center; gap:10px;">
        <div style="width:26px; height:26px;">{LOGO_SVG}</div>
        <div>
          <b style="font-size:13.5px; font-weight:800; letter-spacing:-0.01em;">Sonave</b>
          <small style="display:block; font-size:9.5px; color:var(--muted); letter-spacing:0.06em; font-weight:600;">VOICE AUTHENTICITY · THIS CALL</small>
        </div>
      </div>
      <span style="display:inline-flex; align-items:center; gap:5px; font-size:10px; font-family:var(--mono); color:var(--green); background:rgba(46,229,132,0.1); padding:3px 8px; border-radius:6px; border:1px solid rgba(46,229,132,0.25);">
        <span style="width:5px; height:5px; border-radius:50%; background:var(--green);"></span>ENGINE READY
      </span>
    </div>
    
    <div style="padding:28px 20px; display:flex; flex-direction:column; align-items:center; text-align:center; flex:1;">
      <div style="position:relative; width:88px; height:88px; margin-bottom:18px;">
        <div style="position:absolute; inset:0; border:2px solid rgba(46,229,132,0.25); border-radius:50%;"></div>
        <div style="position:absolute; inset:14px; border:2px solid rgba(46,229,132,0.6); border-radius:50%; box-shadow:0 0 20px rgba(46,229,132,0.3);"></div>
        <div style="position:absolute; inset:28px; width:32px; height:32px;">{LOGO_SVG}</div>
      </div>
      
      <div style="font-size:18px; font-weight:800; letter-spacing:-0.01em; margin-bottom:6px;">Voice verification ready</div>
      <p style="font-size:12.5px; color:var(--ink2); line-height:1.55; max-width:290px; margin-bottom:22px;">
        Every speaker in this meeting is scored live against fine-tuned acoustic detection models.
      </p>
      
      <button style="width:100%; padding:12px; background:var(--green); color:#06130b; border:0; border-radius:10px; font-weight:800; font-size:13.5px; margin-bottom:20px; box-shadow:0 6px 24px rgba(46,229,132,0.3); cursor:pointer;">
        Start voice verification
      </button>
      
      <div style="width:100%; display:flex; flex-direction:column; gap:10px; text-align:left;">
        <div style="display:flex; gap:12px; align-items:flex-start; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:11px 14px;">
          <b style="width:20px; height:20px; border-radius:50%; background:rgba(46,229,132,0.15); color:var(--green); font-size:11px; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:1px;">1</b>
          <span style="font-size:11.5px; color:var(--ink2); line-height:1.5;"><strong style="color:var(--ink); font-weight:700;">Real-time per-speaker analysis</strong> — voices scored while speaking</span>
        </div>
        <div style="display:flex; gap:12px; align-items:flex-start; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:11px 14px;">
          <b style="width:20px; height:20px; border-radius:50%; background:rgba(46,229,132,0.15); color:var(--green); font-size:11px; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:1px;">2</b>
          <span style="font-size:11.5px; color:var(--ink2); line-height:1.5;"><strong style="color:var(--ink); font-weight:700;">4-second window latency</strong> — rolling verdicts refreshed continuously</span>
        </div>
        <div style="display:flex; gap:12px; align-items:flex-start; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:11px 14px;">
          <b style="width:20px; height:20px; border-radius:50%; background:rgba(46,229,132,0.15); color:var(--green); font-size:11px; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:1px;">3</b>
          <span style="font-size:11.5px; color:var(--ink2); line-height:1.5;"><strong style="color:var(--ink); font-weight:700;">Automated wire-hold protection</strong> — webhook pause & compliance PDF</span>
        </div>
      </div>
      
      <div style="margin-top:20px;">
        <span style="color:var(--green); font-size:11.5px; font-weight:600; cursor:pointer;">&#9654; Watch a 15-second simulated detection</span>
      </div>
    </div>
    
    <div style="padding:12px 16px; border-top:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; font-size:11px; color:var(--muted); background:#0b0f14;">
      <span style="display:flex; align-items:center; gap:6px;"><span style="width:6px; height:6px; border-radius:50%; background:var(--green);"></span>detection engine online</span>
      <span style="color:var(--green); font-weight:700; cursor:pointer;">Open console ↗</span>
    </div>
    """
    return make_meet_frame(panel_body, is_fake_call=False)

def get_shot_meet_panel_html():
    panel_body = f"""
    <div style="padding:14px 18px; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; background:#0b0f14;">
      <div style="display:flex; align-items:center; gap:10px;">
        <div style="width:26px; height:26px;">{LOGO_SVG}</div>
        <div>
          <b style="font-size:13.5px; font-weight:800; letter-spacing:-0.01em;">Sonave</b>
          <small style="display:block; font-size:9.5px; color:var(--muted); letter-spacing:0.06em; font-weight:600;">VOICE AUTHENTICITY · THIS CALL</small>
        </div>
      </div>
      <span style="display:inline-flex; align-items:center; gap:5px; font-size:10px; font-family:var(--mono); color:var(--green); background:rgba(46,229,132,0.1); padding:3px 8px; border-radius:6px; border:1px solid rgba(46,229,132,0.25);">
        <span style="width:5px; height:5px; border-radius:50%; background:var(--green);"></span>STREAMING
      </span>
    </div>

    <!-- Top Room Verdict -->
    <div style="padding:14px 18px; border-bottom:1px solid var(--line); background:rgba(46,229,132,0.06); display:flex; justify-content:space-between; align-items:center;">
      <div>
        <div style="font-size:20px; font-weight:900; color:var(--green); letter-spacing:0.02em; display:flex; align-items:center; gap:8px;">
          REAL <span style="font-size:11px; padding:2px 7px; border-radius:4px; background:rgba(46,229,132,0.15); color:var(--green); font-weight:700; font-family:var(--mono);">100% ROOM INTEGRITY</span>
        </div>
        <div style="font-size:11.5px; color:var(--ink2); margin-top:3px;">All participants within authentic biological range.</div>
      </div>
    </div>

    <!-- Speaker List -->
    <div style="padding:12px 16px; display:flex; flex-direction:column; gap:10px; flex:1;">
      <!-- Speaker 1 (Derek) -->
      <div style="border:1px solid rgba(46,229,132,0.3); border-radius:12px; padding:12px 14px; background:#0f151c; box-shadow:0 0 15px rgba(46,229,132,0.06);">
        <div style="display:flex; align-items:center; gap:10px;">
          <div style="width:32px; height:32px; border-radius:50%; background:#132d4a; border:1.5px solid var(--green); display:flex; align-items:center; justify-content:center; font-size:13px; font-weight:700; color:#38bdf8;">D</div>
          <div style="flex:1; min-width:0;">
            <div style="font-size:13px; font-weight:700;">Derek Gallardo <span style="color:var(--muted); font-weight:normal; font-size:11px;">· you</span></div>
            <div style="font-size:10.5px; color:var(--green); margin-top:1px; display:flex; align-items:center; gap:5px;">
              <span style="width:5px; height:5px; border-radius:50%; background:var(--green);"></span> speaking · 4.2s window
            </div>
          </div>
          <span style="font-family:var(--mono); font-size:16px; font-weight:800; color:var(--green);">4%</span>
        </div>
        <div style="height:4px; background:#18222b; border-radius:2px; margin-top:10px; overflow:hidden;">
          <div style="height:100%; width:4%; background:var(--green); border-radius:2px;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--muted); margin-top:8px; font-family:var(--mono);">
          <span>94% speech</span><span>14:20 monitored</span><span style="font-weight:700; color:var(--green);">REAL</span>
        </div>
      </div>

      <!-- Speaker 2 (Maya) -->
      <div style="border:1px solid var(--line); border-radius:12px; padding:12px 14px; background:var(--panel);">
        <div style="display:flex; align-items:center; gap:10px;">
          <div style="width:32px; height:32px; border-radius:50%; background:#311b58; border:1px solid #4a2882; display:flex; align-items:center; justify-content:center; font-size:13px; font-weight:700; color:#c084fc;">M</div>
          <div style="flex:1; min-width:0;">
            <div style="font-size:13px; font-weight:700;">Maya Lin</div>
            <div style="font-size:10.5px; color:var(--muted); margin-top:1px;">muted / quiet · 12s</div>
          </div>
          <span style="font-family:var(--mono); font-size:16px; font-weight:800; color:var(--green);">8%</span>
        </div>
        <div style="height:4px; background:#18222b; border-radius:2px; margin-top:10px; overflow:hidden;">
          <div style="height:100%; width:8%; background:var(--green); border-radius:2px;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--muted); margin-top:8px; font-family:var(--mono);">
          <span>68% speech</span><span>12:45 monitored</span><span style="font-weight:700; color:var(--green);">REAL</span>
        </div>
      </div>

      <!-- Speaker 3 (Alex) -->
      <div style="border:1px solid var(--line); border-radius:12px; padding:12px 14px; background:var(--panel);">
        <div style="display:flex; align-items:center; gap:10px;">
          <div style="width:32px; height:32px; border-radius:50%; background:#0c382b; border:1px solid #145945; display:flex; align-items:center; justify-content:center; font-size:13px; font-weight:700; color:#34d399;">A</div>
          <div style="flex:1; min-width:0;">
            <div style="font-size:13px; font-weight:700;">Alex Chen</div>
            <div style="font-size:10.5px; color:var(--muted); margin-top:1px;">listening</div>
          </div>
          <span style="font-family:var(--mono); font-size:16px; font-weight:800; color:var(--green);">12%</span>
        </div>
        <div style="height:4px; background:#18222b; border-radius:2px; margin-top:10px; overflow:hidden;">
          <div style="height:100%; width:12%; background:var(--green); border-radius:2px;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--muted); margin-top:8px; font-family:var(--mono);">
          <span>45% speech</span><span>08:10 monitored</span><span style="font-weight:700; color:var(--green);">REAL</span>
        </div>
      </div>
    </div>

    <!-- Live Metrics Summary -->
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; padding:0 16px 14px;">
      <div style="border:1px solid var(--line); background:var(--panel); border-radius:10px; padding:9px 12px;">
        <b style="display:block; font-size:8.5px; letter-spacing:0.08em; color:var(--muted); font-weight:700; margin-bottom:3px;">CHECKS</b>
        <span style="font-family:var(--mono); font-size:14px; font-weight:700;">42</span>
      </div>
      <div style="border:1px solid var(--line); background:var(--panel); border-radius:10px; padding:9px 12px;">
        <b style="display:block; font-size:8.5px; letter-spacing:0.08em; color:var(--muted); font-weight:700; margin-bottom:3px;">SESSION</b>
        <span style="font-family:var(--mono); font-size:14px; font-weight:700;">14:20</span>
      </div>
      <div style="border:1px solid var(--line); background:var(--panel); border-radius:10px; padding:9px 12px;">
        <b style="display:block; font-size:8.5px; letter-spacing:0.08em; color:var(--muted); font-weight:700; margin-bottom:3px;">PEAK RISK</b>
        <span style="font-family:var(--mono); font-size:14px; font-weight:700; color:var(--green);">12%</span>
      </div>
    </div>

    <div style="padding:12px 16px; border-top:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; font-size:11px; color:var(--muted); background:#0b0f14;">
      <span style="display:flex; align-items:center; gap:6px;"><span style="width:6px; height:6px; border-radius:50%; background:var(--green);"></span>detection engine online</span>
      <span style="color:var(--green); font-weight:700; cursor:pointer;">Open console ↗</span>
    </div>
    """
    return make_meet_frame(panel_body, is_fake_call=False)

def get_shot_wire_hold_html():
    panel_body = f"""
    <div style="padding:14px 18px; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; background:#0b0f14;">
      <div style="display:flex; align-items:center; gap:10px;">
        <div style="width:26px; height:26px;">{LOGO_SVG}</div>
        <div>
          <b style="font-size:13.5px; font-weight:800; letter-spacing:-0.01em;">Sonave</b>
          <small style="display:block; font-size:9.5px; color:var(--muted); letter-spacing:0.06em; font-weight:600;">VOICE AUTHENTICITY · THIS CALL</small>
        </div>
      </div>
      <span style="display:inline-flex; align-items:center; gap:5px; font-size:10px; font-family:var(--mono); color:var(--red); background:rgba(255,77,94,0.12); padding:3px 8px; border-radius:6px; border:1px solid rgba(255,77,94,0.35);">
        <span style="width:5px; height:5px; border-radius:50%; background:var(--red);"></span>INCIDENT ACTIVE
      </span>
    </div>

    <!-- Alert Banner -->
    <div style="padding:14px 18px; border-bottom:1px solid rgba(255,77,94,0.35); background:rgba(255,77,94,0.12);">
      <div style="font-size:20px; font-weight:900; color:var(--red); letter-spacing:0.02em; display:flex; align-items:center; gap:8px;">
        FAKE <span style="font-size:10.5px; padding:2px 7px; border-radius:4px; background:rgba(255,77,94,0.25); color:#fff; font-weight:800; font-family:var(--mono);">3 RED WINDOWS</span>
      </div>
      <div style="font-size:11.5px; color:#ff949e; margin-top:3px;">Caller flagged as synthetic voice clone (94% confidence).</div>
    </div>

    <!-- Wire Hold Action Box -->
    <div style="margin:12px 16px 0; padding:14px; border-radius:12px; border:1.5px solid rgba(255,77,94,0.5); background:rgba(255,77,94,0.08); box-shadow:0 4px 20px rgba(255,77,94,0.15);">
      <div style="display:flex; align-items:center; gap:8px;">
        {ICONS['shield_alert']}
        <span style="font-size:13px; font-weight:800; color:#ff808d;">Wire hold recommended</span>
      </div>
      <div style="font-size:11.5px; color:var(--ink2); margin-top:6px; line-height:1.5;">
        "Unknown caller" crossed synthetic threshold for 3 consecutive windows. Payment approval webhook triggered.
      </div>
      <div style="display:flex; gap:10px; margin-top:12px;">
        <button style="flex:1; padding:9px; border-radius:8px; border:0; background:var(--red); color:#fff; font-weight:800; font-size:12px; cursor:pointer; box-shadow:0 3px 12px rgba(255,77,94,0.4);">
          Hold the wire
        </button>
        <button style="padding:9px 12px; border-radius:8px; border:1px solid rgba(255,255,255,0.12); background:rgba(255,255,255,0.05); color:var(--ink); font-weight:700; font-size:12px; cursor:pointer;">
          Forensic report
        </button>
      </div>
    </div>

    <!-- Speaker Cards -->
    <div style="padding:12px 16px; display:flex; flex-direction:column; gap:10px; flex:1;">
      <!-- Fake Caller -->
      <div style="border:1.5px solid rgba(255,77,94,0.55); border-radius:12px; padding:12px 14px; background:rgba(255,77,94,0.06); box-shadow:0 0 20px rgba(255,77,94,0.1);">
        <div style="display:flex; align-items:center; gap:10px;">
          <div style="width:32px; height:32px; border-radius:50%; background:#380b0f; border:1.5px solid var(--red); display:flex; align-items:center; justify-content:center; font-size:13px; font-weight:800; color:var(--red);">U</div>
          <div style="flex:1; min-width:0;">
            <div style="font-size:13px; font-weight:800; color:#ff8a95;">Unknown caller</div>
            <div style="font-size:10.5px; color:var(--red); margin-top:1px; font-weight:600;">speaking · synthetic clone</div>
          </div>
          <span style="font-family:var(--mono); font-size:16px; font-weight:800; color:var(--red);">94%</span>
        </div>
        <div style="height:4px; background:#241114; border-radius:2px; margin-top:10px; overflow:hidden;">
          <div style="height:100%; width:94%; background:var(--red); border-radius:2px;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--muted); margin-top:8px; font-family:var(--mono);">
          <span>52% speech</span><span>06:18 monitored</span><span style="font-weight:800; color:var(--red);">FAKE</span>
        </div>
        <div style="margin-top:8px; padding:5px 8px; border-radius:6px; background:rgba(255,77,94,0.15); border:1px solid rgba(255,77,94,0.3); font-size:10px; color:#ffb3ba; font-family:var(--mono);">
          Artifact: Neural TTS / Cross-lingual clone · Opus 16kHz
        </div>
      </div>

      <!-- Derek (Authentic) -->
      <div style="border:1px solid var(--line); border-radius:12px; padding:12px 14px; background:var(--panel);">
        <div style="display:flex; align-items:center; gap:10px;">
          <div style="width:32px; height:32px; border-radius:50%; background:#132d4a; border:1px solid #1a3c63; display:flex; align-items:center; justify-content:center; font-size:13px; font-weight:700; color:#38bdf8;">D</div>
          <div style="flex:1; min-width:0;">
            <div style="font-size:13px; font-weight:700;">Derek Gallardo <span style="color:var(--muted); font-weight:normal; font-size:11px;">· you</span></div>
            <div style="font-size:10.5px; color:var(--muted); margin-top:1px;">listening</div>
          </div>
          <span style="font-family:var(--mono); font-size:16px; font-weight:800; color:var(--green);">4%</span>
        </div>
        <div style="height:4px; background:#18222b; border-radius:2px; margin-top:10px; overflow:hidden;">
          <div style="height:100%; width:4%; background:var(--green); border-radius:2px;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--muted); margin-top:8px; font-family:var(--mono);">
          <span>88% speech</span><span>18:40 monitored</span><span style="font-weight:700; color:var(--green);">REAL</span>
        </div>
      </div>
    </div>

    <!-- Stats -->
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; padding:0 16px 14px;">
      <div style="border:1px solid var(--line); background:var(--panel); border-radius:10px; padding:9px 12px;">
        <b style="display:block; font-size:8.5px; letter-spacing:0.08em; color:var(--muted); font-weight:700; margin-bottom:3px;">CHECKS</b>
        <span style="font-family:var(--mono); font-size:14px; font-weight:700;">68</span>
      </div>
      <div style="border:1px solid var(--line); background:var(--panel); border-radius:10px; padding:9px 12px;">
        <b style="display:block; font-size:8.5px; letter-spacing:0.08em; color:var(--muted); font-weight:700; margin-bottom:3px;">SESSION</b>
        <span style="font-family:var(--mono); font-size:14px; font-weight:700;">18:40</span>
      </div>
      <div style="border:1px solid var(--line); background:var(--panel); border-radius:10px; padding:9px 12px;">
        <b style="display:block; font-size:8.5px; letter-spacing:0.08em; color:var(--muted); font-weight:700; margin-bottom:3px;">PEAK RISK</b>
        <span style="font-family:var(--mono); font-size:14px; font-weight:700; color:var(--red);">94%</span>
      </div>
    </div>

    <div style="padding:12px 16px; border-top:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; font-size:11px; color:var(--muted); background:#0b0f14;">
      <span style="display:flex; align-items:center; gap:6px;"><span style="width:6px; height:6px; border-radius:50%; background:var(--red);"></span>incident recorded #8402</span>
      <span style="color:var(--green); font-weight:700; cursor:pointer;">Open console ↗</span>
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
  background: #080b0e;
}}
.sidebar {{
  width: 240px;
  border-right: 1px solid var(--line);
  padding: 22px 16px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  background: #0a0e13;
}}
.nav-item {{
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  border-radius: 9px;
  font-size: 13.5px;
  font-weight: 600;
  color: var(--ink2);
  cursor: pointer;
}}
.nav-item.active {{
  background: #141d26;
  color: var(--green);
}}
.content {{
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 26px 36px;
  gap: 22px;
  overflow: hidden;
}}
.stat-card {{
  border: 1px solid var(--line);
  background: #0e141b;
  border-radius: 12px;
  padding: 18px 20px;
  flex: 1;
  border-top: 2.5px solid rgba(46,229,132,0.4);
}}
</style>
</head>
<body>
<div class="console-layout">
  <div class="sidebar">
    <div style="display:flex; align-items:center; gap:10px; padding:4px 6px 22px;">
      <div style="width:32px; height:32px;">{LOGO_SVG}</div>
      <div>
        <b style="font-size:16px; font-weight:800; letter-spacing:-0.01em;">Sonave</b>
        <small style="display:block; font-size:9.5px; color:var(--muted); letter-spacing:0.06em; font-weight:700;">ENTERPRISE CONSOLE</small>
      </div>
    </div>
    
    <div class="nav-item active">{ICONS['monitor']} <span>Live Monitor</span></div>
    <div class="nav-item">{ICONS['clock']} <span>Session History</span></div>
    <div class="nav-item">{ICONS['alert_triangle']} <span>Incidents & Alerts</span></div>
    <div class="nav-item">{ICONS['mic_nav']} <span>Voiceprints</span></div>
    <div class="nav-item">{ICONS['zap_nav']} <span>Webhooks & API</span></div>
    <div class="nav-item">{ICONS['settings_nav']} <span>Settings & Billing</span></div>
    
    <div style="flex:1;"></div>
    
    <div style="border:1px solid var(--line); border-radius:12px; padding:14px; background:#0c1117;">
      <div style="font-size:9.5px; color:var(--muted); letter-spacing:0.08em; font-weight:700;">DEPLOYED CHECKPOINT</div>
      <div style="font-family:var(--mono); font-size:12.5px; font-weight:700; color:var(--green); margin-top:5px;">sonave-xlsr-meet-v2</div>
      <div style="display:flex; align-items:center; gap:6px; font-size:11.5px; color:var(--ink2); margin-top:6px;">
        <span style="width:6px; height:6px; border-radius:50%; background:var(--green);"></span>
        <span>95.2% catch on unseen TTS</span>
      </div>
    </div>
  </div>

  <div class="content">
    <div style="display:flex; align-items:center; justify-content:space-between;">
      <div>
        <h1 style="font-size:22px; font-weight:800; letter-spacing:-0.01em;">Realtime Call Forensics & Incident Log</h1>
        <p style="font-size:13px; color:var(--muted); margin-top:3px;">Continuous speaker authentication across active Google Meet conferences.</p>
      </div>
      <button style="padding:10px 18px; background:var(--green); color:#06130b; font-weight:800; font-size:12.5px; border:0; border-radius:9px; display:flex; align-items:center; gap:8px; cursor:pointer;">
        {ICONS['shield_check']} Export Forensic Audit Report
      </button>
    </div>

    <!-- Stats Row -->
    <div style="display:flex; gap:16px;">
      <div class="stat-card">
        <div style="font-size:11px; color:var(--muted); font-weight:700; letter-spacing:0.06em;">ACTIVE CONFERENCES</div>
        <div style="font-size:26px; font-weight:800; font-family:var(--mono); margin-top:8px;">1</div>
        <div style="font-size:11.5px; color:var(--green); margin-top:5px; display:flex; align-items:center; gap:5px;">
          <span style="width:5px; height:5px; border-radius:50%; background:var(--green);"></span> Live WebRTC stream
        </div>
      </div>
      <div class="stat-card">
        <div style="font-size:11px; color:var(--muted); font-weight:700; letter-spacing:0.06em;">AUTHENTICITY VERDICTS</div>
        <div style="font-size:26px; font-weight:800; font-family:var(--mono); margin-top:8px;">1,420</div>
        <div style="font-size:11.5px; color:var(--ink2); margin-top:5px;">4.2s median latency</div>
      </div>
      <div class="stat-card" style="border-top-color:var(--red);">
        <div style="font-size:11px; color:var(--muted); font-weight:700; letter-spacing:0.06em;">INCIDENTS FLAGGED</div>
        <div style="font-size:26px; font-weight:800; font-family:var(--mono); color:var(--red); margin-top:8px;">1</div>
        <div style="font-size:11.5px; color:#ff8a95; margin-top:5px;">Wire hold auto-triggered</div>
      </div>
      <div class="stat-card">
        <div style="font-size:11px; color:var(--muted); font-weight:700; letter-spacing:0.06em;">MONITORING BALANCE</div>
        <div style="font-size:26px; font-weight:800; font-family:var(--mono); margin-top:8px;">5.0 <span style="font-size:15px; color:var(--muted); font-weight:normal;">hrs</span></div>
        <div style="font-size:11.5px; color:var(--green); margin-top:5px;">Free tier active</div>
      </div>
    </div>

    <!-- Recent Session Table -->
    <div style="border:1px solid var(--line); background:#0e141b; border-radius:14px; overflow:hidden; flex:1; display:flex; flex-direction:column;">
      <div style="padding:15px 22px; border-bottom:1px solid var(--line); font-size:13.5px; font-weight:700; display:flex; justify-content:space-between; align-items:center;">
        <span>Recent Meeting Activity & Risk Analysis</span>
        <span style="color:var(--green); font-size:12px; display:flex; align-items:center; gap:6px;">
          <span style="width:6px; height:6px; border-radius:50%; background:var(--green);"></span> Auto-refreshing
        </span>
      </div>
      
      <div style="display:grid; grid-template-columns:1.6fr 1fr 1fr 1fr 1.1fr; padding:11px 22px; border-bottom:1px solid var(--line); font-size:11px; color:var(--muted); font-weight:700; letter-spacing:0.06em; background:#0b1016;">
        <span>MEETING / TOPIC</span><span>DURATION</span><span>SPEAKERS</span><span>PEAK RISK</span><span>VERDICT</span>
      </div>

      <!-- Row 1 (Held Incident) -->
      <div style="display:grid; grid-template-columns:1.6fr 1fr 1fr 1fr 1.1fr; padding:15px 22px; border-bottom:1px solid var(--line); font-size:13px; align-items:center; background:rgba(255,77,94,0.04);">
        <div>
          <b style="color:var(--ink); font-size:13.5px;">wire-approval-q3</b>
          <small style="display:block; color:var(--muted); font-size:11px; margin-top:2px;">Google Meet · today 11:42 AM</small>
        </div>
        <span style="font-family:var(--mono); font-size:12.5px;">18m 40s</span>
        <span style="font-size:12.5px;">4 participants</span>
        <span style="font-family:var(--mono); color:var(--red); font-weight:800; font-size:13.5px;">94%</span>
        <span style="padding:4px 10px; border-radius:6px; background:rgba(255,77,94,0.18); color:var(--red); font-weight:800; font-size:11px; width:fit-content; border:1px solid rgba(255,77,94,0.35);">
          FAKE (HELD)
        </span>
      </div>

      <!-- Row 2 (Real) -->
      <div style="display:grid; grid-template-columns:1.6fr 1fr 1fr 1fr 1.1fr; padding:15px 22px; border-bottom:1px solid var(--line); font-size:13px; align-items:center;">
        <div>
          <b style="color:var(--ink); font-size:13.5px;">weekly-exec-standup</b>
          <small style="display:block; color:var(--muted); font-size:11px; margin-top:2px;">Google Meet · yesterday 2:00 PM</small>
        </div>
        <span style="font-family:var(--mono); font-size:12.5px;">45m 12s</span>
        <span style="font-size:12.5px;">6 participants</span>
        <span style="font-family:var(--mono); color:var(--green); font-weight:800; font-size:13.5px;">6%</span>
        <span style="padding:4px 10px; border-radius:6px; background:rgba(46,229,132,0.15); color:var(--green); font-weight:800; font-size:11px; width:fit-content; border:1px solid rgba(46,229,132,0.3);">
          REAL
        </span>
      </div>

      <!-- Row 3 (Real) -->
      <div style="display:grid; grid-template-columns:1.6fr 1fr 1fr 1fr 1.1fr; padding:15px 22px; font-size:13px; align-items:center;">
        <div>
          <b style="color:var(--ink); font-size:13.5px;">client-vendor-onboarding</b>
          <small style="display:block; color:var(--muted); font-size:11px; margin-top:2px;">Google Meet · Aug 16 10:30 AM</small>
        </div>
        <span style="font-family:var(--mono); font-size:12.5px;">32m 05s</span>
        <span style="font-size:12.5px;">3 participants</span>
        <span style="font-family:var(--mono); color:var(--green); font-weight:800; font-size:13.5px;">9%</span>
        <span style="padding:4px 10px; border-radius:6px; background:rgba(46,229,132,0.15); color:var(--green); font-weight:800; font-size:11px; width:fit-content; border:1px solid rgba(46,229,132,0.3);">
          REAL
        </span>
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
  background: radial-gradient(circle at 50% 12%, #122118 0%, #07090c 58%);
  padding: 28px 56px;
  gap: 36px;
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
  gap: 18px;
  margin-top: 14px;
}}
.badge {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 16px;
  border-radius: 20px;
  border: 1px solid rgba(46,229,132,0.35);
  background: rgba(46,229,132,0.08);
  font-size: 12px;
  font-weight: 800;
  color: var(--green);
  letter-spacing: 0.04em;
}}
.title {{
  font-size: 42px;
  font-weight: 900;
  letter-spacing: -0.025em;
  line-height: 1.15;
  max-width: 820px;
}}
.desc {{
  font-size: 16px;
  color: var(--ink2);
  line-height: 1.6;
  max-width: 680px;
}}
.hero-cta {{
  display: flex;
  gap: 14px;
  margin-top: 4px;
}}
.cards {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 22px;
  margin-top: 8px;
}}
.feat-card {{
  border: 1px solid var(--line);
  background: #0d1217;
  border-radius: 14px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}}
.trust-bar {{
  display: flex;
  justify-content: center;
  gap: 36px;
  font-size: 12px;
  color: var(--muted);
  font-family: var(--mono);
  margin-top: auto;
  padding-top: 10px;
}}
</style>
</head>
<body>
<div class="landing-layout">
  <!-- Nav -->
  <div class="nav">
    <div style="display:flex; align-items:center; gap:12px;">
      <div style="width:32px; height:32px;">{LOGO_SVG}</div>
      <b style="font-size:19px; font-weight:900; letter-spacing:-0.02em;">Sonave</b>
    </div>
    <div style="display:flex; gap:28px; font-size:13.5px; font-weight:600; color:var(--ink2);">
      <span>Product</span>
      <span>How It Works</span>
      <span>Benchmarks</span>
      <span>Pricing</span>
    </div>
    <div style="display:flex; gap:12px;">
      <button style="padding:9px 20px; border-radius:9px; border:1px solid var(--line2); background:transparent; color:var(--ink); font-weight:700; font-size:13px; cursor:pointer;">Sign In</button>
      <button style="padding:9px 20px; border-radius:9px; border:0; background:var(--green); color:#06130b; font-weight:800; font-size:13px; cursor:pointer; box-shadow:0 4px 16px rgba(46,229,132,0.35);">Get Started Free</button>
    </div>
  </div>

  <!-- Hero -->
  <div class="hero">
    <div class="badge"><span style="width:7px; height:7px; border-radius:50%; background:var(--green);"></span> LIVE VOICE AUTHENTICITY FOR MEETINGS</div>
    <div class="title">Stop deepfake voice fraud before the money moves.</div>
    <div class="desc">Sonave verifies speaker authenticity in real time inside Google Meet. Catches AI voice clones in 4 seconds with 95% detection accuracy trained on real meeting codecs.</div>
    <div class="hero-cta">
      <button style="padding:12px 24px; border-radius:10px; border:0; background:var(--green); color:#06130b; font-weight:800; font-size:14px; box-shadow:0 6px 24px rgba(46,229,132,0.3); cursor:pointer;">Start Free — 5 Monitored Hours</button>
      <button style="padding:12px 22px; border-radius:10px; border:1px solid var(--line2); background:#11161d; color:var(--ink); font-weight:700; font-size:14px; cursor:pointer;">Explore Interactive Demo</button>
    </div>
  </div>

  <!-- 3 Value Pillars -->
  <div class="cards">
    <div class="feat-card">
      <div style="width:28px; height:28px; display:flex; align-items:center;">{ICONS['zap']}</div>
      <div style="font-size:16px; font-weight:800;">Sub-4s Realtime Latency</div>
      <div style="font-size:13px; color:var(--muted); line-height:1.55;">Evaluates 4-second audio windows on fine-tuned meeting codecs. Rolling scores refresh continuously without lag.</div>
    </div>
    <div class="feat-card">
      <div style="width:28px; height:28px; display:flex; align-items:center;">{ICONS['target']}</div>
      <div style="font-size:16px; font-weight:800;">95.2% Catch Rate</div>
      <div style="font-size:13px; color:var(--muted); line-height:1.55;">Trained against 27 unseen commercial voice generators, neural TTS, diffusion models, and real-time changers.</div>
    </div>
    <div class="feat-card">
      <div style="width:28px; height:28px; display:flex; align-items:center;">{ICONS['shield_check']}</div>
      <div style="font-size:16px; font-weight:800;">Instant Wire-Hold Webhook</div>
      <div style="font-size:13px; color:var(--muted); line-height:1.55;">Automatically triggers payout holds and generates compliance-ready forensic PDF evidence for audit trails.</div>
    </div>
  </div>

  <!-- Trust Badges -->
  <div class="trust-bar">
    <span>✓ Zero Audio Stored After Call</span>
    <span>✓ Encrypted TLS 1.3 In-Flight</span>
    <span>✓ Google Workspace Marketplace Verified</span>
    <span>✓ SOC 2 Type II In Progress</span>
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
            page.wait_for_timeout(350)
            page.screenshot(path=str(out_file))
            print(f"Generated designs/marketplace/{out_file.name} ({out_file.stat().st_size} bytes)")
            
            # Mirror to railway/ for live web access
            if RAILWAY_DIR.exists():
                railway_target = RAILWAY_DIR / name
                shutil.copy2(out_file, railway_target)
                print(f"Mirrored to railway/{railway_target.name}")
            
        browser.close()

if __name__ == "__main__":
    generate_all_screenshots()
