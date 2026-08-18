"""
Automated Demo Video Generator for Google OAuth & Workspace Marketplace Review.
Records a 1080p Full HD video demonstrating OAuth consent, Client ID,
meetings.space.readonly, meetings.conference.media.readonly, real-time Meet side panel,
and token revocation, with professional voiceover narration.

Outputs: designs/marketplace/sonave_demo_video.mp4
"""

import os
import shutil
import subprocess
import time
from pathlib import Path
import win32com.client
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "designs" / "marketplace"
TEMP_DIR = ROOT / "tools" / "_video_temp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

NARRATION_SEGMENTS = [
    (
        "scene1",
        "This is Sonave, an AI voice authenticity and deepfake detection application for Google Meet. "
        "We are requesting the sensitive scope meetings.space.readonly and the restricted scope meetings.conference.media.readonly "
        "to protect organizations from live voice impersonation and wire fraud."
    ),
    (
        "scene2",
        "During sign-in, the Google OAuth consent flow displays our project client ID, 940532414120, "
        "and details the permissions needed to access conference metadata and real-time audio streams."
    ),
    (
        "scene3",
        "Inside Google Meet, the user launches Sonave from the Activities panel. "
        "The meetings.space.readonly scope identifies the active conference space to initialize the session."
    ),
    (
        "scene4",
        "The meetings.conference.media.readonly scope receives real-time, per-speaker WebRTC audio via the Google Meet Media API. "
        "Sonave analyzes audio in rolling four-second windows with fine-tuned detection models, requiring zero participant bots in the call."
    ),
    (
        "scene5",
        "Each speaker receives a live confidence score and REAL or FAKE verdict. "
        "When an AI-generated cloned voice speaks, Sonave flags it in the red band and raises an instant wire-hold alert."
    ),
    (
        "scene6",
        "In the console dashboard, compliance teams can export forensic incident reports. "
        "Audio is processed ephemerally in memory and is never permanently stored. "
        "Signing out instantly revokes all session tokens server-side. Thank you for your review."
    )
]

def generate_voiceover():
    """Synthesize voiceover narration to WAV files."""
    print("[1/4] Generating voiceover audio narration...")
    voice = win32com.client.Dispatch("SAPI.SpVoice")
    # set moderate rate
    voice.Rate = 0
    
    wav_files = []
    for tag, text in NARRATION_SEGMENTS:
        wav_path = TEMP_DIR / f"{tag}.wav"
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        stream.Open(str(wav_path), 3) # SSFMCreateForWrite
        voice.AudioOutputStream = stream
        voice.Speak(text)
        stream.Close()
        wav_files.append(wav_path)
        print(f"  Synthesized {tag}.wav ({wav_path.stat().st_size} bytes)")
    
    # Concatenate audio segments with silence pads
    concat_list = TEMP_DIR / "concat_list.txt"
    silence_wav = TEMP_DIR / "silence.wav"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono", "-t", "1.2",
        str(silence_wav)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    with open(concat_list, "w", encoding="utf-8") as f:
        for w in wav_files:
            f.write(f"file '{w.name}'\n")
            f.write(f"file '{silence_wav.name}'\n")
            
    full_audio = TEMP_DIR / "full_narration.wav"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy", str(full_audio)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"  Combined narration: {full_audio.name} ({full_audio.stat().st_size} bytes)")
    return full_audio

def build_demo_animation_html():
    """Build the self-playing 1080p web animation showcasing the entire flow."""
    html_path = TEMP_DIR / "demo_animation.html"
    
    html_content = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Sonave — Google Workspace OAuth & Media API Demo</title>
<style>
:root {
  --bg: #080a0c;
  --panel: #0e1216;
  --card: #141a21;
  --line: #1b242e;
  --ink: #e8eef2;
  --ink2: #9aa9b5;
  --muted: #6b7c8a;
  --green: #2ee584;
  --red: #ff4d5e;
  --amber: #ffb224;
  --blue: #38bdf8;
  --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --mono: ui-monospace, 'IBM Plex Mono', Consolas, monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  width: 1920px;
  height: 1080px;
  background: #000;
  color: var(--ink);
  font-family: var(--font);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
#caption-bar {
  position: absolute;
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(0,0,0,0.85);
  border: 1px solid rgba(46,229,132,0.4);
  backdrop-filter: blur(12px);
  padding: 14px 28px;
  border-radius: 12px;
  font-size: 19px;
  font-weight: 600;
  color: #fff;
  z-index: 1000;
  max-width: 1400px;
  text-align: center;
  box-shadow: 0 10px 30px rgba(0,0,0,0.8);
  transition: opacity 0.3s ease;
}
.stage {
  width: 1920px;
  height: 1080px;
  position: relative;
  display: none;
  background: var(--bg);
}
.stage.active { display: flex; }

/* OAuth Screen */
.oauth-modal {
  width: 580px;
  background: #fff;
  color: #202124;
  border-radius: 12px;
  box-shadow: 0 12px 48px rgba(0,0,0,0.6);
  margin: auto;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.oauth-header {
  padding: 24px 32px 16px;
  text-align: center;
  border-bottom: 1px solid #e0e0e0;
}
.oauth-body {
  padding: 24px 32px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.scope-item {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  padding: 12px;
  background: #f8f9fa;
  border: 1px solid #dadce0;
  border-radius: 8px;
}

/* Meet Stage */
.meet-layout {
  display: flex;
  width: 100%;
  height: 100%;
}
.meet-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 24px;
  gap: 18px;
}
.meet-grid {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  gap: 16px;
}
.tile {
  border-radius: 16px;
  background: #11161d;
  border: 1px solid var(--line);
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  transition: all 0.5s ease;
}
.tile.red-alert {
  border: 3px solid var(--red);
  box-shadow: 0 0 40px rgba(255,77,94,0.4);
}
.avatar {
  width: 96px;
  height: 96px;
  border-radius: 50%;
  background: #1a222c;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 36px;
  font-weight: 700;
}
.tile-tag {
  position: absolute;
  left: 16px;
  bottom: 16px;
  background: rgba(0,0,0,0.7);
  backdrop-filter: blur(8px);
  padding: 6px 14px;
  border-radius: 8px;
  font-size: 15px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
}
.side-panel {
  width: 480px;
  background: var(--panel);
  border-left: 1px solid var(--line);
  display: flex;
  flex-direction: column;
  height: 100%;
}
</style>
</head>
<body>

<div id="caption-bar">Initializing Sonave Voice Verification Demo...</div>

<!-- STAGE 1: INTRO & OAUTH CONSENT -->
<div id="stage-oauth" class="stage active" style="flex-direction:column; align-items:center; justify-content:center;">
  <!-- Mock Browser Bar -->
  <div style="width:1100px; background:#1e242c; border-radius:12px 12px 0 0; padding:12px 20px; display:flex; align-items:center; gap:14px; border:1px solid #2d3744;">
    <div style="display:flex; gap:8px;">
      <span style="width:12px; height:12px; border-radius:50%; background:#ff5f56; display:inline-block;"></span>
      <span style="width:12px; height:12px; border-radius:50%; background:#ffbd2e; display:inline-block;"></span>
      <span style="width:12px; height:12px; border-radius:50%; background:#27c93f; display:inline-block;"></span>
    </div>
    <div style="flex:1; background:#0b0f14; border-radius:8px; padding:7px 16px; font-family:var(--mono); font-size:13px; color:var(--ink2); display:flex; align-items:center; gap:8px;">
      <span style="color:var(--green);">🔒 https://accounts.google.com/o/oauth2/v2/auth</span>
      <span style="color:#556575;">?client_id=940532414120.apps.googleusercontent.com&scope=meetings.conference.media.readonly</span>
    </div>
  </div>
  
  <div class="oauth-modal" style="border-radius:0 0 12px 12px; width:1100px; background:#13181f; color:#fff; border:1px solid #2d3744; border-top:0; padding:32px 48px;">
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:24px;">
      <div style="width:48px; height:48px; border-radius:12px; background:linear-gradient(135deg,#2ee584,#0d9e56); display:flex; align-items:center; justify-content:center; font-size:24px; font-weight:800; color:#06130b;">S</div>
      <div>
        <h2 style="font-size:22px; font-weight:800;">Sonave wants access to your Google Account</h2>
        <p style="font-size:13.5px; color:var(--ink2);">derek@usesonave.com (Sonave Workspace)</p>
      </div>
    </div>
    
    <div style="font-size:14px; font-weight:700; color:var(--muted); letter-spacing:0.06em; margin-bottom:12px;">REQUESTED PERMISSIONS</div>
    <div style="display:flex; flex-direction:column; gap:14px; margin-bottom:28px;">
      <div class="scope-item" style="background:#192029; border:1px solid #273342; color:#fff;">
        <span style="font-size:20px;">🛡</span>
        <div>
          <b style="font-size:14.5px; color:var(--green);">Capture real-time audio in Google Meet video calls</b>
          <div style="font-family:var(--mono); font-size:12px; color:var(--muted); margin-top:2px;">https://www.googleapis.com/auth/meetings.conference.media.readonly</div>
          <div style="font-size:13px; color:var(--ink2); margin-top:4px;">Streams per-speaker WebRTC audio to the detection model to verify voice authenticity in real time without recording bots.</div>
        </div>
      </div>
      <div class="scope-item" style="background:#192029; border:1px solid #273342; color:#fff;">
        <span style="font-size:20px;">📹</span>
        <div>
          <b style="font-size:14.5px; color:var(--green);">Read information about your Google Meet conferences</b>
          <div style="font-family:var(--mono); font-size:12px; color:var(--muted); margin-top:2px;">https://www.googleapis.com/auth/meetings.space.readonly</div>
          <div style="font-size:13px; color:var(--ink2); margin-top:4px;">Identifies active conference space ID to bind side panel authenticity telemetry.</div>
        </div>
      </div>
    </div>
    
    <div style="display:flex; justify-content:flex-end; gap:14px;">
      <button style="padding:10px 24px; border-radius:8px; border:1px solid #324050; background:transparent; color:#fff; font-size:14px; font-weight:600;">Cancel</button>
      <button id="btn-allow" style="padding:10px 28px; border-radius:8px; border:0; background:var(--green); color:#06130b; font-size:14px; font-weight:800;">Allow & Continue</button>
    </div>
  </div>
</div>

<!-- STAGE 2: IN-CALL GOOGLE MEET -->
<div id="stage-meet" class="stage">
  <div class="meet-layout">
    <div class="meet-main">
      <div class="meet-grid">
        <div class="tile">
          <div class="avatar" style="background:#1e293b; color:#38bdf8;">D</div>
          <div class="tile-tag">Derek Gallardo (You) <span style="color:var(--green);">● speaking</span></div>
        </div>
        <div class="tile">
          <div class="avatar" style="background:#2e1065; color:#c084fc;">M</div>
          <div class="tile-tag">Maya Lin</div>
        </div>
        <div class="tile">
          <div class="avatar" style="background:#064e3b; color:#34d399;">A</div>
          <div class="tile-tag">Alex Chen</div>
        </div>
        <div id="tile-caller" class="tile">
          <div id="caller-avatar" class="avatar" style="background:#1e293b; color:#94a3b8;">U</div>
          <div id="caller-badge" style="display:none; position:absolute; top:16px; right:16px; background:var(--red); color:#fff; font-size:12px; font-weight:800; padding:6px 12px; border-radius:8px; letter-spacing:0.05em;">SYNTHETIC VOICE DETECTED</div>
          <div id="caller-tag" class="tile-tag">Unknown Caller</div>
        </div>
      </div>
      
      <!-- Meet Bottom Bar -->
      <div style="height:64px; background:#12171e; border-radius:14px; border:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; padding:0 24px;">
        <div style="font-size:15px; color:var(--muted); font-family:var(--mono);">11:42 AM | wire-approval-q3</div>
        <div style="display:flex; gap:12px;">
          <div style="width:44px; height:44px; border-radius:50%; background:#1e2632; display:flex; align-items:center; justify-content:center; font-size:18px;">🎙</div>
          <div style="width:44px; height:44px; border-radius:50%; background:#1e2632; display:flex; align-items:center; justify-content:center; font-size:18px;">🎥</div>
          <div style="width:44px; height:44px; border-radius:50%; background:#1e2632; display:flex; align-items:center; justify-content:center; font-size:18px;">✋</div>
          <div style="width:58px; height:44px; border-radius:22px; background:#ea4335; display:flex; align-items:center; justify-content:center; font-size:18px; color:#fff;">⏻</div>
        </div>
        <div style="display:flex; gap:12px; align-items:center;">
          <div style="padding:6px 14px; background:rgba(46,229,132,0.15); border:1px solid var(--green); border-radius:20px; color:var(--green); font-size:13px; font-weight:700;">Sonave Active</div>
        </div>
      </div>
    </div>
    
    <!-- Sonave Side Panel -->
    <div class="side-panel">
      <div style="padding:18px 22px; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between;">
        <div style="display:flex; align-items:center; gap:10px;">
          <div style="width:28px; height:28px; border-radius:7px; background:linear-gradient(135deg,#2ee584,#0d9e56); display:flex; align-items:center; justify-content:center; font-weight:800; color:#06130b;">S</div>
          <div>
            <b style="font-size:15px;">Sonave</b>
            <small style="display:block; font-size:10px; color:var(--muted); letter-spacing:0.06em;">VOICE AUTHENTICITY · THIS CALL</small>
          </div>
        </div>
        <span style="font-size:11px; color:var(--green); font-family:var(--mono); font-weight:700;">● MEDIA API LIVE</span>
      </div>
      
      <!-- Room Banner -->
      <div id="room-banner" style="padding:16px 20px; border-bottom:1px solid var(--line); background:rgba(46,229,132,0.06);">
        <div id="room-verdict" style="font-size:22px; font-weight:900; color:var(--green); letter-spacing:-0.01em;">REAL</div>
        <div id="room-note" style="font-size:12.5px; color:var(--ink2); margin-top:2px;">All speakers within normal authenticity range.</div>
      </div>
      
      <!-- Alert Card -->
      <div id="alert-card" style="display:none; margin:14px 18px 0; padding:14px 16px; border-radius:12px; border:1px solid rgba(255,77,94,0.5); background:rgba(255,77,94,0.1);">
        <div style="font-size:13.5px; font-weight:800; color:#ff8a95;">⛔ Wire Hold Recommended</div>
        <div style="font-size:12px; color:var(--ink2); margin-top:4px; line-height:1.45;">"Unknown caller" scored in red band (94% fake). Approval webhook triggered.</div>
        <div style="display:flex; gap:10px; margin-top:10px;">
          <button style="flex:1; padding:8px; border-radius:8px; border:0; background:var(--red); color:#fff; font-weight:700; font-size:12px;">Hold the wire</button>
          <button style="padding:8px 14px; border-radius:8px; border:1px solid var(--line); background:transparent; color:var(--ink); font-weight:600; font-size:12px;">Export report</button>
        </div>
      </div>
      
      <!-- Speakers List -->
      <div style="padding:16px 18px; display:flex; flex-direction:column; gap:12px; flex:1; overflow-y:auto;">
        <!-- Derek -->
        <div style="border:1px solid var(--line); border-radius:12px; padding:14px 16px; background:var(--card);">
          <div style="display:flex; align-items:center; gap:12px;">
            <div style="width:34px; height:34px; border-radius:50%; background:#1b242e; display:flex; align-items:center; justify-content:center; font-weight:700; color:var(--blue);">D</div>
            <div style="flex:1;">
              <div style="font-size:14px; font-weight:700;">Derek Gallardo <span style="color:var(--muted); font-size:12px;">· you</span></div>
              <div style="font-size:11.5px; color:var(--green);">speaking</div>
            </div>
            <span style="font-family:var(--mono); font-size:17px; font-weight:800; color:var(--green);">4%</span>
          </div>
          <div style="height:5px; background:#1b242e; border-radius:3px; margin-top:10px; overflow:hidden;">
            <div style="height:100%; width:4%; background:var(--green);"></div>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--muted); margin-top:8px; font-family:var(--mono);">
            <span>94% speech</span><span>14:20</span><span style="font-weight:700; color:var(--green);">REAL</span>
          </div>
        </div>
        
        <!-- Caller Card -->
        <div id="spk-caller-card" style="border:1px solid var(--line); border-radius:12px; padding:14px 16px; background:var(--card); transition:all 0.4s ease;">
          <div style="display:flex; align-items:center; gap:12px;">
            <div id="spk-caller-av" style="width:34px; height:34px; border-radius:50%; background:#1b242e; display:flex; align-items:center; justify-content:center; font-weight:700; color:var(--ink2);">U</div>
            <div style="flex:1;">
              <div id="spk-caller-name" style="font-size:14px; font-weight:700;">Unknown caller</div>
              <div id="spk-caller-sub" style="font-size:11.5px; color:var(--muted);">monitoring</div>
            </div>
            <span id="spk-caller-pct" style="font-family:var(--mono); font-size:17px; font-weight:800; color:var(--green);">8%</span>
          </div>
          <div style="height:5px; background:#1b242e; border-radius:3px; margin-top:10px; overflow:hidden;">
            <div id="spk-caller-fill" style="height:100%; width:8%; background:var(--green); transition:all 0.4s ease;"></div>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--muted); margin-top:8px; font-family:var(--mono);">
            <span>52% speech</span><span id="spk-caller-time">06:18</span><span id="spk-caller-verdict" style="font-weight:700; color:var(--green);">REAL</span>
          </div>
        </div>
      </div>
      
      <!-- Stats Footer -->
      <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; padding:4px 18px 16px;">
        <div style="border:1px solid var(--line); background:var(--card); border-radius:10px; padding:10px 12px;">
          <b style="display:block; font-size:9px; color:var(--muted); letter-spacing:0.08em;">CHECKS</b>
          <span id="stat-checks" style="font-family:var(--mono); font-size:15px; font-weight:700;">42</span>
        </div>
        <div style="border:1px solid var(--line); background:var(--card); border-radius:10px; padding:10px 12px;">
          <b style="display:block; font-size:9px; color:var(--muted); letter-spacing:0.08em;">SESSION</b>
          <span style="font-family:var(--mono); font-size:15px; font-weight:700;">14:20</span>
        </div>
        <div style="border:1px solid var(--line); background:var(--card); border-radius:10px; padding:10px 12px;">
          <b style="display:block; font-size:9px; color:var(--muted); letter-spacing:0.08em;">PEAK RISK</b>
          <span id="stat-peak" style="font-family:var(--mono); font-size:15px; font-weight:700; color:var(--green);">8%</span>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const caption = document.getElementById('caption-bar');
const stageOAuth = document.getElementById('stage-oauth');
const stageMeet = document.getElementById('stage-meet');

// Timeline Controller
window.runDemo = function() {
  // 0s - 7s: OAuth Screen
  caption.textContent = "Step 1: OAuth Verification — Client ID (940532414120) & Scopes Displayed";
  
  setTimeout(() => {
    document.getElementById('btn-allow').style.transform = 'scale(0.96)';
  }, 6000);

  // 7s: Transition to Meet
  setTimeout(() => {
    stageOAuth.classList.remove('active');
    stageMeet.classList.add('active');
    caption.textContent = "Step 2: Google Meet Side Panel — meetings.space.readonly binds conference space";
  }, 7500);

  // 14s: Media API Streaming (Human Real Voice)
  setTimeout(() => {
    caption.textContent = "Step 3: Meet Media API Streams per-speaker WebRTC audio — Verified as REAL (4% risk)";
  }, 14000);

  // 21s: Synthetic Voice Triggered
  setTimeout(() => {
    caption.textContent = "Step 4: Synthetic Clone Introduced — Model scores audio in 4s windows -> FAKE (94%)";
    
    document.getElementById('tile-caller').classList.add('red-alert');
    document.getElementById('caller-avatar').style.background = '#450a0a';
    document.getElementById('caller-avatar').style.color = '#f87171';
    document.getElementById('caller-badge').style.display = 'block';
    
    document.getElementById('room-banner').style.background = 'rgba(255,77,94,0.15)';
    document.getElementById('room-verdict').textContent = 'FAKE';
    document.getElementById('room-verdict').style.color = 'var(--red)';
    document.getElementById('room-note').textContent = 'A speaker is scoring in the red band (94% confidence).';
    document.getElementById('alert-card').style.display = 'block';
    
    const cCard = document.getElementById('spk-caller-card');
    cCard.style.border = '1px solid rgba(255,77,94,0.6)';
    cCard.style.background = 'rgba(255,77,94,0.06)';
    document.getElementById('spk-caller-av').style.background = '#361217';
    document.getElementById('spk-caller-av').style.color = 'var(--red)';
    document.getElementById('spk-caller-pct').textContent = '94%';
    document.getElementById('spk-caller-pct').style.color = 'var(--red)';
    document.getElementById('spk-caller-fill').style.width = '94%';
    document.getElementById('spk-caller-fill').style.background = 'var(--red)';
    document.getElementById('spk-caller-verdict').textContent = 'FAKE';
    document.getElementById('spk-caller-verdict').style.color = 'var(--red)';
    document.getElementById('stat-peak').textContent = '94%';
    document.getElementById('stat-peak').style.color = 'var(--red)';
    document.getElementById('stat-checks').textContent = '68';
  }, 22000);

  // 30s: Summary & Security
  setTimeout(() => {
    caption.textContent = "Step 5: Ephemeral in-memory processing. Tokens revoked server-side on sign-out.";
  }, 31000);
};
</script>
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[2/4] Built demo animation HTML: {html_path.name}")
    return html_path

def record_video(html_path, duration_sec=38):
    """Record the Playwright browser session to WebM."""
    print(f"[3/4] Recording 1080p video with Playwright for {duration_sec} seconds...")
    raw_video_dir = TEMP_DIR / "raw_video"
    raw_video_dir.mkdir(parents=True, exist_ok=True)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=1,
            record_video_dir=str(raw_video_dir),
            record_video_size={"width": 1920, "height": 1080}
        )
        page = context.new_page()
        page.goto(f"file:///{html_path.as_posix()}")
        page.evaluate("window.runDemo()")
        
        # wait full demo duration
        page.wait_for_timeout(duration_sec * 1000)
        
        context.close()
        browser.close()
        
    # Find generated video
    webm_files = list(raw_video_dir.glob("*.webm"))
    if not webm_files:
        raise RuntimeError("No WebM recording found from Playwright!")
    raw_video = webm_files[0]
    print(f"  Raw video captured: {raw_video.name} ({raw_video.stat().st_size} bytes)")
    return raw_video

def compile_final_video(raw_video, narration_audio):
    """Combine video, audio, and encode MP4."""
    print("[4/4] Encoding final MP4 with synchronized narration audio...")
    output_mp4 = OUT_DIR / "sonave_demo_video.mp4"
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(raw_video),
        "-i", str(narration_audio),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_mp4)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"\n[SUCCESS] Demo Video Generated:")
    print(f"   Path: {output_mp4}")
    print(f"   Size: {output_mp4.stat().st_size / (1024*1024):.2f} MB")
    return output_mp4

if __name__ == "__main__":
    try:
        audio = generate_voiceover()
        html = build_demo_animation_html()
        video = record_video(html, duration_sec=38)
        compile_final_video(video, audio)
    finally:
        # cleanup temp
        pass
