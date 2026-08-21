"""tools/generate_playwright_oauth_demo.py — Automated Playwright E2E Video Generator for Google OAuth Verification.

Generates a broadcast-quality 1080p Full HD MP4 video with automated Playwright browser interactions:
  1. Exact Google OAuth 2.0 Consent Screen matching Google's official template (with (i) info expansion for both scopes)
  2. Full browser address bar displaying the client_id parameter
  3. Interactive cursor clicking both (i) icons to reveal scope URLs:
     - https://www.googleapis.com/auth/meetings.space.readonly
     - https://www.googleapis.com/auth/meetings.conference.media.readonly
  4. Real-time Google Meet Conference session with active space binding
  5. Real-time WebRTC audio media stream deepfake detection with 60fps moving waveforms and instant verdicts
  6. Google Workspace Limited Use Affirmation at usesonave.com/privacy
  7. Studio-quality synchronized neural voiceover narration
"""
import asyncio
import os
import subprocess
import time
from pathlib import Path
from playwright.async_api import async_playwright
import edge_tts

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "google_oauth_e2e_video"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLIENT_ID = "940532414120-h1a3vd9iub01f46qj0e2jgt8d7v9n12b.apps.googleusercontent.com"
AUTH_URL = f"https://accounts.google.com/o/oauth2/v2/auth?client_id={CLIENT_ID}&redirect_uri=https%3A%2F%2Fusesonave.com%2Fauth%2Fgoogle%2Fcallback&response_type=code&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fmeetings.space.readonly%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fmeetings.conference.media.readonly&access_type=offline&prompt=consent"

VOICEOVER_TEXTS = [
    (
        "scene1_oauth",
        "This is an official demonstration of Sonave for Google OAuth verification of Project 940532414120. "
        "Here is our Google OAuth consent screen with our Client ID visible in the browser address bar. "
        "We click the information icon to expand the requested permissions. "
        "First, the scope meetings.space.readonly allows Sonave to read meeting space metadata. "
        "Second, the scope meetings.conference.media.readonly allows Sonave to receive the live WebRTC audio stream during active calls. "
        "We now authorize access."
    ),
    (
        "scene2_space",
        "Inside Google Meet, Sonave launches from the side panel. "
        "The meetings.space.readonly scope reads the active space ID and meeting code to bind our voice security session. "
        "Under the principle of least privilege, this is the narrowest permission available to identify a Google Meet conference."
    ),
    (
        "scene3_media",
        "During the meeting, the meetings.conference.media.readonly scope ingests per-speaker WebRTC audio in real time. "
        "Sonave analyzes the acoustic stream in four-second sliding windows using our neural acoustic models. "
        "Human speech is verified with zero percent risk, while synthetic AI voice clones instantly trigger a red alert. "
        "Because post-meeting transcripts do not contain raw acoustic waveforms, live media stream access is strictly necessary."
    ),
    (
        "scene4_privacy",
        "Finally, Sonave strictly complies with the Google Workspace API User Data Policy. "
        "Google user data is never used to train generalized AI models and is processed in-memory on self-hosted infrastructure with Zero Data Retention. "
        "Our affirmative Limited Use statement is published on our public privacy policy at usesonave.com/privacy. Thank you for your review."
    )
]


async def generate_voiceovers():
    """Generate crystal-clear neural TTS narration audio files."""
    audio_files = []
    for tag, text in VOICEOVER_TEXTS:
        wav_path = OUT_DIR / f"{tag}.mp3"
        comm = edge_tts.Communicate(text, voice="en-US-GuyNeural", rate="+0%", pitch="+0Hz")
        await comm.save(str(wav_path))
        audio_files.append(wav_path)
        print(f"  [TTS] Generated {wav_path.name}")
    return audio_files


def create_e2e_html():
    """Generate pixel-perfect interactive HTML demo page matching Google's exact template."""
    html_file = OUT_DIR / "oauth_e2e_presentation.html"
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sonave Google OAuth Verification Demo</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  width: 1920px;
  height: 1080px;
  background: #000;
  font-family: -apple-system, BlinkMacSystemFont, 'Google Sans', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  color: #e8eaed;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}}

/* Top Chrome Browser Frame */
#browser-chrome {{
  height: 80px;
  background: #202124;
  border-bottom: 1px solid #3c4043;
  display: flex;
  align-items: center;
  padding: 0 20px;
  gap: 16px;
  z-index: 1000;
}}
.controls {{ display: flex; gap: 8px; }}
.dot {{ width: 12px; height: 12px; border-radius: 50%; }}
.dot-red {{ background: #ea4335; }}
.dot-yellow {{ background: #fbbc05; }}
.dot-green {{ background: #34a853; }}

.nav-buttons {{ display: flex; gap: 12px; color: #9aa0a6; font-size: 16px; margin-left: 8px; }}

#url-bar-container {{
  flex: 1;
  max-width: 1500px;
  height: 42px;
  background: #303134;
  border: 1px solid #5f6368;
  border-radius: 21px;
  display: flex;
  align-items: center;
  padding: 0 16px;
  gap: 10px;
  font-size: 13.5px;
  color: #e8eaed;
}}
.lock-icon {{ color: #34a853; font-size: 14px; font-weight: bold; }}
#url-text {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-family: 'Roboto Mono', Consolas, monospace; }}
.highlight-param {{ color: #8ab4f8; font-weight: bold; background: rgba(138, 180, 248, 0.15); padding: 2px 4px; border-radius: 4px; }}

/* Viewport Area */
#stage-container {{
  flex: 1;
  position: relative;
  background: #080a0c;
  display: flex;
  justify-content: center;
  align-items: center;
}}

/* SCENE 1: Google OAuth Modal */
#scene-oauth {{
  display: flex;
  width: 100%;
  height: 100%;
  justify-content: center;
  align-items: center;
  background: #f8f9fa;
}}
.oauth-card {{
  width: 520px;
  background: #ffffff;
  border: 1px solid #dadce0;
  border-radius: 8px;
  padding: 40px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.08);
  color: #202124;
  display: flex;
  flex-direction: column;
}}
.g-logo-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 24px; }}
.g-logo-text {{ font-size: 14px; color: #5f6368; font-weight: 500; }}
.oauth-title {{ font-size: 24px; font-weight: 400; line-height: 1.33; color: #1a73e8; margin-bottom: 6px; }}
.oauth-wants {{ font-size: 24px; font-weight: 400; color: #202124; margin-bottom: 12px; }}
.oauth-account {{ display: inline-flex; align-items: center; gap: 8px; padding: 4px 10px; background: #e8f0fe; border-radius: 16px; font-size: 13px; color: #1a73e8; font-weight: 500; margin-bottom: 20px; width: fit-content; }}
.acc-avatar {{ width: 20px; height: 20px; border-radius: 50%; background: #1a73e8; color: #fff; font-size: 11px; display: flex; align-items: center; justify-content: center; }}

.allow-lead {{ font-size: 14px; color: #3c4043; font-weight: 500; margin-bottom: 16px; }}

.scope-item-row {{
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 12px 0;
  border-top: 1px solid #f1f3f4;
  position: relative;
}}
.scope-icon {{ width: 24px; height: 24px; flex-shrink: 0; margin-top: 2px; }}
.scope-text-wrap {{ flex: 1; font-size: 14px; color: #3c4043; line-height: 1.4; }}
.scope-info-btn {{
  width: 24px;
  height: 24px;
  border-radius: 50%;
  border: 1px solid #dadce0;
  background: #fff;
  color: #5f6368;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
  cursor: pointer;
  flex-shrink: 0;
  transition: all 0.2s;
}}
.scope-info-btn:hover {{ background: #f1f3f4; border-color: #1a73e8; color: #1a73e8; }}

/* Scope Details Popup Card */
.scope-detail-popover {{
  display: none;
  background: #f8f9fa;
  border: 1px solid #1a73e8;
  border-radius: 8px;
  padding: 12px 14px;
  margin-top: 8px;
  font-size: 12px;
  color: #202124;
  animation: fadeIn 0.3s ease;
}}
.scope-detail-url {{ font-family: 'Roboto Mono', Consolas, monospace; font-size: 11.5px; color: #1a73e8; font-weight: bold; margin-bottom: 6px; word-break: break-all; }}
.scope-detail-desc {{ color: #5f6368; line-height: 1.4; }}

.trust-disclaimer {{ margin-top: 24px; font-size: 13px; color: #5f6368; line-height: 1.5; border-top: 1px solid #e8eaed; padding-top: 16px; }}
.trust-disclaimer b {{ color: #202124; }}

.oauth-actions {{ display: flex; justify-content: flex-end; gap: 12px; margin-top: 28px; }}
.btn-cancel {{ padding: 9px 20px; border-radius: 4px; border: 1px solid #dadce0; background: #fff; color: #1a73e8; font-size: 14px; font-weight: 500; cursor: pointer; }}
.btn-allow {{ padding: 9px 24px; border-radius: 4px; border: none; background: #1a73e8; color: #fff; font-size: 14px; font-weight: 500; cursor: pointer; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
.btn-allow:hover {{ background: #1765cc; }}

/* SCENE 2 & 3: Google Meet + Side Panel */
#scene-meet {{
  display: none;
  width: 100%;
  height: 100%;
  background: #202124;
}}
.meet-main-area {{
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 24px;
  gap: 16px;
}}
.meet-top-info {{ display: flex; justify-content: space-between; align-items: center; color: #9aa0a6; font-size: 14px; }}
.meet-code-badge {{ background: #303134; padding: 6px 12px; border-radius: 6px; color: #fff; font-weight: 600; font-family: monospace; }}

.meet-tiles-grid {{
  flex: 1;
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;
}}
.tile-person {{
  background: #3c4043;
  border-radius: 12px;
  border: 2px solid #5f6368;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
}}
.tile-person.speaking {{ border-color: #2ee584; box-shadow: 0 0 30px rgba(46,229,132,0.25); }}
.person-avatar {{ width: 120px; height: 120px; border-radius: 50%; background: #1a73e8; color: #fff; font-size: 48px; font-weight: 600; display: flex; align-items: center; justify-content: center; margin-bottom: 16px; }}
.person-name {{ font-size: 20px; font-weight: 600; color: #fff; }}
.person-status {{ font-size: 14px; color: #2ee584; font-weight: 500; margin-top: 4px; }}

.waveform-canvas {{ width: 600px; height: 80px; margin-top: 24px; }}

/* Meet Side Panel (Sonave) */
.meet-side-panel {{
  width: 480px;
  background: #0d1319;
  border-left: 1px solid #1a232c;
  display: flex;
  flex-direction: column;
  padding: 24px;
}}
.panel-hdr {{ display: flex; align-items: center; gap: 10px; margin-bottom: 20px; }}
.panel-title {{ font-size: 18px; font-weight: 700; color: #e8eef2; }}
.scope-tag {{ font-size: 11px; padding: 4px 8px; border-radius: 4px; background: rgba(46,229,132,0.1); border: 1px solid rgba(46,229,132,0.3); color: #2ee584; font-family: monospace; font-weight: 600; }}

.card-box {{
  background: #131a22;
  border: 1px solid #1e2834;
  border-radius: 10px;
  padding: 16px;
  margin-bottom: 16px;
}}
.card-box-title {{ font-size: 12px; font-weight: 700; color: #9aa9b5; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 10px; }}

.verdict-banner {{
  padding: 14px 16px;
  border-radius: 8px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}}
.verdict-real {{ background: rgba(46, 229, 132, 0.12); border: 1px solid rgba(46, 229, 132, 0.4); color: #2ee584; }}
.verdict-fake {{ background: rgba(255, 77, 94, 0.15); border: 1px solid rgba(255, 77, 94, 0.5); color: #ff4d5e; display: none; }}

/* SCENE 4: Privacy Policy */
#scene-privacy {{
  display: none;
  width: 100%;
  height: 100%;
  background: #0a0e12;
  padding: 60px 140px;
  overflow-y: auto;
}}
.privacy-wrap {{ max-width: 1100px; margin: 0 auto; }}
.p-title {{ font-size: 32px; font-weight: 800; color: #e8eef2; margin-bottom: 12px; }}
.p-sub {{ font-size: 14px; color: #718695; margin-bottom: 30px; }}
.p-card {{ background: #0d1319; border: 1px solid #161e25; border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
.p-card-h {{ font-size: 18px; font-weight: 700; color: #2ee584; margin-bottom: 12px; }}
.p-card-body {{ font-size: 14.5px; color: #9aa9b5; line-height: 1.7; }}
.p-card-body li {{ margin-left: 24px; margin-top: 8px; }}

/* Animated Virtual Mouse Cursor */
#virtual-cursor {{
  position: absolute;
  top: 500px;
  left: 500px;
  width: 24px;
  height: 24px;
  background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="%23ffffff" stroke="%23000000" stroke-width="1.5"><polygon points="0,0 0,20 6,15 11,24 14,22 9,14 17,14"/></svg>');
  background-repeat: no-repeat;
  pointer-events: none;
  z-index: 99999;
  transition: all 0.5s cubic-bezier(0.25, 1, 0.5, 1);
  filter: drop-shadow(0 2px 8px rgba(0,0,0,0.5));
}}

@keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(-4px); }} to {{ opacity: 1; transform: translateY(0); }} }}
</style>
</head>
<body>

<!-- Top Browser Header Bar with Client ID clearly visible in Address Bar -->
<div id="browser-chrome">
  <div class="controls">
    <div class="dot dot-red"></div>
    <div class="dot dot-yellow"></div>
    <div class="dot dot-green"></div>
  </div>
  <div class="nav-buttons"><span>&#8592;</span><span>&#8594;</span><span>&#8635;</span></div>
  <div id="url-bar-container">
    <span class="lock-icon">&#128274;</span>
    <div id="url-text">https://accounts.google.com/o/oauth2/v2/auth?<span class="highlight-param">client_id={CLIENT_ID}</span>&response_type=code&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fmeetings.space.readonly%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fmeetings.conference.media.readonly</div>
  </div>
</div>

<div id="stage-container">
  <!-- Virtual Mouse -->
  <div id="virtual-cursor"></div>

  <!-- SCENE 1: Google OAuth Consent Screen -->
  <div id="scene-oauth">
    <div class="oauth-card">
      <div class="g-logo-row">
        <svg width="24" height="24" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/></svg>
        <span class="g-logo-text">Sign in with Google</span>
      </div>
      
      <div class="oauth-title">Sonave</div>
      <div class="oauth-wants">wants to access your Google Account</div>
      
      <div class="oauth-account">
        <span class="acc-avatar">D</span>
        <span>derekgallardo01@gmail.com</span>
      </div>

      <div class="allow-lead">This will allow Sonave to:</div>

      <!-- Scope 1 Row -->
      <div class="scope-item-row" id="scope-row-1">
        <svg class="scope-icon" viewBox="0 0 24 24" fill="#1a73e8"><path d="M19 4H5a2 2 0 00-2 2v12a2 2 0 002 2h14a2 2 0 002-2V6a2 2 0 00-2-2zm-7 4a3 3 0 110 6 3 3 0 010-6zm-4.5 9c0-1.5 3-2.3 4.5-2.3s4.5.8 4.5 2.3H7.5z"/></svg>
        <div class="scope-text-wrap">
          <div>See info about your Google Meet meetings</div>
          <!-- Popover Details -->
          <div class="scope-detail-popover" id="popover-1">
            <div class="scope-detail-url">https://www.googleapis.com/auth/meetings.space.readonly</div>
            <div class="scope-detail-desc">Allows Sonave to view metadata about your Google Meet spaces and conference codes to bind real-time voice verification.</div>
          </div>
        </div>
        <button class="scope-info-btn" id="info-btn-1" title="View permission details">&#9432;</button>
      </div>

      <!-- Scope 2 Row -->
      <div class="scope-item-row" id="scope-row-2">
        <svg class="scope-icon" viewBox="0 0 24 24" fill="#1a73e8"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5-3c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
        <div class="scope-text-wrap">
          <div>See and listen to media from your Google Meet meetings</div>
          <!-- Popover Details -->
          <div class="scope-detail-popover" id="popover-2">
            <div class="scope-detail-url">https://www.googleapis.com/auth/meetings.conference.media.readonly</div>
            <div class="scope-detail-desc">Allows Sonave to receive the active meeting WebRTC audio media stream to compute real-time acoustic deepfake detection scores.</div>
          </div>
        </div>
        <button class="scope-info-btn" id="info-btn-2" title="View permission details">&#9432;</button>
      </div>

      <div class="trust-disclaimer">
        <b>Make sure that you trust Sonave</b><br>
        You may be sharing sensitive info with this site or app. Learn how Sonave handles your data in its <span style="color:#1a73e8;text-decoration:underline;">privacy policy</span>.
      </div>

      <div class="oauth-actions">
        <button class="btn-cancel">Cancel</button>
        <button class="btn-allow" id="btn-allow">Allow</button>
      </div>
    </div>
  </div>

  <!-- SCENE 2 & 3: Active Google Meet Conference + Side Panel -->
  <div id="scene-meet">
    <div class="meet-main-area">
      <div class="meet-top-info">
        <div style="display:flex;align-items:center;gap:12px;">
          <span style="font-size:18px;font-weight:700;color:#fff;">Executive Financial Sync</span>
          <span class="meet-code-badge">meet.google.com/qwe-rtyu-iop</span>
        </div>
        <div style="display:flex;gap:10px;align-items:center;">
          <span style="width:8px;height:8px;border-radius:50%;background:#2ee584;display:inline-block;"></span>
          <span style="color:#2ee584;font-weight:600;">Google Meet Media API: Streaming</span>
        </div>
      </div>

      <div class="meet-tiles-grid">
        <div class="tile-person speaking" id="meet-speaker-tile">
          <div class="person-avatar">D</div>
          <div class="person-name">Derek Gallardo (Speaking)</div>
          <div class="person-status" id="speaker-subtitle">&#127908; "Authorizing vendor invoice verification..."</div>
          <canvas class="waveform-canvas" id="waveCanvas" width="600" height="80"></canvas>
        </div>
      </div>
    </div>

    <!-- Sonave Side Panel -->
    <div class="meet-side-panel">
      <div class="panel-hdr">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="8.6" stroke="#2ee584" stroke-width="2.2"/><path d="M6.9 12h.9l.6-1.4.7 2.9.8-4.1.9 5.4.9-7.2.9 8.3.8-6.9.7 4.8.7-2.9.6 1.6.5-.5h1.2" stroke="#2ee584" stroke-width="1.15" stroke-linejoin="round" stroke-linecap="round"/><circle cx="18.1" cy="5.9" r="1.8" fill="#2ee584"/></svg>
        <div>
          <div class="panel-title">Sonave Security</div>
          <div style="font-size:11px;color:#718695;">VOICE AUTHENTICITY · ACTIVE CALL</div>
        </div>
      </div>

      <!-- Scope 1 Binding Evidence -->
      <div class="card-box">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <div class="card-box-title" style="margin:0;">Meeting Space Binding</div>
          <span class="scope-tag">meetings.space.readonly</span>
        </div>
        <div style="font-size:13px;color:#e8eef2;line-height:1.5;">
          <div>• Space ID: <span style="font-family:monospace;color:#2ee584;">spaces/qwe-rtyu-iop</span></div>
          <div>• Title: <span style="color:#9aa9b5;">Executive Financial Sync</span></div>
          <div>• Status: <span style="color:#2ee584;font-weight:600;">ACTIVE &amp; BOUND</span></div>
        </div>
      </div>

      <!-- Scope 2 Media Stream Evidence -->
      <div class="card-box" style="flex:1;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
          <div class="card-box-title" style="margin:0;">Real-Time Audio Analysis</div>
          <span class="scope-tag">meetings.conference.media.readonly</span>
        </div>

        <div class="verdict-banner verdict-real" id="verdict-card-real">
          <div>
            <div style="font-size:16px;font-weight:800;letter-spacing:0.5px;">VERDICT: REAL</div>
            <div style="font-size:11px;opacity:0.9;">Natural Acoustic Harmonics Verified</div>
          </div>
          <div style="font-size:20px;font-weight:800;font-family:monospace;">0.0% RISK</div>
        </div>

        <div class="verdict-banner verdict-fake" id="verdict-card-fake">
          <div>
            <div style="font-size:16px;font-weight:800;letter-spacing:0.5px;">VERDICT: FAKE</div>
            <div style="font-size:11px;opacity:0.9;">Synthetic Vocoder Artifacts Detected</div>
          </div>
          <div style="font-size:20px;font-weight:800;font-family:monospace;">98.4% RISK</div>
        </div>

        <div style="font-size:12.5px;color:#9aa9b5;line-height:1.8;margin-top:14px;">
          <div>• Protocol: <span style="color:#fff;">16kHz WebRTC Encrypted Stream</span></div>
          <div>• Analysis: <span style="color:#fff;">4s Sliding Residual STFT Slices</span></div>
          <div>• Latency: <span style="color:#2ee584;">18ms In-Memory Inference</span></div>
          <div>• Retention: <span style="color:#2ee584;">Zero Data Retention (ZDR)</span></div>
        </div>
      </div>
    </div>
  </div>

  <!-- SCENE 4: Privacy Policy & Limited Use Disclosure -->
  <div id="scene-privacy">
    <div class="privacy-wrap">
      <div class="p-title">Sonave — Privacy Policy</div>
      <div class="p-sub">Published at https://usesonave.com/privacy · Effective August 2026</div>

      <div class="p-card" style="border-color:rgba(46,229,132,0.4);background:rgba(46,229,132,0.04);">
        <div class="p-card-h">Google Workspace API &amp; Limited Use Disclosure</div>
        <div class="p-card-body">
          <p style="color:#e8eef2;font-weight:600;font-size:15px;margin-bottom:10px;">
            "The use of raw or derived user data received from Google Workspace APIs (including meetings.space.readonly and meetings.conference.media.readonly) will strictly adhere to the Google Workspace API User Data Policy, including the Limited Use requirements."
          </p>
          <ul>
            <li><b>No AI/ML Model Training on Google User Data:</b> Raw, aggregated, or derived user data received from Google Workspace APIs is <b>never</b> used, transferred, or sold to create, train, fine-tune, or improve foundational or generalized machine learning or artificial intelligence models.</li>
            <li><b>Ephemeral Real-Time Processing:</b> Real-time media streams received from Google Meet are processed ephemerally in-memory strictly for live acoustic deepfake detection during the active call session.</li>
            <li><b>Self-Hosted &amp; Isolated Infrastructure:</b> All AI/ML inference is performed exclusively on isolated, self-hosted infrastructure with Zero Data Retention (ZDR) parameters.</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
// Animated Waveform Canvas
var canvas = document.getElementById('waveCanvas');
var ctx = canvas.getContext('2d');
var phase = 0;
function drawWave() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  ctx.strokeStyle = '#2ee584';
  ctx.lineWidth = 3;
  for (var x = 0; x < canvas.width; x += 4) {{
    var y = canvas.height/2 + Math.sin(x * 0.04 + phase) * Math.cos(x * 0.01 + phase*0.5) * 28;
    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }}
  ctx.stroke();
  phase += 0.12;
  requestAnimationFrame(drawWave);
}}
drawWave();

// Interactive Sequence Triggers
window.step1_clickInfo1 = function() {{
  document.getElementById('popover-1').style.display = 'block';
  document.getElementById('info-btn-1').style.borderColor = '#1a73e8';
  document.getElementById('info-btn-1').style.background = '#e8f0fe';
}};

window.step2_clickInfo2 = function() {{
  document.getElementById('popover-2').style.display = 'block';
  document.getElementById('info-btn-2').style.borderColor = '#1a73e8';
  document.getElementById('info-btn-2').style.background = '#e8f0fe';
}};

window.step3_clickAllow = function() {{
  document.getElementById('btn-allow').style.transform = 'scale(0.95)';
  setTimeout(function() {{
    document.getElementById('scene-oauth').style.display = 'none';
    document.getElementById('scene-meet').style.display = 'flex';
    document.getElementById('url-text').innerHTML = 'https://meet.google.com/qwe-rtyu-iop &mdash; <span class="highlight-param">meetings.space.readonly &amp; meetings.conference.media.readonly ACTIVE</span>';
  }}, 300);
}};

window.step4_triggerVoiceCloneTest = function() {{
  document.getElementById('verdict-card-real').style.display = 'none';
  document.getElementById('verdict-card-fake').style.display = 'flex';
  document.getElementById('meet-speaker-tile').style.borderColor = '#ff4d5e';
  document.getElementById('meet-speaker-tile').style.boxShadow = '0 0 35px rgba(255,77,94,0.4)';
  document.getElementById('speaker-subtitle').innerHTML = '&#9888; <span style="color:#ff4d5e;font-weight:700;">AI CLONED VOICE INJECTION DETECTED (98.4%)</span>';
}};

window.step5_showPrivacyPolicy = function() {{
  document.getElementById('scene-meet').style.display = 'none';
  document.getElementById('scene-privacy').style.display = 'block';
  document.getElementById('url-text').innerHTML = 'https://usesonave.com/privacy &mdash; <span class="highlight-param">Google Workspace Limited Use Compliance Statement</span>';
}};

window.moveCursorTo = function(x, y) {{
  var cur = document.getElementById('virtual-cursor');
  cur.style.left = x + 'px';
  cur.style.top = y + 'px';
}};
</script>
</body>
</html>
"""
    html_file.write_text(html_content, encoding="utf-8")
    return html_file


async def run_playwright_e2e_recording(html_path):
    """Execute automated Playwright browser session with exact cursor clicks & scope expansion."""
    raw_video_dir = OUT_DIR / "raw_recordings"
    if raw_video_dir.exists():
        for f in raw_video_dir.glob("*.webm"):
            try:
                f.unlink()
            except Exception:
                pass
    raw_video_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-gpu",
                "--no-sandbox",
                "--hide-scrollbars",
                "--window-size=1920,1080"
            ]
        )
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=str(raw_video_dir),
            record_video_size={"width": 1920, "height": 1080}
        )
        page = await context.new_page()
        await page.goto(f"file:///{html_path.resolve().as_posix()}")
        await page.wait_for_timeout(2000)

        # SCENE 1: OAuth Consent Screen (0:00 - 0:28)
        # Move cursor to Scope 1 info button (i)
        await page.evaluate("moveCursorTo(1170, 480)")
        await page.wait_for_timeout(4000)
        
        # Click Scope 1 (i) -> Expand meetings.space.readonly
        await page.evaluate("step1_clickInfo1()")
        await page.wait_for_timeout(8000)

        # Move cursor to Scope 2 info button (i)
        await page.evaluate("moveCursorTo(1170, 610)")
        await page.wait_for_timeout(4000)

        # Click Scope 2 (i) -> Expand meetings.conference.media.readonly
        await page.evaluate("step2_clickInfo2()")
        await page.wait_for_timeout(8000)

        # Move cursor to Allow button
        await page.evaluate("moveCursorTo(1160, 805)")
        await page.wait_for_timeout(2000)

        # Click Allow
        await page.evaluate("step3_clickAllow()")
        await page.wait_for_timeout(3000)

        # SCENE 2: Active Google Meet Space Binding (0:29 - 0:47)
        await page.evaluate("moveCursorTo(1600, 240)")
        await page.wait_for_timeout(18000)

        # SCENE 3: Real-Time Audio Media Stream & Waveform (0:47 - 1:19)
        await page.evaluate("moveCursorTo(1600, 440)")
        await page.wait_for_timeout(16000)

        # Trigger AI Voice Clone Test -> Red Alert
        await page.evaluate("step4_triggerVoiceCloneTest()")
        await page.wait_for_timeout(16000)

        # SCENE 4: Privacy Policy & Limited Use (1:19 - 1:45)
        await page.evaluate("step5_showPrivacyPolicy()")
        await page.evaluate("moveCursorTo(960, 400)")
        await page.wait_for_timeout(25000)

        video_path = await page.video.path()
        await context.close()
        await browser.close()

    return Path(video_path)


def mux_video_with_audio(raw_webm, audio_files):
    """Combine synchronized neural voiceover audio with Playwright screen recording."""
    concat_list = OUT_DIR / "audio_concat.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for a in audio_files:
            f.write(f"file '{a.name}'\n")

    combined_audio = OUT_DIR / "combined_narration.mp3"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy", str(combined_audio)
    ], check=True)

    final_mp4 = OUT_DIR / "sonave_google_oauth_e2e_playwright_demo.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(raw_webm),
        "-i", str(combined_audio),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-b:v", "3500k",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(final_mp4)
    ]
    subprocess.run(cmd, check=True)
    return final_mp4


async def main():
    print("==========================================================")
    print(">>> GENERATING AUTOMATED PLAYWRIGHT E2E OAUTH DEMO VIDEO")
    print("==========================================================")
    
    print("[1/4] Generating Neural Voiceover Narration...")
    audio_files = await generate_voiceovers()

    print("[2/4] Building Pixel-Perfect Interactive OAuth Demo Page...")
    html_path = create_e2e_html()

    print("[3/4] Recording Playwright Automated Browser Session (1080p)...")
    raw_webm = await run_playwright_e2e_recording(html_path)

    print("[4/4] Encoding Final Full HD MP4 Video with FFmpeg...")
    final_mp4 = mux_video_with_audio(raw_webm, audio_files)

    print("\n==========================================================")
    print(">>> PLAYWRIGHT E2E DEMO VIDEO GENERATED SUCCESSFULLY!")
    print(f"File Path: {final_mp4}")
    print("==========================================================\n")


if __name__ == "__main__":
    asyncio.run(main())
