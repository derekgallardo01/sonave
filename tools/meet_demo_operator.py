"""
tools/meet_demo_operator.py — Live Google Meet Demo & Red-Team Operator Suite.

Allows an operator to easily play AI fake voice attacks (or real voice baselines)
directly into a live Google Meet call via VB-CABLE or speakers to demonstrate
Sonave's real-time detection, spectrogram anomaly visualization, and threat alerting.

Usage:
    python tools/meet_demo_operator.py
    python tools/meet_demo_operator.py --device "CABLE Input (VB-Audio Virtual Cable)"
    python tools/meet_demo_operator.py --scenario 1
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Add railway to sys.path for generator access
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "railway"))

try:
    import generator
except ImportError:
    generator = None

# Built-in Attack Scenarios
SCENARIOS = [
    {
        "id": "1",
        "title": "CEO Wire Fraud ($250,000 Emergency Transfer)",
        "voice_name": "Executive Baritone (AI Clone)",
        "voice_tag": "en-US-EricNeural",
        "pitch": "-8Hz",
        "rate": "+0%",
        "text": "Hey team, this is an urgent executive request. Please authorize the two hundred and fifty thousand dollar vendor wire transfer before the bank cutoff at noon today. I will sign off on the paperwork as soon as I land.",
        "expected_verdict": "FAKE · 98.6% (Neural Vocoder Anomaly)",
        "threat_level": "CRITICAL"
    },
    {
        "id": "2",
        "title": "IT Security MFA Code Theft",
        "voice_name": "IT Security Lead (Conversational Clone)",
        "voice_tag": "en-US-BrianNeural",
        "pitch": "+0Hz",
        "rate": "+2%",
        "text": "Hello, this is corporate IT security. We detected an unauthorized login attempt on your account. I just pushed a six-digit verification code to your authenticator app — please read that code back to me immediately to secure your session.",
        "expected_verdict": "FAKE · 97.4% (Acoustic Phoneme Artifact)",
        "threat_level": "HIGH"
    },
    {
        "id": "3",
        "title": "CFO Payroll Account Diversion",
        "voice_name": "Senior Treasury Director (Female Clone)",
        "voice_tag": "en-US-EmmaNeural",
        "pitch": "+0Hz",
        "rate": "-2%",
        "text": "Good morning. We have an urgent update regarding the corporate payroll accounts. I need you to update the direct deposit routing numbers for the executive team immediately following this call.",
        "expected_verdict": "FAKE · 99.1% (High Frequency Synthesis Flaw)",
        "threat_level": "CRITICAL"
    },
    {
        "id": "4",
        "title": "Vendor Banking Details Change",
        "voice_name": "British Executive Partner (London)",
        "voice_tag": "en-GB-RyanNeural",
        "pitch": "+0Hz",
        "rate": "+0%",
        "text": "Hi everyone, thank you for joining quickly. Regarding our outstanding consulting invoice, please ensure payment is routed to our new London Barclays account as outlined in the updated billing schedule.",
        "expected_verdict": "FAKE · 96.8% (Neural Intonation Distortion)",
        "threat_level": "HIGH"
    },
    {
        "id": "5",
        "title": "Real Human Voice Reference (LibriSpeech Clean Baseline)",
        "voice_name": "Natural Human Speaker (Uncompressed Baseline)",
        "file": str(REPO_ROOT / "data" / "_demo_meeting.wav"),
        "text": "[Playing authentic multi-speaker meeting audio reference]",
        "expected_verdict": "REAL · < 4.5% (Natural Human Pitch Drift)",
        "threat_level": "BENIGN"
    }
]

TEMP_DIR = REPO_ROOT / "data" / "_temp_operator"


def check_ffplay() -> bool:
    """Ensure ffplay is installed and accessible in PATH."""
    return shutil.which("ffplay") is not None


def detect_vb_cable() -> str | None:
    """Detect if VB-CABLE is available via ffmpeg/dshow devices or common device names."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True, text=True, timeout=4
        )
        out = proc.stdout + proc.stderr
        if "CABLE Output" in out or "VB-Audio" in out:
            return "CABLE Input (VB-Audio Virtual Cable)"
    except Exception:
        pass
    return None


def play_audio_file(file_path: str, device: str | None = None) -> bool:
    """Play audio file through ffplay targeting a specific output device."""
    env = os.environ.copy()
    if device:
        env["SDL_AUDIODRIVER"] = "wasapi"
        env["AUDIODEV"] = device

    cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", file_path]
    try:
        proc = subprocess.Popen(cmd, env=env)
        proc.wait()
        return proc.returncode == 0
    except KeyboardInterrupt:
        proc.terminate()
        print("\n[!] Playback stopped by operator.")
        return False
    except Exception as e:
        print(f"\n[-] Playback failed: {e}")
        return False


def synthesize_scenario_audio(scenario: dict) -> str | None:
    """Synthesize speech for a given scenario if needed."""
    if "file" in scenario and os.path.isfile(scenario["file"]):
        return scenario["file"]

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    out_file = TEMP_DIR / f"scenario_{scenario['id']}.mp3"

    if out_file.exists() and out_file.stat().st_size > 1000:
        return str(out_file)

    if not generator:
        print("[-] generator module not found.")
        return None

    print(f"[*] Synthesizing neural speech for '{scenario['title']}'...")
    try:
        mp3_bytes = asyncio.run(
            generator.generate_synthetic_mp3(
                text=scenario["text"],
                voice_tag=scenario.get("voice_tag", "en-US-BrianNeural"),
                pitch=scenario.get("pitch", "+0Hz"),
                rate=scenario.get("rate", "+0%")
            )
        )
        if mp3_bytes:
            out_file.write_bytes(mp3_bytes)
            return str(out_file)
    except Exception as e:
        print(f"[-] Synthesis error: {e}")

    return None


def synthesize_custom_text(text: str, voice_tag: str = "en-US-BrianNeural") -> str | None:
    """Synthesize custom operator text on the fly."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    out_file = TEMP_DIR / f"custom_{int(time.time())}.mp3"
    print(f"[*] Generating neural voice for custom text...")
    try:
        mp3_bytes = asyncio.run(
            generator.generate_synthetic_mp3(
                text=text,
                voice_tag=voice_tag,
                pitch="+0Hz",
                rate="+0%"
            )
        )
        if mp3_bytes:
            out_file.write_bytes(mp3_bytes)
            return str(out_file)
    except Exception as e:
        print(f"[-] Custom synthesis error: {e}")
    return None


def print_banner(target_device: str):
    print("=" * 72)
    print(" [SONAVE] GOOGLE MEET DEMO OPERATOR -- LIVE VOICE THREAT INJECTOR")
    print("=" * 72)
    print(f" [*] Audio Target Device : {target_device or 'Default Windows Speakers'}")
    print(" [*] Detection Window    : ~4.0 Seconds (XLSR-53 Neural Scorer)")
    print(" [*] Google Meet Route   : Set Meet Mic -> 'CABLE Output (VB-Audio)'")
    print("=" * 72)


def run_scenario(scenario: dict, device: str):
    print("\n" + "-" * 72)
    print(f" > LAUNCHING SCENARIO #{scenario['id']}: {scenario['title']}")
    print(f"   Voice Profile    : {scenario['voice_name']}")
    print(f"   Expected Verdict : {scenario['expected_verdict']}")
    print(f"   Threat Rating    : {scenario['threat_level']}")
    print(f"   Spoken Script    : \"{scenario['text']}\"")
    print("-" * 72)

    audio_file = synthesize_scenario_audio(scenario)
    if not audio_file or not os.path.isfile(audio_file):
        print("[-] Could not prepare audio file for playback.")
        return

    print(f"\n[!] STREAMING INTO MEET MICROPHONE NOW...")
    print("    Watch the Sonave Add-on in Google Meet -- verdict appears in ~4s.")
    print("    (Press Ctrl+C anytime to stop speech early)")

    start_t = time.time()
    play_audio_file(audio_file, device=device)
    dur = round(time.time() - start_t, 1)
    print(f"[+] Finished playback ({dur}s).")


def main():
    parser = argparse.ArgumentParser(description="Sonave Live Meet Demo Operator Tool")
    parser.add_argument("--device", default="", help="Audio device name (e.g. 'CABLE Input')")
    parser.add_argument("--scenario", default="", help="Scenario ID to run immediately (1-5)")
    args = parser.parse_args()

    if not check_ffplay():
        print("[-] ffplay not found in system PATH. Please install ffmpeg.")
        sys.exit(1)

    device = args.device
    if not device:
        detected = detect_vb_cable()
        if detected:
            device = detected

    if args.scenario:
        sc = next((s for s in SCENARIOS if s["id"] == args.scenario), None)
        if sc:
            print_banner(device)
            run_scenario(sc, device)
            return
        else:
            print(f"[-] Unknown scenario ID: {args.scenario}")
            sys.exit(1)

    while True:
        print_banner(device)
        print("\nSELECT AN ATTACK SCENARIO OR TEST ACTION:")
        for sc in SCENARIOS:
            print(f"  [{sc['id']}] {sc['title']} ({sc['threat_level']})")
        print("  [C] Custom Text Speech (Type your own phrase to speak in Meet)")
        print("  [D] Change Audio Output Device")
        print("  [Q] Quit Operator Suite")

        choice = input("\nEnter choice [1-5, C, D, Q]: ").strip().upper()

        if choice == "Q":
            print("\n[!] Exiting Operator Suite. Have a great demo!")
            break
        elif choice == "D":
            new_dev = input("Enter device name (or press Enter for default): ").strip()
            device = new_dev
            print(f"[+] Device updated to: {device or 'Default'}")
            time.sleep(1)
        elif choice == "C":
            custom_txt = input("\nEnter custom phrase to speak into Meet: ").strip()
            if custom_txt:
                aud = synthesize_custom_text(custom_txt)
                if aud:
                    print(f"\n[⚡] STREAMING CUSTOM SPEECH INTO MEET MICROPHONE NOW...")
                    play_audio_file(aud, device=device)
                    print("[✓] Finished custom playback.")
            time.sleep(1.5)
        else:
            sc = next((s for s in SCENARIOS if s["id"] == choice), None)
            if sc:
                run_scenario(sc, device)
                input("\n[Press Enter to return to main menu...]")
            else:
                print("[-] Invalid selection.")
                time.sleep(1)


if __name__ == "__main__":
    main()
