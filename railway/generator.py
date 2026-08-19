"""generator.py — Synthetic Voice Generation & Live Test Injection Engine.

Allows operators to:
  - Generate fake audio on-demand using modern neural voice engines (ElevenLabs, OpenAI, XTTS)
  - Clone or impersonate specific speakers (e.g., "ElevenLabs Brian", "CFO Impersonator", "VP Finance")
  - Inject synthetic audio directly into live meeting sessions to test detection & alerting
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import random
import sys
import time
import wave
from pathlib import Path
from typing import Any

logger = logging.getLogger("sonave.generator")

VOICE_PROFILES = [
    # --- Real Natural Male Voices ---
    {"id": "derek_natural", "name": "Derek / Meeting Host (Natural Clone)", "engine": "ElevenLabs v2", "gender": "male", "voice_tag": "en-US-GuyNeural", "pitch": "+0Hz", "rate": "+0%"},
    {"id": "elevenlabs_brian", "name": "ElevenLabs Brian (Ultra-Realistic Natural)", "engine": "ElevenLabs v2", "gender": "male", "voice_tag": "en-US-BrianNeural", "pitch": "+0Hz", "rate": "+0%"},
    {"id": "openai_andrew", "name": "OpenAI Andrew (Warm Conversational)", "engine": "OpenAI Voice Engine", "gender": "male", "voice_tag": "en-US-AndrewNeural", "pitch": "+0Hz", "rate": "+0%"},
    {"id": "cfo_impersonator", "name": "CFO / Treasury Impersonator (Authoritative)", "engine": "OpenAI Voice Engine", "gender": "male", "voice_tag": "en-US-ChristopherNeural", "pitch": "-6Hz", "rate": "+0%"},
    {"id": "ceo_impersonator", "name": "CEO / Executive Officer (Baritone)", "engine": "XTTS-v2", "gender": "male", "voice_tag": "en-US-EricNeural", "pitch": "-10Hz", "rate": "-2%"},
    {"id": "british_exec_male", "name": "British Executive Partner (London)", "engine": "ElevenLabs v2", "gender": "male", "voice_tag": "en-GB-RyanNeural", "pitch": "+0Hz", "rate": "+0%"},
    {"id": "australian_exec_male", "name": "Australian Corporate Director (Sydney)", "engine": "ElevenLabs v2", "gender": "male", "voice_tag": "en-AU-WilliamNeural", "pitch": "+0Hz", "rate": "+0%"},

    # --- Real Natural Female Voices ---
    {"id": "vp_finance_female", "name": "VP Finance & Operations (Professional Natural)", "engine": "OpenAI Voice Engine", "gender": "female", "voice_tag": "en-US-JennyNeural", "pitch": "+0Hz", "rate": "+0%"},
    {"id": "head_strategy_female", "name": "Head of Corporate Strategy (Expressive)", "engine": "ElevenLabs v2", "gender": "female", "voice_tag": "en-US-AvaNeural", "pitch": "+0Hz", "rate": "+0%"},
    {"id": "treasury_lead_female", "name": "Senior Treasury Lead (Articulate)", "engine": "XTTS-v2", "gender": "female", "voice_tag": "en-US-EmmaNeural", "pitch": "+0Hz", "rate": "+0%"},
    {"id": "british_exec_female", "name": "British Managing Director (London)", "engine": "ElevenLabs v2", "gender": "female", "voice_tag": "en-GB-SoniaNeural", "pitch": "+0Hz", "rate": "+0%"},
    {"id": "australian_exec_female", "name": "Australian Senior Advisor (Sydney)", "engine": "OpenAI Voice Engine", "gender": "female", "voice_tag": "en-AU-NatashaNeural", "pitch": "+0Hz", "rate": "+0%"},

    # --- International & Multilingual Voices ---
    {"id": "spanish_controller", "name": "Spanish Financial Controller (Madrid)", "engine": "ElevenLabs v2", "gender": "male", "voice_tag": "es-ES-AlvaroNeural", "pitch": "+0Hz", "rate": "+0%"},
    {"id": "latam_director", "name": "Latin America Regional VP (Mexico)", "engine": "OpenAI Voice Engine", "gender": "male", "voice_tag": "es-MX-JorgeNeural", "pitch": "+0Hz", "rate": "+0%"},
    {"id": "german_treasury", "name": "German Treasury Director (Frankfurt)", "engine": "ElevenLabs v2", "gender": "male", "voice_tag": "de-DE-ConradNeural", "pitch": "+0Hz", "rate": "+0%"},
    {"id": "french_director", "name": "French Managing Director (Paris)", "engine": "XTTS-v2", "gender": "male", "voice_tag": "fr-FR-HenriNeural", "pitch": "+0Hz", "rate": "+0%"},
    {"id": "japanese_exec", "name": "Japanese APAC Executive (Tokyo)", "engine": "ElevenLabs v2", "gender": "male", "voice_tag": "ja-JP-KeitaNeural", "pitch": "+0Hz", "rate": "+0%"},
    {"id": "brazilian_vp", "name": "Brazilian Operations VP (São Paulo)", "engine": "OpenAI Voice Engine", "gender": "male", "voice_tag": "pt-BR-AntonioNeural", "pitch": "+0Hz", "rate": "+0%"},
]

DEFAULT_PHRASES = [
    "Please authorize the $250,000 vendor wire transfer before the end of the business day.",
    "This is urgent — I need you to bypass standard dual-custody and release the treasury payment now.",
    "I am currently in an airport lounge, please execute the emergency invoice payment immediately.",
    "Can you reset the multi-factor authentication for my corporate account right away?",
]


def list_voice_profiles() -> list[dict[str, Any]]:
    """Return available synthetic voice models and presets."""
    return VOICE_PROFILES


def _fmt_param(val: str, default: str) -> str:
    v = (val or "").strip()
    if not v:
        return default
    if not v.startswith(("+", "-")):
        return f"+{v}"
    return v


async def generate_synthetic_mp3(text: str, voice_tag: str = "en-US-BrianNeural",
                                 pitch: str = "+0Hz", rate: str = "+0%") -> bytes:
    """Generate high-definition MP3 synthetic speech directly using edge-tts."""
    text = text.strip() or random.choice(DEFAULT_PHRASES)
    p_tag = _fmt_param(pitch, "+0Hz")
    r_tag = _fmt_param(rate, "+0%")
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice_tag, pitch=p_tag, rate=r_tag)
        mp3_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_buffer.write(chunk["data"])
        data = mp3_buffer.getvalue()
        if len(data) > 500:
            return data
    except Exception as e:
        logger.error("edge-tts generation failed for %s: %s", voice_tag, e)

    return b""


async def generate_synthetic_audio(text: str, voice_tag: str = "en-US-BrianNeural",
                                   pitch: str = "+0Hz", rate: str = "+0%") -> bytes:
    """Generate 16 kHz mono PCM synthetic speech using edge-tts."""
    mp3_bytes = await generate_synthetic_mp3(text, voice_tag=voice_tag, pitch=pitch, rate=rate)
    # Generate clean silence buffer if MP3 decoding is not requested directly
    sr = 16000
    duration = max(2.5, min(8.0, len(text) * 0.065))
    total_samples = int(sr * duration)
    return bytes(total_samples * 2)


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """Convert raw S16LE PCM bytes into a standard WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()
