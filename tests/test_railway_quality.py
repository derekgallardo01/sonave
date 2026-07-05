"""The audio-quality math on the Railway capture path — the stdlib rewrite that once
broke production (audioop removed in 3.13). These run on synthetic PCM, no audio libs."""
import math

from conftest import pcm16


def test_quality_ignores_empty_pcm(railway_mod):
    railway_mod._quality("spk", b"")
    assert "spk" not in railway_mod.QUALITY  # empty -> no-op, never crashes


def test_quality_odd_byte_pcm_does_not_crash(railway_mod):
    # odd length (dropped byte mid-sample) must be tolerated, not raise
    railway_mod._quality("spk", b"\x01\x02\x03")
    assert "spk" in railway_mod.QUALITY


def test_silence_reads_too_quiet(railway_mod):
    railway_mod._quality("spk", pcm16([0.0] * 16000 * 4))  # 4 s of silence
    v = railway_mod._quality_verdict(railway_mod.QUALITY["spk"])
    assert v == "TOO QUIET — raise volume"


def test_loud_tone_reads_good(railway_mod):
    tone = [0.3 * math.sin(2 * math.pi * 220 * i / 16000) for i in range(16000 * 4)]
    railway_mod._quality("spk", pcm16(tone))
    q = railway_mod.QUALITY["spk"]
    assert q["level"] > 0.01 and q["speech_sec"] > 0
    assert railway_mod._quality_verdict(q) == "good"


def test_clipping_detected(railway_mod):
    q = railway_mod.QUALITY.setdefault("spk", {"level": 0.5, "peak": 0.0, "clips": 0,
                                               "speech_sec": 10.0, "total_sec": 10.0})
    railway_mod._quality("spk", pcm16([0.999, -0.999] * 8000))  # full-scale -> peak~1
    assert railway_mod.QUALITY["spk"]["peak"] >= 0.985
    assert railway_mod._quality_verdict(railway_mod.QUALITY["spk"]) == "CLIPPING — lower volume"


def test_warming_up_under_3s(railway_mod):
    railway_mod._quality("spk", pcm16([0.3] * 16000))  # 1 s only
    assert railway_mod._quality_verdict(railway_mod.QUALITY["spk"]) == "warming up"


def test_total_sec_accumulates(railway_mod):
    for _ in range(3):
        railway_mod._quality("spk", pcm16([0.2] * 16000 * 2))  # 2 s each
    assert railway_mod.QUALITY["spk"]["total_sec"] == 6.0
