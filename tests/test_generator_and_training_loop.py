"""Tests for Synthetic Voice Generator, Test Injection, and Continuous HF Training Loop:
  - Voice profiles & presets
  - Synthetic audio generation & PCM/WAV conversion
  - Continuous training loop execution & lineage tracking
  - API Endpoints: /api/generator/voices, /api/generator/synthesize, /api/generator/inject-test
"""
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.mark.asyncio
async def test_generator_audio_synthesis():
    import generator

    voices = generator.list_voice_profiles()
    assert len(voices) >= 3
    assert any(v["id"] in ("derek_natural", "me_clone") for v in voices)

    # Test synthesis
    pcm = await generator.generate_synthetic_audio("Test audio phrase", voice_tag="en-US-GuyNeural")
    assert len(pcm) > 1000
    assert len(pcm) % 2 == 0  # S16LE 2-byte aligned

    wav = generator.pcm_to_wav_bytes(pcm)
    assert wav.startswith(b"RIFF")


def test_continuous_training_loop(tmp_path, monkeypatch):
    import training_loop
    lineage_path = tmp_path / "lineage.json"
    monkeypatch.setattr(training_loop, "LINEAGE_FILE", lineage_path)

    run = training_loop.run_continuous_training_iteration(epochs=1, batch_size=8)
    assert run is not None
    assert "final_accuracy" in run
    assert lineage_path.exists()

    lin = training_loop.load_lineage()
    assert len(lin["runs"]) == 1
    assert lin["total_epochs_trained"] == 1


def test_generator_http_endpoints(railway_mod, tmp_path, monkeypatch):
    import db
    import incidents
    monkeypatch.setattr(db, "_db_path", lambda: tmp_path / "test_gen.db")
    monkeypatch.setattr(incidents, "DB_PATH", tmp_path / "test_gen_inc.db")

    u = railway_mod.db.upsert_google_user("sub_gen", "operator@corp.com", "Operator", "", "admin")
    uid = u["id"]
    sess = railway_mod.auth.sign_session(uid)
    client = TestClient(railway_mod.app)
    client.cookies.set("sonave_session", sess)

    # 1. GET /api/generator/voices
    r_v = client.get("/api/generator/voices")
    assert r_v.status_code == 200
    d_v = r_v.json()
    assert "voices" in d_v
    assert "default_phrases" in d_v

    # 2. POST /api/generator/synthesize
    r_syn = client.post("/api/generator/synthesize", json={"text": "Test synthesis", "voice_id": "me_clone"})
    assert r_syn.status_code == 200
    assert "audio/" in r_syn.headers["content-type"]
    assert len(r_syn.content) > 100

    # 3. POST /api/generator/inject-test
    r_inj = client.post("/api/generator/inject-test", json={"text": "Urgent wire transfer", "voice_id": "me_clone", "speaker_name": "Derek (Clone)"})
    assert r_inj.status_code == 200
    d_inj = r_inj.json()
    assert d_inj["ok"] is True
    assert d_inj["p_fake"] > 0.95
    assert d_inj["audio_base64"] is not None
