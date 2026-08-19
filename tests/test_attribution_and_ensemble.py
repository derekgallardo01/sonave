"""Tests for Attribution Fingerprinting, Ambient Mismatch, and Multi-Foundation Ensemble:
  - Acoustic attribution and engine identification (ElevenLabs, OpenAI Voice, XTTS, RVC)
  - Ambient acoustic mismatch and room reverberation estimation
  - MultiFoundationAcousticEnsemble forward pass & Anti-Adversarial head
  - UniversalDeepfakeBenchmarkDataset loader
  - Integration with /api/quality endpoint
"""
import sys
from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def test_attribution_and_ambient_mismatch():
    import attribution

    # Below threshold -> None
    assert attribution.attribute_synthesis_engine(0.40) is None

    # Above threshold -> Engine identified
    attr = attribution.attribute_synthesis_engine(0.88, speaker_name="Caller_Target")
    assert attr is not None
    assert "engine_name" in attr
    assert "anomaly_band" in attr
    assert attr["attribution_confidence"] > 0.70

    # Ambient mismatch
    amb_clean = attribution.compute_ambient_mismatch(0.15)
    assert amb_clean["is_mismatched"] is False

    amb_fake = attribution.compute_ambient_mismatch(0.92)
    assert amb_fake["is_mismatched"] is True
    assert "Studio-Dry" in amb_fake["status"]


def test_multi_foundation_ensemble_forward_pass():
    from models.ensemble import MultiFoundationAcousticEnsemble, AntiAdversarialDefenseHead

    # Test Defense Head
    defense = AntiAdversarialDefenseHead(p_augment=1.0)
    defense.train()
    x = torch.randn(2, 50, 1024)
    x_out = defense(x)
    assert x_out.shape == x.shape

    # Test Full Ensemble Model
    model = MultiFoundationAcousticEnsemble(embed_dim=1024, num_generators=10)
    audio = torch.randn(4, 16000 * 2)  # 4 samples of 2s audio
    
    # Standalone forward pass
    out = model(audio)
    assert out["logits_binary"].shape == (4, 2)
    assert out["logits_attribution"].shape == (4, 10)
    assert out["prob_fake"].shape == (4,)
    assert (out["prob_fake"] >= 0.0).all() and (out["prob_fake"] <= 1.0).all()


def test_benchmark_dataset_loader():
    from datasets.benchmark_loader import UniversalDeepfakeBenchmarkDataset

    ds = UniversalDeepfakeBenchmarkDataset()
    ds.add_sample("fake_path_1.wav", label=1, generator_id=2, speaker="Target_A")
    ds.add_sample("fake_path_2.wav", label=0, generator_id=0, speaker="Target_B")

    assert len(ds) == 2
    item = ds[0]
    assert item["waveform"].shape == (ds.max_samples,)
    assert item["label"].item() == 1
    assert item["generator_id"].item() == 2
    assert item["speaker"] == "Target_A"


def test_api_quality_attribution_integration(railway_mod, tmp_path, monkeypatch):
    import db
    monkeypatch.setattr(db, "_db_path", lambda: tmp_path / "test_attr.db")

    u = railway_mod.db.upsert_google_user("sub_attr", "tester@test.com", "Tester", "", "admin")
    uid = u["id"]
    sess = railway_mod.auth.sign_session(uid)
    client = TestClient(railway_mod.app)
    client.cookies.set("sonave_session", sess)

    # Mock quality state with (uid, spk) tuple key
    railway_mod.QUALITY[(uid, "spk_test_impersonator")] = {
        "state": "speaking",
        "total_sec": 45.0,
        "quiet_sec": 0.0,
        "speech_pct": 92.0,
        "level": 0.35,
        "clips": 10
    }
    railway_mod.VERDICTS[(uid, "spk_test_impersonator")] = {
        "verdict": "fake",
        "rolling": 0.93,
        "n": 8,
        "latency_ms": 42
    }

    r = client.get("/api/quality")
    assert r.status_code == 200
    d = r.json()
    spk_data = d.get("spk_test_impersonator")
    assert spk_data is not None
    assert spk_data["attribution"] is not None
    assert "engine_name" in spk_data["attribution"]
    assert spk_data["ambient"]["is_mismatched"] is True
