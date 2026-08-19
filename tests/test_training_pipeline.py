"""tests/test_training_pipeline.py — Comprehensive Unit & Integration Tests for Training Pipeline."""
import json
import torch
import pytest
from pathlib import Path

from src.pipeline.data_ingestion import DatasetManifestBuilder, GENERATOR_CATALOG
from src.pipeline.audio_augmentations import MeetingAudioAugmentor
from src.pipeline.model_architectures import MultiFoundationAcousticEnsemble, PhaseSpectralResidualCNN
from src.pipeline.trainer import DeepfakePipelineTrainer, FocalLoss
from src.pipeline.evaluator import DeepfakeBenchmarkEvaluator, compute_eer, compute_catch_rate_at_far
from src.pipeline.model_registry import ModelRegistry


def test_data_ingestion_manifest_builder(tmp_path):
    builder = DatasetManifestBuilder(data_root=tmp_path)
    samples = builder.generate_synthetic_benchmark_corpus(num_samples=40)
    assert len(samples) == 40
    assert any(s["label"] == 0 for s in samples)
    assert any(s["label"] == 1 for s in samples)

    splits = builder.build_stratified_manifests(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    assert "train" in splits and "val" in splits and "test" in splits

    train_data = json.loads(splits["train"].read_text(encoding="utf-8"))
    assert train_data["split"] == "train"
    assert len(train_data["samples"]) > 0


def test_hf_corpus_harvester(tmp_path):
    from src.pipeline.hf_corpus_harvester import HFCorpusHarvester
    harvester = HFCorpusHarvester(cache_dir=tmp_path / "hf_corpora")

    corpora = harvester.list_supported_hf_corpora()
    assert len(corpora) >= 4
    assert any(c["category"] == "neural_vocoders" for c in corpora)
    assert any(c["category"] == "flow_matching_diffusion" for c in corpora)

    samples = harvester.sync_huggingface_manifests(max_samples_per_corpus=10)
    assert len(samples) > 0
    assert harvester.manifest_file.exists()


def test_meeting_audio_augmentations():
    augmentor = MeetingAudioAugmentor(sample_rate=16000)
    dummy_audio = torch.randn(32000).numpy() # 2 seconds

    # Test individual transforms
    plc_audio = augmentor.apply_packet_loss(dummy_audio)
    assert len(plc_audio) == len(dummy_audio)

    reverb_audio = augmentor.apply_reverb(dummy_audio)
    assert len(reverb_audio) == len(dummy_audio)

    opus_audio = augmentor.apply_opus_simulation(dummy_audio)
    assert len(opus_audio) == len(dummy_audio)

    noise_audio = augmentor.apply_background_noise(dummy_audio)
    assert len(noise_audio) == len(dummy_audio)

    # Test full chain
    chain_audio = augmentor.augment(dummy_audio)
    assert len(chain_audio) == len(dummy_audio)
    assert abs(chain_audio).max() <= 1.05


def test_model_architecture_forward():
    model = MultiFoundationAcousticEnsemble()
    waveforms = torch.randn(4, 1, 64000) # Batch of 4 4.0s audio clips

    outputs = model(waveforms)
    assert "logits" in outputs
    assert "probs" in outputs
    assert "fake_score" in outputs
    assert "embeddings" in outputs

    assert outputs["logits"].shape == (4, 2)
    assert outputs["probs"].shape == (4, 2)
    assert outputs["fake_score"].shape == (4,)
    assert (outputs["fake_score"] >= 0.0).all() and (outputs["fake_score"] <= 1.0).all()


def test_focal_loss_and_trainer():
    model = MultiFoundationAcousticEnsemble()
    trainer = DeepfakePipelineTrainer(model, lr=1e-3, device="cpu")

    # Mock dataset
    mock_batch = {
        "waveform": torch.randn(4, 1, 32000),
        "label": torch.tensor([0, 1, 0, 1], dtype=torch.long)
    }
    dataloader = [mock_batch, mock_batch]

    metrics = trainer.train_epoch(dataloader, epoch=1, total_epochs=1)
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert metrics["loss"] > 0

    val_metrics = trainer.validate(dataloader)
    assert "val_loss" in val_metrics
    assert "val_accuracy" in val_metrics
    assert "val_catch_rate" in val_metrics


def test_evaluator_metrics():
    # Test EER with perfectly separated scores
    bonafide = torch.tensor([0.05, 0.10, 0.15, 0.20]).numpy()
    spoof = torch.tensor([0.80, 0.85, 0.90, 0.95]).numpy()
    eer, th = compute_eer(bonafide, spoof)
    assert eer == 0.0

    catch_rate = compute_catch_rate_at_far(bonafide, spoof, target_far=0.01)
    assert catch_rate == 1.0


def test_model_registry_and_lineage(tmp_path):
    registry = ModelRegistry(checkpoints_dir=tmp_path / "checkpoints")
    model = MultiFoundationAcousticEnsemble()

    run_record = registry.register_training_run(
        model=model,
        train_metrics={"loss": 0.05, "accuracy": 0.95},
        eval_metrics={"equal_error_rate_pct": 2.5, "catch_rate_at_1pct_far": 97.5},
        hyperparams={"epochs": 2, "batch_size": 16}
    )

    assert "model_version" in run_record
    assert Path(run_record["pytorch_checkpoint"]).exists()


def test_adversarial_hardened_benchmark():
    from src.pipeline.hardened_benchmark import AdversarialBenchmarkEngine
    model = MultiFoundationAcousticEnsemble()
    engine = AdversarialBenchmarkEngine(num_samples=40)
    results = engine.run_hardened_evaluation(model, device="cpu")

    assert results["total_adversarial_samples"] == 40
    assert "hardened_equal_error_rate_pct" in results
    assert "in_the_wild_adversarial_catch_pct" in results
    assert len(results["generator_catch_matrix"]) > 0
