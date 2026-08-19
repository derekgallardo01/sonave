"""src/pipeline/run_pipeline.py — Master Pipeline CLI & Execution Orchestrator.

Commands:
  - python src/pipeline/run_pipeline.py --mode full --epochs 5 --batch-size 16
  - python src/pipeline/run_pipeline.py --mode eval
  - python src/pipeline/run_pipeline.py --mode ingest
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src to Python path
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
from torch.utils.data import Dataset, DataLoader

from pipeline.data_ingestion import DatasetManifestBuilder, GENERATOR_CATALOG
from pipeline.audio_augmentations import MeetingAudioAugmentor
from pipeline.model_architectures import MultiFoundationAcousticEnsemble
from pipeline.trainer import DeepfakePipelineTrainer
from pipeline.evaluator import DeepfakeBenchmarkEvaluator
from pipeline.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [SONAVE-PIPELINE] %(message)s")
logger = logging.getLogger("sonave.pipeline")


class SyntheticBenchmarkDataset(Dataset):
    """In-memory benchmark dataset simulating multi-corpus inputs with on-the-fly augmentations."""

    def __init__(self, samples: list[dict], augment: bool = False, sample_rate: int = 16000):
        self.samples = samples
        self.augment = augment
        self.sample_rate = sample_rate
        self.augmentor = MeetingAudioAugmentor(sample_rate=sample_rate) if augment else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        duration_sec = s.get("duration_sec", 4.0)
        n_samples = int(self.sample_rate * duration_sec)
        
        # Synthesize realistic testing waveform
        t = torch.linspace(0, duration_sec, n_samples)
        f0 = 140.0 + (s["generator_id"] * 12.0)
        waveform = 0.5 * torch.sin(2 * 3.14159 * f0 * t) + 0.2 * torch.sin(2 * 3.14159 * 2 * f0 * t)
        
        if s["label"] == 1:
            # Inject vocoder artifact
            waveform += 0.1 * torch.sin(2 * 3.14159 * 5500 * t)

        if self.augment and self.augmentor:
            augmented_np = self.augmentor.augment(waveform.numpy())
            waveform = torch.from_numpy(augmented_np).float()

        target_len = int(self.sample_rate * 4.0)
        if waveform.shape[-1] < target_len:
            pad_len = target_len - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        elif waveform.shape[-1] > target_len:
            waveform = waveform[:target_len]

        return {
            "waveform": waveform.unsqueeze(0),
            "label": torch.tensor(s["label"], dtype=torch.long),
            "generator_id": s["generator_id"],
            "generator_name": s["generator_name"]
        }


def execute_full_pipeline(epochs: int = 3, batch_size: int = 16, lr: float = 1e-4) -> dict:
    """Execute complete end-to-end training and evaluation pipeline."""
    logger.info("==================================================================")
    logger.info("🚀 STARTING SONAVE PRODUCTION DEEPFAKE TRAINING PIPELINE")
    logger.info("==================================================================")

    # 1. Ingestion & Stratified Splitting
    builder = DatasetManifestBuilder()
    samples = builder.generate_synthetic_benchmark_corpus(num_samples=160)
    splits = builder.build_stratified_manifests(samples)

    import json
    train_manifest = json.loads(splits["train"].read_text(encoding="utf-8"))
    val_manifest = json.loads(splits["val"].read_text(encoding="utf-8"))
    test_manifest = json.loads(splits["test"].read_text(encoding="utf-8"))

    train_ds = SyntheticBenchmarkDataset(train_manifest["samples"], augment=True)
    val_ds = SyntheticBenchmarkDataset(val_manifest["samples"], augment=False)
    test_ds = SyntheticBenchmarkDataset(test_manifest["samples"], augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. Model Initialization
    model = MultiFoundationAcousticEnsemble()
    trainer = DeepfakePipelineTrainer(model, lr=lr)

    # 3. Training Loop
    history = []
    for epoch in range(1, epochs + 1):
        tr_metrics = trainer.train_epoch(train_loader, epoch, epochs)
        val_metrics = trainer.validate(val_loader)
        history.append({"epoch": epoch, **tr_metrics, **val_metrics})

    # 4. Comprehensive Evaluation
    evaluator = DeepfakeBenchmarkEvaluator(model)
    benchmark_results = evaluator.evaluate_dataset(test_loader)

    # 5. Export to ONNX & Record Lineage
    registry = ModelRegistry()
    run_record = registry.register_training_run(
        model=model,
        train_metrics=history[-1],
        eval_metrics=benchmark_results,
        hyperparams={"epochs": epochs, "batch_size": batch_size, "lr": lr}
    )

    logger.info("==================================================================")
    logger.info("✅ PIPELINE EXECUTION COMPLETED")
    logger.info("Version: %s | Test EER: %.2f%% | Catch@1%%FAR: %.2f%%",
                run_record["model_version"], benchmark_results["equal_error_rate_pct"], benchmark_results["catch_rate_at_1pct_far"])
    logger.info("==================================================================")
    return run_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sonave Deepfake Detection Training Pipeline")
    parser.add_argument("--mode", choices=["full", "ingest", "eval"], default="full")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    if args.mode == "full":
        execute_full_pipeline(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
