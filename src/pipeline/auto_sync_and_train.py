"""src/pipeline/auto_sync_and_train.py — Automated Continuous Learning Pipeline.

Fully automates:
  1. Hugging Face Discovery & Manifest Harvest (tracking newest TTS/VC models & benchmark corpora).
  2. Production Capture Synchronization (pulls latest real/fake session audio from Railway).
  3. Corpus Windowing & Balancing (src/add_captured.py -> data/corpus_meet.csv).
  4. Multi-Foundation Ensemble Retraining & TorchScript Compilation.
  5. XLS-R Meeting Detector Retraining with audio augmentation.
  6. Regression Gate Verification & Metric Updates.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [SONAVE-AUTO] %(message)s"
)
logger = logging.getLogger("sonave.auto")

PY = str(_ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PY).exists():
    PY = sys.executable


def run_cmd(args: list[str], desc: str) -> None:
    logger.info(">>> STEP: %s", desc)
    logger.info("Running: %s", " ".join(args))
    res = subprocess.run([PY, *args], cwd=str(_ROOT))
    if res.returncode != 0:
        logger.error("Step failed with exit code %d: %s", res.returncode, desc)
        raise RuntimeError(f"Step failed: {desc}")
    logger.info("Completed: %s", desc)


def step_hf_sync(limit: int = 25) -> dict:
    """Discover trending Hugging Face speech models and synchronize corpora manifests."""
    logger.info(">>> STEP: Hugging Face Model Discovery & Corpus Harvesting")
    from src.pipeline.hf_corpus_harvester import HFCorpusHarvester
    harvester = HFCorpusHarvester()
    
    # 1. Discover trending speech/TTS models
    models = harvester.discover_trending_hf_models(limit=limit)
    logger.info("Discovered %d trending Hugging Face models", len(models))
    
    # 2. Sync datasets manifests
    manifest_samples = harvester.sync_huggingface_manifests()
    logger.info("Synchronized %d dataset samples across Hugging Face corpora", len(manifest_samples))
    return {"models_tracked": len(models), "samples_harvested": len(manifest_samples)}


def step_pull_captures(url: str = "https://sonave-production-3ca2.up.railway.app") -> None:
    """Pull fresh meeting audio captures from Railway production."""
    logger.info(">>> STEP: Pull Production Audio Captures")
    try:
        run_cmd(["src/pull_captures.py", url], "Pull Real Audio Captures")
    except Exception as e:
        logger.warning("Pull captures encountered an issue (continuing with local data): %s", e)


def step_window_corpus() -> None:
    """Window and balance captured audio into the training corpus."""
    logger.info(">>> STEP: Ingest & Window Captures")
    run_cmd(["src/add_captured.py"], "Ingest & Window Captures into corpus_meet.csv")


def step_train_ensemble(epochs: int = 2, batch_size: int = 16) -> None:
    """Train the Multi-Foundation Acoustic Ensemble."""
    logger.info(">>> STEP: Train Multi-Foundation Acoustic Ensemble")
    run_cmd([
        "src/pipeline/run_pipeline.py",
        "--mode", "full",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size)
    ], "Multi-Foundation Ensemble Training")


def step_retrain_xlsr(epochs: int = 5, batch_size: int = 8) -> None:
    """Retrain the production XLS-R meeting detector head."""
    logger.info(">>> STEP: Retrain XLS-R Meeting Detector Head")
    run_cmd([
        "src/train_xlsr.py",
        "--manifest", "data/corpus_meet.csv",
        "--out", "models/sonave_xlsr_meet",
        "--epochs", str(epochs),
        "--batch", str(batch_size),
        "--augment"
    ], "XLS-R + SLS Meeting Detector Training")


def step_validate_and_test() -> None:
    """Validate on held-out captured meeting windows and run GPU regression gates."""
    logger.info(">>> STEP: Model Validation & Quality Gates")
    run_cmd(["src/retrain_from_captures.py", "--validate-only"], "Held-Out Audio Split Validation")
    run_cmd(["-m", "pytest", "-m", "gpu", "tests/test_model_regression.py", "-q"], "GPU Regression Gate")


def update_scheduler_metadata() -> None:
    """Record the automated run in scheduler_config.json."""
    config_file = _ROOT / "models" / "scheduler_config.json"
    data = {}
    if config_file.exists():
        try:
            data = json.loads(config_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    data["last_run_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    data["status"] = "active"
    config_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sonave Automated Model Sync & Retrain Orchestrator")
    parser.add_argument("--skip-pull", action="store_true", help="Skip pulling captures from Railway")
    parser.add_argument("--skip-ensemble", action="store_true", help="Skip training the ensemble pipeline")
    parser.add_argument("--skip-xlsr", action="store_true", help="Skip training the XLS-R model")
    parser.add_argument("--ensemble-epochs", type=int, default=2, help="Ensemble training epochs")
    parser.add_argument("--xlsr-epochs", type=int, default=5, help="XLS-R training epochs")
    parser.add_argument("--hf-limit", type=int, default=25, help="Limit of trending HF models to discover")
    args = parser.parse_args()

    t_start = datetime.datetime.now()
    logger.info("==================================================================")
    logger.info("AUTOMATED SONAVE SYNC & RETRAIN PIPELINE STARTING")
    logger.info("Timestamp: %s", t_start.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("==================================================================")

    # 1. Hugging Face Discovery & Harvester Sync
    hf_stats = step_hf_sync(limit=args.hf_limit)

    # 2. Remote Audio Sync
    if not args.skip_pull:
        step_pull_captures()

    # 3. Audio Ingestion & Dataset Windowing
    step_window_corpus()

    # 4. Multi-Foundation Ensemble Retrain
    if not args.skip_ensemble:
        step_train_ensemble(epochs=args.ensemble_epochs)

    # 5. XLS-R Meeting Detector Retrain
    if not args.skip_xlsr:
        step_retrain_xlsr(epochs=args.xlsr_epochs)

    # 6. Quality Gates & Regression Verification
    step_validate_and_test()

    # 7. Update scheduler metadata
    update_scheduler_metadata()

    elapsed = datetime.datetime.now() - t_start
    logger.info("==================================================================")
    logger.info("AUTOMATED PIPELINE COMPLETED in %s", elapsed)
    logger.info("HF Models Tracked: %d | Manifest Samples: %d", hf_stats["models_tracked"], hf_stats["samples_harvested"])
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
