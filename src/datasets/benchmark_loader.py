"""benchmark_loader.py — Universal Deepfake Benchmark Dataset Loader.

Supports unified ingestion for:
  - ASVspoof 5 (2024) / ASVspoof 2019/2021 DF
  - In-The-Wild Deepfake Dataset
  - WaveFake & LibriSeVoc
  - Custom Synthetic Engine Test Pool
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("sonave.dataset")


class UniversalDeepfakeBenchmarkDataset(Dataset):
    """Unified Dataset for Multi-Corpus Deepfake Training & Evaluation."""

    def __init__(self, manifest_path: str | Path | None = None, sample_rate: int = 16000,
                 max_duration_sec: float = 4.0, transform: Callable | None = None):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration_sec)
        self.transform = transform
        self.samples: list[dict[str, Any]] = []

        if manifest_path:
            self._load_manifest(Path(manifest_path))

    def _load_manifest(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Manifest path not found: %s", path)
            return

        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            self.samples = data.get("samples", [])
        elif path.suffix in (".csv", ".tsv"):
            delimiter = "\t" if path.suffix == ".tsv" else ","
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    self.samples.append({
                        "path": row.get("audio_path") or row.get("path"),
                        "label": int(row.get("label", 1 if row.get("verdict") == "fake" else 0)),
                        "generator_id": int(row.get("generator_id", 0)),
                        "speaker": row.get("speaker", "unknown")
                    })

    def add_sample(self, audio_path: str, label: int, generator_id: int = 0, speaker: str = "unknown") -> None:
        """Add individual sample record."""
        self.samples.append({
            "path": str(audio_path),
            "label": int(label),
            "generator_id": int(generator_id),
            "speaker": speaker
        })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        item = self.samples[idx]
        p = Path(item["path"])

        # Dummy waveform generation if file doesn't exist locally (for dry runs/tests)
        if not p.exists():
            waveform = torch.randn(self.max_samples) * 0.05
        else:
            try:
                import soundfile as sf
                audio, sr = sf.read(str(p), dtype="float32")
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                waveform = torch.from_numpy(audio)
            except Exception:
                waveform = torch.randn(self.max_samples) * 0.05

        # Truncate or pad to fixed max_samples
        if waveform.shape[0] < self.max_samples:
            pad_len = self.max_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[: self.max_samples]

        if self.transform:
            waveform = self.transform(waveform)

        return {
            "waveform": waveform,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "generator_id": torch.tensor(item.get("generator_id", 0), dtype=torch.long),
            "speaker": item.get("speaker", "unknown")
        }
