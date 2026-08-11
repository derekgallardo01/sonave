"""
Railway-local enrollment module (vendored from service/enroll.py).
No external repo dependencies — uses env vars for paths.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ENROLL_DIR = Path(os.environ.get("SONAVE_ENROLL_DIR", "/data/enrollments"))
# Cosine sim above this = same speaker. With ECAPA the gap is wide (~0.6 same vs ~0.05
# impostor) even at 4 s, so a mid threshold is robust.
MATCH_THRESHOLD = 0.35

_ENC = None


def _enc():
    """Load ECAPA-TDNN once."""
    global _ENC
    if _ENC is None:
        import types
        import torch
        import speechbrain.utils.importutils as iu
        _orig = iu.LazyModule.ensure_module

        def _safe(self, stacklevel):
            try:
                return _orig(self, stacklevel)
            except Exception:
                m = types.ModuleType(self.target)
                self.lazy_module = m
                return m
        iu.LazyModule.ensure_module = _safe

        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        cache_dir = Path(os.environ.get("SONAVE_MODEL_CACHE", "/data/models/ecapa"))
        _ENC = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(cache_dir),
            run_opts={"device": dev},
            local_strategy=LocalStrategy.COPY_SKIP_CACHE)
    return _ENC


def embed(source) -> np.ndarray:
    """Speaker embedding from a wav path or a 16 kHz float array."""
    import torch
    if isinstance(source, (str, Path)):
        import librosa
        wav, _ = librosa.load(str(source), sr=16_000, mono=True)
    else:
        wav = np.asarray(source, dtype=np.float32)
    t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
    e = _enc().encode_batch(t).squeeze().detach().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-8)


def _cos(a, b) -> float:
    return float(np.dot(a, b))


def is_enrolled(speaker_id: str) -> bool:
    return (ENROLL_DIR / f"{speaker_id}.npy").exists()


def list_enrolled() -> list:
    return [p.stem for p in ENROLL_DIR.glob("*.npy")] if ENROLL_DIR.exists() else []


def enroll(speaker_id: str, wav_paths: list) -> np.ndarray:
    """Build + persist a voiceprint from several real clips of one person."""
    ENROLL_DIR.mkdir(parents=True, exist_ok=True)
    embs = [embed(p) for p in wav_paths]
    vp = np.mean(embs, axis=0)
    vp = vp / (np.linalg.norm(vp) + 1e-8)
    np.save(ENROLL_DIR / f"{speaker_id}.npy", vp)
    return vp
