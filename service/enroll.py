"""
enroll.py — speaker enrollment / voiceprint verification (the differentiator).

Deepfake detection alone asks the hard question "is this audio synthetic?".
Enrollment adds an easier, independent one: "is this the person it's supposed to
be?". If you have a known-real voiceprint for someone (from past calls), a live
voice that doesn't match is a red flag EVEN when the deepfake score is uncertain —
and vice versa. Two independent signals -> far fewer false positives AND stronger
catches, especially for the wire-fraud vertical where the caller claims an identity.

Uses ECAPA-TDNN (SpeechBrain) speaker embeddings — a wide same-vs-different gap even
on short 4 s windows (~0.6 same-speaker vs ~0.05 impostor). Voiceprints are the L2-
normalized mean embedding over a person's enrollment clips.

API:
    enroll(speaker_id, wav_paths)      -> saves a voiceprint
    verify(speaker_id, audio)          -> {similarity, match}
    fused_risk(p_fake, speaker_id, audio) -> combined verdict for the product
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

ENROLL_DIR = config.ROOT / "enrollments"
# Cosine sim above this = same speaker. With ECAPA the gap is wide (~0.6 same vs ~0.05
# impostor) even at 4 s, so a mid threshold is robust. Calibrate per-deployment if needed.
MATCH_THRESHOLD = 0.35

_ENC = None


def _enc():
    """Load ECAPA-TDNN once. Patches SpeechBrain's lazy loader so its unused optional
    integrations (k2/flair/...) can't crash on Windows, and copies model files instead
    of symlinking (no admin needed)."""
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
        _ENC = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(config.ROOT / "models" / "ecapa"),
            run_opts={"device": dev},
            local_strategy=LocalStrategy.COPY_SKIP_CACHE)
    return _ENC


def embed(source) -> np.ndarray:
    """Speaker embedding from a wav path or a 16 kHz float array."""
    import torch
    if isinstance(source, (str, Path)):
        import librosa
        wav, _ = librosa.load(str(source), sr=config.SAMPLE_RATE, mono=True)
    else:
        wav = np.asarray(source, dtype=np.float32)
    t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
    e = _enc().encode_batch(t).squeeze().detach().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-8)


def _cos(a, b) -> float:
    return float(np.dot(a, b))     # inputs are L2-normalized


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


def verify(speaker_id: str, source, threshold: float = MATCH_THRESHOLD) -> dict:
    """Compare audio against an enrolled voiceprint."""
    f = ENROLL_DIR / f"{speaker_id}.npy"
    if not f.exists():
        return {"speaker": speaker_id, "enrolled": False}
    vp = np.load(f)
    sim = _cos(embed(source), vp)
    return {"speaker": speaker_id, "enrolled": True,
            "similarity": round(sim, 3), "match": sim >= threshold}


def fused_risk(p_fake: float, speaker_id: str | None = None, source=None) -> dict:
    """Combine the deepfake score with speaker verification into one verdict.

    - No enrolled identity: fall back to the deepfake score alone.
    - Enrolled + claimed identity: a mismatch is itself high risk. Risk is the
      MAX of 'looks synthetic' and 'not the claimed person'.
    """
    out = {"p_fake": round(p_fake, 3)}
    mismatch = 0.0        # 'not the claimed person' risk (raises)
    match_conf = 0.0      # 'confidently the claimed person' (dampens deepfake jitter)
    if speaker_id and source is not None:
        v = verify(speaker_id, source)
        out["speaker_check"] = v
        if v.get("enrolled"):
            sim = v["similarity"]
            mismatch = float(np.clip((MATCH_THRESHOLD - sim) / 0.20, 0, 1))
            match_conf = float(np.clip((sim - MATCH_THRESHOLD) / 0.15, 0, 1))
    # a strong voiceprint match trusts the voice as real (dampens up to 70% of the
    # deepfake score); a mismatch is its own high risk.
    damped = p_fake * (1 - 0.7 * match_conf)
    risk = max(damped, mismatch)
    out["mismatch_risk"] = round(mismatch, 3)
    out["match_conf"] = round(match_conf, 3)
    out["risk"] = round(risk, 3)
    out["verdict"] = "fake" if risk >= 0.7 else "suspect" if risk >= 0.4 else "real"
    return out
