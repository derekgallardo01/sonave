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


def verify(speaker_id: str, source, threshold: float = MATCH_THRESHOLD) -> dict:
    """Compare audio against an enrolled voiceprint."""
    f = ENROLL_DIR / f"{speaker_id}.npy"
    if not f.exists():
        return {"speaker": speaker_id, "enrolled": False}
    vp = np.load(f)
    return verify_with_voiceprint(speaker_id, source, vp, threshold)


def verify_with_voiceprint(speaker_id: str, source, voiceprint: np.ndarray,
                           threshold: float = MATCH_THRESHOLD) -> dict:
    """Compare audio against a provided voiceprint embedding (e.g. from base64)."""
    sim = _cos(embed(source), voiceprint)
    return {"speaker": speaker_id, "enrolled": True,
            "similarity": round(sim, 3), "match": sim >= threshold}


def fused_risk(p_fake: float, speaker_id: str | None = None, source=None) -> dict:
    """Combine the deepfake score with speaker verification into one verdict."""
    out = {"p_fake": round(p_fake, 3)}
    mismatch = 0.0
    match_conf = 0.0
    if speaker_id and source is not None:
        v = verify(speaker_id, source)
        out["speaker_check"] = v
        if v.get("enrolled"):
            sim = v["similarity"]
            mismatch = float(np.clip((MATCH_THRESHOLD - sim) / 0.20, 0, 1))
            match_conf = float(np.clip((sim - MATCH_THRESHOLD) / 0.15, 0, 1))
    damped = p_fake * (1 - 0.7 * match_conf)
    risk = max(damped, mismatch)
    out["mismatch_risk"] = round(mismatch, 3)
    out["match_conf"] = round(match_conf, 3)
    out["risk"] = round(risk, 3)
    out["verdict"] = "fake" if risk >= 0.7 else "suspect" if risk >= 0.4 else "real"
    return out


def fused_risk_with_voiceprint(p_fake: float, speaker_id: str, source,
                               voiceprint: np.ndarray) -> dict:
    """Like fused_risk but with a pre-loaded voiceprint (e.g. from base64 inline)."""
    out = {"p_fake": round(p_fake, 3)}
    v = verify_with_voiceprint(speaker_id, source, voiceprint)
    out["speaker_check"] = v
    sim = v["similarity"]
    mismatch = float(np.clip((MATCH_THRESHOLD - sim) / 0.20, 0, 1))
    match_conf = float(np.clip((sim - MATCH_THRESHOLD) / 0.15, 0, 1))
    damped = p_fake * (1 - 0.7 * match_conf)
    risk = max(damped, mismatch)
    out["mismatch_risk"] = round(mismatch, 3)
    out["match_conf"] = round(match_conf, 3)
    out["risk"] = round(risk, 3)
    out["verdict"] = "fake" if risk >= 0.7 else "suspect" if risk >= 0.4 else "real"
    return out
