"""
detector.py — the product's detection core. Loads sonave_xlsr_rw once and turns a
chunk of audio into a calibrated verdict. Shared by the API (app.py) and the offline
analyzer (analyze_meeting.py).

Verdict policy (tri-state, tunable): P(fake) is compared to two thresholds so the
product can be cautious rather than binary —
    p < TAU_REAL      -> "real"
    TAU_REAL..TAU_FAKE-> "suspect"   (watch / escalate)
    p >= TAU_FAKE     -> "fake"
Defaults come from the calibrated operating point in results/detector_v2_progress.md
(~64% catch / ~92% real-acc on real-world at tau~0.4). Override via env or config.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np

# make src/ importable (model_sls lives there)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import model_sls  # noqa: E402


def _load_dotenv():
    """Load repo-root .env so threshold/model overrides take effect."""
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

MODEL_DIR = Path(os.environ.get("SONAVE_MODEL", ROOT / "models" / "sonave_xlsr_rw"))
MODEL_VERSION = MODEL_DIR.name
TAU_REAL = float(os.environ.get("SONAVE_TAU_REAL", "0.40"))
TAU_FAKE = float(os.environ.get("SONAVE_TAU_FAKE", "0.70"))

_MODEL = None
_DEVICE = None


def load():
    """Load the detector once (cached)."""
    global _MODEL, _DEVICE
    if _MODEL is None:
        import torch
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = model_sls.SLSDetector.load(MODEL_DIR, _DEVICE)
    return _MODEL, _DEVICE


def verdict(p_fake: float) -> str:
    if p_fake >= TAU_FAKE:
        return "fake"
    if p_fake >= TAU_REAL:
        return "suspect"
    return "real"


def _score_wavs(wavs: list[np.ndarray]) -> list[float]:
    """Score a batch of raw mono 16 kHz arrays -> P(fake) each."""
    import torch
    model, device = load()
    fitted = [model_sls.fit_length(w.astype(np.float32), train=False) for w in wavs]
    inp = model_sls.make_inputs(fitted, device)
    with torch.no_grad():
        probs = torch.softmax(model(**inp), dim=-1)[:, 1]   # P(fake=1)
    return [float(x) for x in probs.detach().cpu().numpy()]


def score_array(wav: np.ndarray) -> dict:
    """Score one mono 16 kHz float array."""
    p = _score_wavs([wav])[0]
    return _result(p)


def score_bytes(audio: bytes) -> dict:
    """Decode wav/flac/ogg bytes to 16 kHz mono and score."""
    import librosa
    import soundfile as sf
    try:
        wav, sr = sf.read(io.BytesIO(audio))
    except Exception:
        # fall back to librosa (handles more containers)
        wav, sr = librosa.load(io.BytesIO(audio), sr=None, mono=True)
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    if sr != model_sls.SR:
        wav = librosa.resample(np.asarray(wav, dtype="float32"),
                               orig_sr=sr, target_sr=model_sls.SR)
    return score_array(np.asarray(wav, dtype="float32"))


def _result(p: float) -> dict:
    # confidence = distance from the decision boundary, scaled to [0,1]
    v = verdict(p)
    conf = min(1.0, abs(p - TAU_REAL) / max(TAU_REAL, 1 - TAU_REAL))
    return {"p_fake": round(p, 4), "verdict": v, "confidence": round(conf, 3),
            "model_version": MODEL_VERSION}


def batch_score_arrays(wavs: list[np.ndarray]) -> list[dict]:
    return [_result(p) for p in _score_wavs(wavs)]
