"""
detect.py — load an open-source anti-spoofing model and score a WAV.

Public contract (used by evaluate.py):

    score_wav(path: str | Path) -> float        # P(fake) in [0, 1], higher = faker
    score_batch(paths: list) -> list[float]

Run directly for the smoke test:

    python src/detect.py                 # loads model, prints CUDA status + a
                                         # real-vs-fake margin on two sample clips

The smoke test is the go/no-go gate for the detector itself: if the model can't
tell an obvious real clip from an obvious fake with a clear margin, we swap to the
AASIST fallback BEFORE building the rest of the pipeline. See notes at the bottom.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

# Make `import config` work whether run as `python src/detect.py` or imported.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


# --- Model loading -----------------------------------------------------------
@lru_cache(maxsize=1)
def _load():
    """
    Load the HF audio-classification model + feature extractor once, cached.

    Returns (model, feature_extractor, device, fake_index) where fake_index is
    the logit column that corresponds to the 'fake/spoof' class — we resolve it
    from the model's id2label map so we never hardcode a column and get the
    polarity backwards.
    """
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = config.DETECTOR_HF_MODEL

    extractor = AutoFeatureExtractor.from_pretrained(name)
    model = AutoModelForAudioClassification.from_pretrained(name).to(device).eval()

    fake_index = _resolve_fake_index(model.config.id2label)
    return model, extractor, device, fake_index


def _resolve_fake_index(id2label: dict[int, str]) -> int:
    """
    Find which output column means 'fake'. Different checkpoints label their
    classes differently (fake/spoof/deepfake vs real/bonafide/genuine), so match
    on keywords instead of assuming an order.
    """
    fake_words = ("fake", "spoof", "deepfake", "synthetic")
    real_words = ("real", "bonafide", "bona-fide", "genuine", "human")
    for idx, label in id2label.items():
        if any(w in label.lower() for w in fake_words):
            return int(idx)
    # Fall back: if only 'real' is identifiable, fake is the other class.
    for idx, label in id2label.items():
        if any(w in label.lower() for w in real_words):
            return 1 - int(idx)
    raise RuntimeError(
        f"Could not map id2label={id2label} to a fake/real class. "
        "Inspect the model card and set fake_index manually."
    )


# --- Audio loading -----------------------------------------------------------
def _load_audio(path: str | Path) -> np.ndarray:
    """Load a WAV as mono float32 at the model's expected sample rate."""
    import librosa

    wav, _ = librosa.load(str(path), sr=config.SAMPLE_RATE, mono=True)
    return wav.astype(np.float32)


# --- Scoring -----------------------------------------------------------------
def score_batch(paths: list[str | Path]) -> list[float]:
    """Score a list of WAV paths. Returns P(fake) in [0,1] per path."""
    import torch

    model, extractor, device, fake_index = _load()
    scores: list[float] = []
    # Small batches keep VRAM flat on the 8 GB card; clips are short.
    for path in paths:
        wav = _load_audio(path)
        inputs = extractor(
            wav, sampling_rate=config.SAMPLE_RATE, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        scores.append(float(probs[fake_index].item()))
    return scores


def score_wav(path: str | Path) -> float:
    """Score a single WAV. Returns P(fake) in [0,1] (higher = more likely fake)."""
    return score_batch([path])[0]


# --- Smoke test --------------------------------------------------------------
def _smoke_test() -> int:
    """
    Prove the detector works before we build anything on it.

    We need two example clips: a known-real and a known-fake. If prepare_data.py /
    generate_fakes.py have already run, we pull one from each of data/real and
    data/fake. Otherwise we print guidance and exit non-zero.
    """
    import torch

    print("=== Sonave detector smoke test ===")
    print(f"torch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("!! WARNING: running on CPU. On the RTX 5060 this means the cu128")
        print("   torch install step was skipped — see requirements.txt.")

    print(f"\nLoading model: {config.DETECTOR_HF_MODEL} ...")
    model, _, _, fake_index = _load()
    print(f"id2label       : {model.config.id2label}")
    print(f"fake_index     : {fake_index}  "
          f"(-> '{model.config.id2label[fake_index]}')")

    # Score a small BATCH per class, not a single pair. Deepfake scores are noisy
    # per-clip (any one clip can be an outlier), so a 1-vs-1 comparison is a weak
    # gate — we judge the detector on mean separation + AUC across ~20 clips each.
    reals = _sample_wavs(config.REAL_DIR, 20)
    fakes = (_sample_wavs(config.FAKE_XTTS_DIR, 20)
             or _sample_wavs(config.FAKE_ITW_DIR, 20))
    if not reals or not fakes:
        print("\n[smoke test incomplete] Need real + fake clips on disk.")
        print(f"  real found: {len(reals)}  fake found: {len(fakes)}")
        print("Run prepare_data.py (and generate_fakes.py) first, then re-run.")
        return 2

    rs = np.array(score_batch(reals))
    fs = np.array(score_batch(fakes))
    margin = float(fs.mean() - rs.mean())
    auc = _auc(rs, fs)
    print(f"\nreal clips (n={len(rs)}): mean P(fake) = {rs.mean():.3f}")
    print(f"fake clips (n={len(fs)}): mean P(fake) = {fs.mean():.3f}")
    print(f"mean margin (fake - real) = {margin:+.3f}")
    print(f"separation AUC            = {auc:.3f}")

    # AUC >= 0.75 means the detector meaningfully ranks fakes above reals.
    if auc >= 0.75:
        print("\nPASS — detector discriminates. Proceed with the pipeline.")
        return 0
    print("\nFAIL — separation too weak. Pick a better detector (see "
          "scratchpad/shootout.py) or the AASIST fallback before building on it.")
    return 1


def _auc(real_scores: np.ndarray, fake_scores: np.ndarray) -> float:
    """Probability a random fake outranks a random real (Mann-Whitney / ROC AUC)."""
    from sklearn.metrics import roc_auc_score

    y = np.r_[np.zeros(len(real_scores)), np.ones(len(fake_scores))]
    s = np.r_[real_scores, fake_scores]
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, s))


def _sample_wavs(directory: Path, n: int) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.rglob("*.wav"))[:n]
    return None


if __name__ == "__main__":
    raise SystemExit(_smoke_test())


# -----------------------------------------------------------------------------
# AASIST fallback (only if the smoke test above FAILS):
#   The primary path is a wav2vec2-based HF classifier because SSL front-ends
#   generalize better to unseen fake types. If it can't discriminate, vendor the
#   official AASIST model (clovaai/aasist) into src/models/ and reimplement
#   _load()/score_batch() to run raw 4s @16kHz waveforms through it, mapping its
#   single spoof logit through a sigmoid to P(fake). Keep the same public
#   contract (score_wav / score_batch) so evaluate.py doesn't change.
# -----------------------------------------------------------------------------
