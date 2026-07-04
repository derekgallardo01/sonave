"""
meetify.py — apply a Google-Meet-style processing chain to audio, offline.

The live false-positive came from Meet's WebRTC audio processing (the real APM
module won't build on Windows, so this reproduces its main effects with signal
processing) + the Opus codec. Running our existing clips through this "Meet-ifies"
them, so we can generate thousands of Meet-domain training clips of BOTH classes
without live meetings.

Chain (order matches WebRTC APM):
  high-pass (remove rumble) -> noise suppression (spectral gate) -> AGC
  (normalize loudness + gentle compression) -> Opus round-trip (real codec).

This is an APPROXIMATION — its whole justification is empirical: we validate that
training on meetified audio makes the model score REAL captured Meet audio (ground
truth) as real. If it does, the approximation is good enough.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

SR = config.SAMPLE_RATE


def high_pass(x, cutoff=100.0):
    from scipy.signal import butter, filtfilt
    b, a = butter(2, cutoff / (SR / 2), btype="high")
    return filtfilt(b, a, x).astype(np.float32)


def noise_suppress(x, reduction=0.7):
    """Spectral-gate NS: estimate a noise floor from the quietest frames and
    softly attenuate spectral bins near it (the smoothing/'musical noise' that
    makes processed speech look synthetic to the detector)."""
    import librosa
    S = librosa.stft(x, n_fft=512, hop_length=256)
    mag, phase = np.abs(S), np.angle(S)
    # noise floor = 10th percentile magnitude per frequency bin
    floor = np.percentile(mag, 10, axis=1, keepdims=True)
    mask = mag / (mag + reduction * floor + 1e-8)      # soft Wiener-ish gate
    mag2 = mag * mask
    y = librosa.istft(mag2 * np.exp(1j * phase), hop_length=256, length=len(x))
    return y.astype(np.float32)


def agc(x, target_rms=0.06, comp=0.6):
    """Automatic gain: normalize toward a target loudness with gentle compression."""
    rms = np.sqrt(np.mean(x ** 2)) + 1e-8
    x = x * (target_rms / rms)
    # soft-knee compression toward the target (reduce dynamic range)
    x = np.sign(x) * (np.abs(x) ** comp) * (target_rms ** (1 - comp))
    peak = np.max(np.abs(x)) + 1e-8
    if peak > 0.98:
        x = x / peak * 0.98
    return x.astype(np.float32)


def meetify(x: np.ndarray, bitrate: str = "24k") -> np.ndarray:
    """Full offline Meet channel. Returns 16 kHz mono float32."""
    x = high_pass(x.astype(np.float32))
    x = noise_suppress(x)
    x = agc(x)
    x = _opus(x, bitrate)
    m = np.max(np.abs(x)) + 1e-8
    return (x / m * 0.98).astype(np.float32)


def _opus(x, bitrate):
    """Real Opus round-trip via ffmpeg (reuse the pipeline)."""
    import soundfile as sf
    from compress import opus_roundtrip
    tmp_in = config.DATA / "_meetify_in.wav"
    tmp_out = config.DATA / "_meetify_out.wav"
    sf.write(str(tmp_in), x, SR)
    try:
        opus_roundtrip(tmp_in, tmp_out, bitrate)
        import librosa
        y, _ = librosa.load(str(tmp_out), sr=SR, mono=True)
    except Exception:
        y = x
    finally:
        tmp_in.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)
    return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


if __name__ == "__main__":
    import glob
    import soundfile as sf
    import librosa
    p = sorted(glob.glob(str(config.DATA / "corpus" / "captured_test" / "*.wav")))
    if p:
        w, _ = librosa.load(p[0], sr=SR, mono=True)
        m = meetify(w)
        print(f"meetify smoke: in {len(w)} -> out {len(m)}, "
              f"finite={np.isfinite(m).all()}, rms={np.sqrt(np.mean(m**2)):.4f}")
