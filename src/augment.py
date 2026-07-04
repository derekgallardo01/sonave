"""
augment.py — on-the-fly "real call" degradation for training (the product wedge).

Why: the XLS-R model trained on clean studio audio over-flags noisy real-world REAL
speech as fake (In-the-Wild real-acc collapsed to 42%). Real calls are messy —
compressed, band-limited, noisy. Training on degraded audio teaches the detector to
judge fakeness *through* that degradation, which both (a) fixes the false positives
on real-world audio and (b) is Sonave's differentiator: "built for real calls."

Two families, applied at random per clip during training:
  - RawBoost (Tak et al. 2022): the standard anti-spoofing augmentation — convolutive
    + impulsive + stationary colored noise. Boosts generalization to unseen fakes.
  - Channel sim: band-limiting (telephone bandwidth), mu-law companding (telephony),
    additive noise at random SNR, random gain. Cheap in-memory proxies for the
    Meet/Zoom/phone channel. (Real Opus round-trips are done offline via compress.py
    for the degraded TEST set; per-step ffmpeg would bottleneck training.)

All functions take/return float32 mono at 16 kHz.
"""
from __future__ import annotations

import numpy as np

SR = 16_000
_rng = np.random.default_rng()


# --- RawBoost (compact faithful port of the official algorithms) -------------
def _rand_range(x1, x2, integer=False):
    # numpy's Generator.uniform raises when high < low (stdlib random.uniform,
    # which the original RawBoost used, tolerates it) — so order the bounds.
    lo, hi = (x1, x2) if x1 <= x2 else (x2, x1)
    if hi == lo:
        return int(lo) if integer else lo
    y = _rng.uniform(lo, hi)
    return int(y) if integer else y


def _norm(x):
    m = np.max(np.abs(x))
    return x / m if m > 0 else x


def _notch_coeffs(nBands, minF, maxF, minBW, minCoeff, maxCoeff, minG, maxG, fs):
    b = np.array([1.0])
    for _ in range(_rand_range(0, nBands, True)):
        fc = _rand_range(minF, maxF)
        bw = _rand_range(minBW, fc)
        c = _rand_range(minCoeff, maxCoeff, True)
        c = c + 1 if c % 2 == 0 else c
        f = np.arange(c) - (c - 1) / 2
        lo = (fc - bw / 2) / (fs / 2)
        hi = (fc + bw / 2) / (fs / 2)
        f_lo = np.sinc(lo * f) * lo
        f_hi = np.sinc(hi * f) * hi
        h = f_hi - f_lo
        h = h * np.hamming(c)
        g = 10 ** (_rand_range(minG, maxG) / 20)
        b = np.convolve(b, g * h)
    return b


def _fir(x, b):
    return np.convolve(x, b, mode="same")


def _lnl_conv(x, fs):
    b = _notch_coeffs(5, 20, 8000, 100, 10, 100, -5, 0, fs)
    y = _fir(x, b)
    nonlin = _rand_range(1, 3, True)
    z = np.zeros_like(y)
    for i in range(1, nonlin + 1):
        z = z + (y ** i) / i
    return _norm(z)


def _isd_noise(x):
    p = _rand_range(5, 20)
    n = int(x.shape[0] * p / 100)
    idx = _rng.choice(x.shape[0], n, replace=False)
    noise = np.zeros_like(x)
    g = _rand_range(0, 10, True)
    noise[idx] = _rng.uniform(-1, 1, n) * (10 ** (g / 20))
    return _norm(x + noise * x)


def _ssi_noise(x, fs):
    b = _notch_coeffs(5, 20, 8000, 100, 10, 100, -5, 0, fs)
    noise = _rng.normal(0, 1, x.shape[0])
    noise = _fir(noise, b)
    snr = _rand_range(10, 40)
    px = np.mean(x ** 2) + 1e-9
    pn = np.mean(noise ** 2) + 1e-9
    k = np.sqrt(px / (pn * 10 ** (snr / 10)))
    return _norm(x + k * noise)


def rawboost(x, fs=SR):
    """Apply a random RawBoost variant (algos 1/2/3 and their combinations)."""
    algo = _rng.integers(1, 6)   # 1..5
    if algo == 1:
        return _lnl_conv(x, fs)
    if algo == 2:
        return _isd_noise(x)
    if algo == 3:
        return _ssi_noise(x, fs)
    if algo == 4:                # 1 then 2 (series)
        return _isd_noise(_lnl_conv(x, fs))
    return _norm(_lnl_conv(x, fs) + _isd_noise(x))  # 5: parallel


# --- Channel / call simulation ----------------------------------------------
def band_limit(x, fs=SR):
    """Downsample to a random telephone-ish rate then back (bandwidth loss).

    Gentler than a pure 6/8 kHz telephone: bias toward milder rates so we simulate
    Meet/Zoom wideband loss without destroying the fine cues the detector needs.
    """
    import librosa
    target = int(_rng.choice([8000, 12000, 12000, 16000]))
    if target >= fs:
        return x
    down = librosa.resample(x, orig_sr=fs, target_sr=target)
    return librosa.resample(down, orig_sr=target, target_sr=fs)[: len(x)]


def mulaw(x, mu=255):
    """mu-law companding round-trip (telephony non-linearity)."""
    s = np.sign(x)
    c = s * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((c + 1) / 2 * mu) / mu * 2 - 1        # 8-bit quantize
    return np.sign(q) * (1 / mu) * ((1 + mu) ** np.abs(q) - 1)


def add_noise(x):
    snr = _rand_range(12, 35)          # higher SNR floor -> milder noise
    n = _rng.normal(0, 1, x.shape[0])
    px = np.mean(x ** 2) + 1e-9
    k = np.sqrt(px / (10 ** (snr / 10)))
    return x + k * n


def augment(x: np.ndarray, fs: int = SR) -> np.ndarray:
    """Apply a GENTLE random subset of real-call degradations.

    Tuned down from the first (too-aggressive) attempt that wiped out spoof cues:
    lower per-op probabilities, milder intensities, mu-law made rare. The CALLER
    also only augments a FRACTION of clips (see train_xlsr ClipSet), so the model
    still sees plenty of clean audio and keeps its sharp clean-audio detection.
    Returns float32, peak-normalized.
    """
    x = x.astype(np.float32)
    if _rng.random() < 0.6:
        x = rawboost(x, fs)
    if _rng.random() < 0.4:
        x = band_limit(x, fs)
    if _rng.random() < 0.15:           # mu-law is destructive -> rare
        x = mulaw(x)
    if _rng.random() < 0.4:
        x = add_noise(x)
    m = np.max(np.abs(x))
    if m > 0:
        x = x / m * 0.98
    return x.astype(np.float32)
