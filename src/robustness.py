"""
robustness.py — adversarial evasion test: how easily can a fake dodge the detector?

Runs in the detector env (.venv):

    python src/robustness.py [--model models/sonave_xlsr_rw]

For a fraud/security product you must know your blind spots before an attacker does.
This applies common, cheap manipulations an attacker would try to make a FAKE read as
real — gain, added noise, re-encoding (mp3/opus), pitch/time shifts, filtering,
reverb — and measures the drop in catch rate. It also checks each trick doesn't spike
false alarms on REAL voices. Big catch-rate drop = a vulnerability to harden (augment
training against that transform).
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import model_sls  # noqa: E402

SR = config.SAMPLE_RATE
N = 80          # clips per class


# --- manipulations (all: 16 kHz float in -> out) -----------------------------
def _norm(x):
    m = np.max(np.abs(x)) + 1e-8
    return (x / m * 0.98).astype(np.float32) if m > 1 else x.astype(np.float32)


def gain(db):
    return lambda x: _norm(x * (10 ** (db / 20)))


def add_noise(snr):
    def f(x):
        n = np.random.normal(0, 1, len(x))
        k = np.sqrt((np.mean(x ** 2) + 1e-9) / (10 ** (snr / 10)))
        return _norm(x + k * n)
    return f


def codec(fmt, bitrate):
    def f(x):
        import soundfile as sf
        import librosa
        with tempfile.TemporaryDirectory() as d:
            wav, enc, out = Path(d) / "i.wav", Path(d) / f"e.{fmt}", Path(d) / "o.wav"
            sf.write(str(wav), x, SR)
            for cmd in ([["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(wav),
                          "-b:a", bitrate] + (["-c:a", "libopus"] if fmt == "opus" else []) + [str(enc)]],
                        [["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(enc),
                          "-ar", str(SR), "-ac", "1", str(out)]]):
                subprocess.run(cmd[0], check=True)
            y, _ = librosa.load(str(out), sr=SR, mono=True)
        return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    return f


def pitch(semitones):
    import librosa
    return lambda x: librosa.effects.pitch_shift(x, sr=SR, n_steps=semitones).astype(np.float32)


def stretch(rate):
    import librosa
    def f(x):
        y = librosa.effects.time_stretch(x, rate=rate)
        return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    return f


def lowpass(cut):
    from scipy.signal import butter, filtfilt
    b, a = butter(4, cut / (SR / 2), btype="low")
    return lambda x: filtfilt(b, a, x).astype(np.float32)


def reverb(x):
    ir = np.exp(-np.linspace(0, 6, int(0.25 * SR))) * np.random.randn(int(0.25 * SR))
    ir[0] = 1.0
    y = np.convolve(x, ir)[:len(x)]
    return _norm(y)


ATTACKS = {
    "clean": lambda x: x,
    "gain +6dB": gain(6), "gain -10dB": gain(-10),
    "noise SNR20": add_noise(20), "noise SNR10": add_noise(10),
    "mp3 64k": codec("mp3", "64k"), "opus 16k": codec("opus", "16k"),
    "pitch +1 semitone": pitch(1), "time 1.06x": stretch(1.06),
    "lowpass 3.4kHz": lowpass(3400), "reverb": reverb,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(config.ROOT / "models" / "sonave_xlsr_rw"))
    args = ap.parse_args()
    import librosa
    import torch

    model = model_sls.SLSDetector.load(Path(args.model),
                                       "cuda" if torch.cuda.is_available() else "cpu")

    def score(wavs):
        ps = []
        for i in range(0, len(wavs), 8):
            b = [model_sls.fit_length(w.astype(np.float32), False) for w in wavs[i:i + 8]]
            dev = next(model.parameters()).device.type
            with torch.no_grad():
                ps += torch.softmax(model(**model_sls.make_inputs(b, dev)), -1)[:, 1].cpu().numpy().tolist()
        return np.array(ps)

    rng = np.random.default_rng(config.SEED)
    fakes = sorted(glob.glob(str(config.DATA / "corpus" / "mlaad" / "test" / "*" / "*.wav")))
    reals = (sorted(glob.glob(str(config.DATA / "dataset" / "test" / "real" / "libri_*.wav")))
             + sorted(glob.glob(str(config.REAL_DIR / "itw_real_*.wav"))))
    rng.shuffle(fakes); rng.shuffle(reals)
    fake_w = [librosa.load(p, sr=SR, mono=True)[0] for p in fakes[:N]]
    real_w = [librosa.load(p, sr=SR, mono=True)[0] for p in reals[:N]]
    print(f"model={Path(args.model).name}  fakes={len(fake_w)} reals={len(real_w)}\n")
    print(f"{'attack':20} {'fake catch%':>11} {'drop':>7} {'real kept%':>11}")
    print("-" * 52)

    base = None
    rows = []
    for name, fn in ATTACKS.items():
        fk = score([fn(w) for w in fake_w])
        rl = score([fn(w) for w in real_w])
        catch = float((fk >= 0.5).mean()) * 100
        kept = float((rl < 0.5).mean()) * 100
        if name == "clean":
            base = catch
        drop = base - catch
        flag = "  <-- EVASION" if drop >= 15 else ("  <- false alarms" if kept < 85 else "")
        print(f"{name:20} {catch:10.1f}% {drop:6.1f} {kept:10.1f}%{flag}")
        rows.append({"attack": name, "fake_catch_%": round(catch, 1),
                     "drop": round(drop, 1), "real_kept_%": round(kept, 1)})

    import pandas as pd
    pd.DataFrame(rows).to_csv(config.RESULTS / "robustness.csv", index=False)
    print(f"\nwrote {config.RESULTS / 'robustness.csv'}  "
          "(attacks with big drops are the ones to augment against)")


if __name__ == "__main__":
    main()
