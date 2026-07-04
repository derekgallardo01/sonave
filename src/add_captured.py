"""
add_captured.py — fold captured real Meet-piped audio into training (the real fix).

Runs in the detector env (.venv):

    python src/add_captured.py

Takes the real voices captured through the live Meet->Recall pipeline
(data/captured/*.wav) and adds them to the corpus as label=real, so the detector
learns that Meet's audio processing is NOT a deepfake artifact.

Honest split: TIME-based (first 70% of each capture -> train, last 30% -> test),
so train and test windows never overlap (no leakage). Train windows are dense
(0.5 s hop) to squeeze signal from limited data; test windows are sparse (2 s hop)
and written out separately for validation.
"""
from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

WIN = 4 * config.SAMPLE_RATE
CAP_DIR = config.DATA / "captured"
TRAIN_DIR = config.DATA / "corpus" / "captured_train"
TEST_DIR = config.DATA / "corpus" / "captured_test"
BASE_CSV = config.DATA / "corpus_rw.csv"          # best corpus so far
OUT_CSV = config.DATA / "corpus_meet.csv"


def _windows(wav, hop):
    for s in range(0, max(1, len(wav) - WIN + 1), hop):
        w = wav[s:s + WIN]
        if np.sqrt(np.mean(w ** 2)) >= 0.005:      # skip silence
            yield s, w


CAP_FAKE_DIR = config.DATA / "captured_fake"      # fake-session captures go here


def _ingest(cap_dir: Path, label: str, gen: str, hold_out_test: bool):
    """Window every capture in a dir. Real captures also hold out a test split."""
    train_rows, n_test = [], 0
    caps = sorted(glob.glob(str(cap_dir / "*.wav")))
    for cap in caps:
        spk = Path(cap).stem
        wav, _ = librosa.load(cap, sr=config.SAMPLE_RATE, mono=True)
        split = int(0.7 * len(wav)) if hold_out_test else len(wav)
        tr, te = wav[:split], wav[split:]
        for s, w in _windows(tr, hop=config.SAMPLE_RATE // 2):     # dense 0.5 s
            out = TRAIN_DIR / f"{label}_{spk}_tr_{s}.wav"
            sf.write(str(out), w.astype(np.float32), config.SAMPLE_RATE)
            train_rows.append({"path": out.relative_to(config.ROOT).as_posix(),
                               "label": label, "generator": gen, "split": "train"})
        for s, w in _windows(te, hop=2 * config.SAMPLE_RATE):      # sparse 2 s test
            out = TEST_DIR / f"{label}_{spk}_te_{s}.wav"
            sf.write(str(out), w.astype(np.float32), config.SAMPLE_RATE)
            n_test += 1
    return train_rows, n_test, len(caps)


def main() -> None:
    if not glob.glob(str(CAP_DIR / "*.wav")) and not glob.glob(str(CAP_FAKE_DIR / "*.wav")):
        raise SystemExit(f"No captures in {CAP_DIR} or {CAP_FAKE_DIR}. Capture first.")
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    real_rows, real_test, n_real = _ingest(CAP_DIR, "real", "meet_real", hold_out_test=True)
    fake_rows, fake_test, n_fake = _ingest(CAP_FAKE_DIR, "fake", "meet_fake", hold_out_test=True)
    train_rows = real_rows + fake_rows

    with open(BASE_CSV, newline="", encoding="utf-8") as f:
        base = list(csv.DictReader(f))
    cols = list(base[0].keys())
    all_rows = base + train_rows
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    print(f"real captures: {n_real} files -> {len(real_rows)} train windows / {real_test} test")
    print(f"fake captures: {n_fake} files -> {len(fake_rows)} train windows / {fake_test} test")
    print(f"wrote {OUT_CSV} ({len(all_rows)} rows)")
    print("Next: python src/train_xlsr.py --manifest data/corpus_meet.csv "
          "--out models/sonave_xlsr_meet --augment")


if __name__ == "__main__":
    main()
