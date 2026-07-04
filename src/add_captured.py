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


def main() -> None:
    caps = sorted(glob.glob(str(CAP_DIR / "*.wav")))
    if not caps:
        raise SystemExit(f"No captures in {CAP_DIR}. Do a Meet capture session first.")
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    train_rows, n_test = [], 0
    for cap in caps:
        spk = Path(cap).stem
        wav, _ = librosa.load(cap, sr=config.SAMPLE_RATE, mono=True)
        split = int(0.7 * len(wav))
        tr, te = wav[:split], wav[split:]

        for s, w in _windows(tr, hop=config.SAMPLE_RATE // 2):     # dense 0.5 s
            out = TRAIN_DIR / f"{spk}_tr_{s}.wav"
            sf.write(str(out), w.astype(np.float32), config.SAMPLE_RATE)
            train_rows.append({"path": out.relative_to(config.ROOT).as_posix(),
                               "label": "real", "generator": "meet_real",
                               "split": "train"})
        for s, w in _windows(te, hop=2 * config.SAMPLE_RATE):      # sparse 2 s
            out = TEST_DIR / f"{spk}_te_{s}.wav"
            sf.write(str(out), w.astype(np.float32), config.SAMPLE_RATE)
            n_test += 1

    # corpus_meet.csv = corpus_rw.csv + captured train reals
    with open(BASE_CSV, newline="", encoding="utf-8") as f:
        base = list(csv.DictReader(f))
    cols = list(base[0].keys())
    all_rows = base + train_rows
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    print(f"captured train windows (real, Meet-piped): {len(train_rows)}")
    print(f"held-out test windows (for validation):     {n_test}  -> {TEST_DIR}")
    print(f"wrote {OUT_CSV} ({len(all_rows)} rows)")
    print("Next: python src/train_xlsr.py --manifest data/corpus_meet.csv "
          "--out models/sonave_xlsr_meet --augment")


if __name__ == "__main__":
    main()
