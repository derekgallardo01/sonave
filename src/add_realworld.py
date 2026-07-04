"""
add_realworld.py — add REAL-WORLD real speech to training (close the last gap).

Runs in the detector env (.venv):

    python src/add_realworld.py

The persistent In-the-Wild false-positive problem is a DOMAIN gap: the detector only
ever heard clean studio real voices, so it flags noisy real-world real speech as
fake. Synthetic degradation didn't fix it (Stage 3b). The real fix is real-world REAL
audio in training. VoxPopuli = real parliamentary recordings (varied speakers, mics,
rooms, background) — genuine real-world real speech, and it streams cleanly at 16 kHz.

We pull a sample, add it as label=real into the diverse corpus (based on corpus.csv,
NOT the Stage-3b doubled one, since doubling hurt), and write data/corpus_rw.csv.
In-the-Wild stays the external real-world test (different source), so improvement
there is genuine generalization.
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

N_CLIPS = 600
MAX_SEC = 8
RW_DIR = config.DATA / "corpus" / "realworld"
CORPUS_CSV = config.DATA / "corpus.csv"
OUT_CSV = config.DATA / "corpus_rw.csv"


def main() -> None:
    from datasets import load_dataset, Audio

    RW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Streaming VoxPopuli English, taking {N_CLIPS} real-world real clips...")
    ds = load_dataset("facebook/voxpopuli", "en", split="train",
                      streaming=True).cast_column("audio", Audio(decode=False))

    rows = []
    for ex in ds:
        raw = ex["audio"].get("bytes")
        if not raw:
            continue
        try:
            wav, sr = sf.read(io.BytesIO(raw))
        except Exception:
            continue
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        if sr != config.SAMPLE_RATE:
            wav = librosa.resample(wav.astype("float32"), orig_sr=sr,
                                   target_sr=config.SAMPLE_RATE)
        wav = wav[: MAX_SEC * config.SAMPLE_RATE]
        if len(wav) < config.SAMPLE_RATE:          # skip <1 s
            continue
        i = len(rows)
        dest = RW_DIR / f"vp_{i:04d}.wav"
        sf.write(str(dest), wav.astype(np.float32), config.SAMPLE_RATE)
        rows.append({"path": dest.relative_to(config.ROOT).as_posix(),
                     "label": "real", "generator": "realworld_voxpopuli",
                     "split": "train"})
        if len(rows) % 100 == 0:
            print(f"  {len(rows)}/{N_CLIPS}", flush=True)
        if len(rows) >= N_CLIPS:
            break

    with open(CORPUS_CSV, newline="", encoding="utf-8") as f:
        base = list(csv.DictReader(f))
    cols = list(base[0].keys())
    all_rows = base + rows
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    import pandas as pd
    d = pd.DataFrame(all_rows)
    print(f"\nAdded {len(rows)} real-world real clips.")
    print(d.groupby(["split", "label"]).size())
    print(f"wrote {OUT_CSV} ({len(all_rows)} rows)")
    print("Next: python src/train_xlsr.py --manifest data/corpus_rw.csv "
          "--out models/sonave_xlsr_rw --augment")


if __name__ == "__main__":
    main()
