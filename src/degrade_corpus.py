"""
degrade_corpus.py — add REAL Opus-compressed copies of the training set (Stage 3b).

Runs in the detector env (.venv):

    python src/degrade_corpus.py

The Stage-3 gentle augmentation used a resample proxy for compression. This adds the
REAL thing: every TRAIN clip (real AND fake) is round-tripped through the actual
Google-Meet Opus codec (reusing src/compress.py opus_roundtrip) at a random VoIP
bitrate, and added as extra training rows. So the model literally sees Meet-quality
audio at train time — the core of "built for real calls". TEST stays clean here;
compressed testing is done separately by degrading In-the-Wild.

Output: data/corpus_aug.csv = corpus.csv + the Opus-degraded train rows.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from compress import opus_roundtrip  # reuse the real Opus pipeline

import numpy as np  # noqa: E402

CORPUS_CSV = config.DATA / "corpus.csv"
OUT_CSV = config.DATA / "corpus_aug.csv"
DEGRADED_DIR = config.DATA / "corpus" / "degraded"


def main() -> None:
    with open(CORPUS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = list(rows[0].keys())
    train = [r for r in rows if r["split"] == "train"]
    rng = np.random.default_rng(config.SEED)
    print(f"Opus-degrading {len(train)} train clips (real Meet codec)...")

    new_rows = []
    done = 0
    for r in train:
        src = config.ROOT / r["path"]
        if not src.exists():
            continue
        bitrate = str(rng.choice(config.OPUS_BITRATES))
        dest = DEGRADED_DIR / f"{Path(r['path']).stem}__opus{bitrate}.wav"
        try:
            opus_roundtrip(src, dest, bitrate)
        except Exception:
            continue
        nr = dict(r)
        nr["path"] = dest.relative_to(config.ROOT).as_posix()
        nr["generator"] = r["generator"] + f"|opus{bitrate}"
        new_rows.append(nr)
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{len(train)}", flush=True)

    all_rows = rows + new_rows
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    import pandas as pd
    d = pd.DataFrame(all_rows)
    print(f"\nAdded {len(new_rows)} Opus-degraded train clips.")
    print(d.groupby(["split", "label"]).size())
    print(f"wrote {OUT_CSV} ({len(all_rows)} rows)")
    print("Next: python src/train_xlsr.py --manifest data/corpus_aug.csv "
          "--out models/sonave_xlsr_final --augment")


if __name__ == "__main__":
    main()
