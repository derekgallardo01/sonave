"""
make_meetify_corpus.py — build a Meet-domain training set by Meet-ifying the corpus.

Runs in the detector env (.venv):

    python src/make_meetify_corpus.py

Samples a balanced set of REAL and FAKE clips from corpus_rw.csv, runs each through
src/meetify.py (offline Meet channel), and adds them to training with the SAME
labels. This teaches the detector both classes in the Meet domain -> fixes the
false-positive AND recovers fake-catch, at scale, offline.

The captured real Meet audio (data/captured/, data/corpus/captured_test/) is left
ENTIRELY out, so it stays a clean ground-truth test of whether offline meetifying
matches real Meet.

Output: data/corpus_meetify.csv = corpus_rw.csv + meetified clips.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import meetify  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

BASE = config.DATA / "corpus_rw.csv"
OUT = config.DATA / "corpus_meetify.csv"
MDIR = config.DATA / "corpus" / "meetified"
N_PER_CLASS = 700


def main() -> None:
    MDIR.mkdir(parents=True, exist_ok=True)
    with open(BASE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = list(rows[0].keys())
    train = [r for r in rows if r["split"] == "train"]
    reals = [r for r in train if r["label"] == "real"]
    fakes = [r for r in train if r["label"] == "fake"]

    rng = np.random.default_rng(config.SEED)
    rng.shuffle(reals)
    rng.shuffle(fakes)
    pick = reals[:N_PER_CLASS] + fakes[:N_PER_CLASS]
    print(f"meetifying {len(pick)} clips ({min(len(reals),N_PER_CLASS)} real / "
          f"{min(len(fakes),N_PER_CLASS)} fake)...")

    new_rows, done = [], 0
    for r in pick:
        src = config.ROOT / r["path"]
        if not src.exists():
            continue
        try:
            w, _ = librosa.load(str(src), sr=config.SAMPLE_RATE, mono=True)
            m = meetify.meetify(w)
        except Exception:
            continue
        out = MDIR / f"meet_{Path(r['path']).stem}.wav"
        sf.write(str(out), m.astype(np.float32), config.SAMPLE_RATE)
        nr = dict(r)
        nr["path"] = out.relative_to(config.ROOT).as_posix()
        nr["generator"] = r["generator"] + "|meetified"
        new_rows.append(nr)
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(pick)}", flush=True)

    all_rows = rows + new_rows
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nadded {len(new_rows)} meetified clips -> {OUT} ({len(all_rows)} rows)")
    print("Next: python src/train_xlsr.py --manifest data/corpus_meetify.csv "
          "--out models/sonave_xlsr_meetify --augment")


if __name__ == "__main__":
    main()
