"""
build_trainset.py — assemble the training/test set for our own detector.

Runs in the detector env (.venv):

    python src/build_trainset.py

Produces a SPEAKER-DISJOINT split so the test set measures generalization, not
memorization: LibriSpeech speakers are partitioned into train vs test, and no
speaker appears in both. Three ingredients:

  real        : LibriSpeech clips (already extracted) + ASVspoof bonafide
  fake_old    : ASVspoof 2019 LA spoof (2019-era attacks) — keep detecting these
  fake_modern : XTTS-v2 clones — generated separately by generate_trainfakes.py,
                which reads the to-generate list this script writes.

Everything lands under data/dataset/ with a single manifest data/dataset.csv:
    path, label(real|fake), kind(real|modern|old), split(train|test), speaker
"""
from __future__ import annotations

import csv
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

DSET = config.DATA / "dataset"
DSET_CSV = config.DATA / "dataset.csv"
TOGEN_CSV = config.DATA / "xtts_to_generate.csv"   # consumed by generate_trainfakes.py

LIBRI_ROOT = config.DOWNLOADS / "LibriSpeech" / "test-clean"
ASV_ZIP = config.DOWNLOADS / "LA.zip"

# Split / sizing knobs.
N_TEST_SPEAKERS = 12          # of 40 LibriSpeech speakers -> test; rest -> train
REAL_PER_SPEAKER = 15         # LibriSpeech real clips per speaker
XTTS_PER_SPEAKER = 12         # modern-fake clones per speaker (generated later)
ASV_TRAIN_PER_CLASS = 300     # ASVspoof bonafide/spoof for train
ASV_TEST_PER_CLASS = 100      # ASVspoof bonafide/spoof for test
MIN_REF_SEC = 3.0             # clips shorter than this can't be XTTS references

COLS = ["path", "label", "kind", "split", "speaker"]


def _write_wav(wav: np.ndarray, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), wav.astype(np.float32), config.SAMPLE_RATE)


def build_libri(rng) -> tuple[list[dict], list[dict]]:
    """Real LibriSpeech clips + the XTTS to-generate list, speaker-disjoint."""
    speakers = sorted(p.name for p in LIBRI_ROOT.iterdir() if p.is_dir())
    rng.shuffle(speakers)
    test_speakers = set(speakers[:N_TEST_SPEAKERS])
    print(f"  speakers: {len(speakers)} total, "
          f"{len(test_speakers)} held out for test")

    rows, togen = [], []
    for spk in speakers:
        split = "test" if spk in test_speakers else "train"
        flacs = sorted((LIBRI_ROOT / spk).rglob("*.flac"))
        # keep only clips long enough to double as XTTS references
        usable = [f for f in flacs
                  if librosa.get_duration(path=str(f)) >= MIN_REF_SEC]
        rng.shuffle(usable)
        picked = usable[:REAL_PER_SPEAKER]
        for f in picked:
            wav, _ = librosa.load(str(f), sr=config.SAMPLE_RATE, mono=True)
            out = DSET / split / "real" / f"libri_{spk}_{f.stem}.wav"
            _write_wav(wav, out)
            rows.append({"path": out.relative_to(config.ROOT).as_posix(),
                         "label": "real", "kind": "real",
                         "split": split, "speaker": spk})
        # queue XTTS clones cloned from this speaker's own real clips
        for i in range(XTTS_PER_SPEAKER):
            if not picked:
                break
            ref = picked[i % len(picked)]
            togen.append({"ref": (DSET / split / "real" /
                                  f"libri_{spk}_{ref.stem}.wav").as_posix(),
                          "speaker": spk, "split": split, "idx": i})
    return rows, togen


def build_asvspoof(rng) -> list[dict]:
    """ASVspoof bonafide (real) + spoof (fake_old), random disjoint split."""
    rows = []
    with zipfile.ZipFile(ASV_ZIP) as z:
        names = z.namelist()
        proto = next(n for n in names if n.endswith("cm.eval.trl.txt"))
        bona, spoof = [], []
        for line in io.TextIOWrapper(z.open(proto), encoding="utf-8"):
            p = line.split()
            if len(p) >= 5:
                (bona if p[-1].lower() == "bonafide" else spoof).append(p[1])
        flac_member = {Path(n).stem: n for n in names
                       if "ASVspoof2019_LA_eval/flac/" in n and n.endswith(".flac")}

        for ids, label, kind in ((bona, "real", "real"), (spoof, "fake", "old")):
            rng.shuffle(ids)
            need = ASV_TRAIN_PER_CLASS + ASV_TEST_PER_CLASS
            chosen = ids[:need]
            for j, fid in enumerate(chosen):
                member = flac_member.get(fid)
                if not member:
                    continue
                split = "train" if j < ASV_TRAIN_PER_CLASS else "test"
                raw = io.BytesIO(z.read(member))
                wav, _ = librosa.load(raw, sr=config.SAMPLE_RATE, mono=True)
                sub = "real" if label == "real" else "fake_old"
                out = DSET / split / sub / f"asv_{kind}_{fid}.wav"
                _write_wav(wav, out)
                rows.append({"path": out.relative_to(config.ROOT).as_posix(),
                             "label": label, "kind": kind,
                             "split": split, "speaker": f"asv_{fid}"})
    return rows


def main() -> None:
    config.ensure_dirs()
    rng = np.random.default_rng(config.SEED)

    print("[1/2] LibriSpeech real + XTTS to-generate list")
    libri_rows, togen = build_libri(rng)
    print(f"  real libri clips: {len(libri_rows)}; XTTS queued: {len(togen)}")

    print("[2/2] ASVspoof bonafide + spoof")
    asv_rows = build_asvspoof(rng)
    print(f"  asv clips: {len(asv_rows)}")

    rows = libri_rows + asv_rows
    with open(DSET_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    with open(TOGEN_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ref", "speaker", "split", "idx"])
        w.writeheader()
        w.writerows(togen)

    # Report the split composition.
    import pandas as pd
    d = pd.DataFrame(rows)
    print("\nDataset so far (modern fakes still to generate):")
    print(d.groupby(["split", "label", "kind"]).size())
    print(f"\nWrote {DSET_CSV} and {TOGEN_CSV}")
    print("Next: python src/generate_trainfakes.py   (in .venv-tts)")


if __name__ == "__main__":
    main()
