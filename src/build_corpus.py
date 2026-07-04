"""
build_corpus.py — assemble a GENERATOR-DIVERSE corpus (the real ceiling-breaker).

Runs in the detector env (.venv):

    python src/build_corpus.py

Our In-the-Wild EER stuck at ~15% because training saw only 2 similar neural
cloners + 2019 ASVspoof. This pulls a small sample from MANY English MLAAD TTS
systems (175 systems total; we take a broad subset) and splits them GENERATOR-wise:
some systems go to train, others are HELD OUT for test — so the test measures
generalization to fake *methods* never seen in training. In-the-Wild stays fully
external on top of that.

English-only (fake/en/...) to avoid the "non-English == fake" shortcut, since our
real clips are English.

Output: data/corpus.csv  (path,label,generator,split), reusing our existing real +
XTTS/YourTTS/ASVspoof clips and adding the MLAAD fakes.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

MLAAD_REPO = "mueller91/MLAAD"
CORPUS_DIR = config.DATA / "corpus" / "mlaad"
CORPUS_CSV = config.DATA / "corpus.csv"

CLIPS_PER_MODEL = 20          # small sample per generator -> breadth over depth
TEST_MODEL_FRACTION = 0.30    # hold out ~30% of generators entirely for test
MAX_MODELS = 90               # cap total English models used
COLS = ["path", "label", "generator", "split"]


def _write16k(src: str, dest: Path) -> bool:
    try:
        wav, _ = librosa.load(src, sr=config.SAMPLE_RATE, mono=True)
        if len(wav) < config.SAMPLE_RATE // 2:      # skip <0.5 s junk
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), wav.astype(np.float32), config.SAMPLE_RATE)
        return True
    except Exception:
        return False


def fetch_mlaad() -> list[dict]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    wavs = [f for f in api.list_repo_files(MLAAD_REPO, repo_type="dataset")
            if f.startswith("fake/en/") and f.endswith(".wav")]
    by_model: dict[str, list[str]] = defaultdict(list)
    for f in wavs:
        by_model[f.split("/")[2]].append(f)   # fake/en/<model>/...
    models = sorted(by_model)
    rng = np.random.default_rng(config.SEED)
    rng.shuffle(models)
    models = models[:MAX_MODELS]
    n_test = int(len(models) * TEST_MODEL_FRACTION)
    test_models = set(models[:n_test])
    print(f"MLAAD English generators: {len(models)} used "
          f"({n_test} held out for test)")

    rows = []
    for mi, model in enumerate(models):
        split = "test" if model in test_models else "train"
        files = sorted(by_model[model])
        rng.shuffle(files)
        got = 0
        for f in files:
            if got >= CLIPS_PER_MODEL:
                break
            try:
                local = hf_hub_download(MLAAD_REPO, f, repo_type="dataset")
            except Exception:
                continue
            dest = CORPUS_DIR / split / model / Path(f).name
            if _write16k(local, dest):
                rows.append({"path": dest.relative_to(config.ROOT).as_posix(),
                             "label": "fake", "generator": f"mlaad:{model}",
                             "split": split})
                got += 1
        if (mi + 1) % 10 == 0:
            print(f"  {mi+1}/{len(models)} models, {len(rows)} clips", flush=True)
    return rows


def reuse_existing() -> list[dict]:
    """Fold in our existing real + XTTS/YourTTS/ASVspoof clips from dataset.csv."""
    src = config.DATA / "dataset.csv"
    if not src.exists():
        return []
    rows = []
    with open(src, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gen = {"real": "real", "old": "asvspoof", "modern": "local"}.get(r["kind"], r["kind"])
            rows.append({"path": r["path"], "label": r["label"],
                         "generator": gen if r["label"] == "fake" else "real",
                         "split": r["split"]})
    return rows


def main() -> None:
    config.ensure_dirs()
    print("Downloading MLAAD English subset (small sample per generator)...")
    mlaad = fetch_mlaad()
    existing = reuse_existing()
    rows = existing + mlaad

    with open(CORPUS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    import pandas as pd
    d = pd.DataFrame(rows)
    print("\nCorpus composition:")
    print(d.groupby(["split", "label"]).size())
    print(f"\ndistinct generators: {d['generator'].nunique()} "
          f"(train-only held out from test = generalization signal)")
    print(f"MLAAD test generators (unseen): "
          f"{d[(d.split=='test') & d.generator.str.startswith('mlaad')].generator.nunique()}")
    print(f"\nwrote {CORPUS_CSV} ({len(rows)} rows)")
    print("Next: python src/train_xlsr.py --manifest data/corpus.csv --out models/sonave_xlsr_corpus")


if __name__ == "__main__":
    main()
