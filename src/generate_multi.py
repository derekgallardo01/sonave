"""
generate_multi.py — add a SECOND generator architecture (YourTTS) to training.

Runs in the ISOLATED TTS env (.venv-tts):

    python src/generate_multi.py

sonave_v0 trained on ONE generator (XTTS-v2, a GPT-based TTS) generalized only
partially to unseen tools (61% on In-the-Wild). The fix is training on a VARIETY of
synthesis architectures. YourTTS is a good second: a flow-based VITS voice-cloning
model — a genuinely different family than XTTS — and it needs no espeak backend
(the VITS/Tacotron coqui models do, which isn't installed here).

Clones are made from TRAIN-split real clips only and written as kind=modern into the
train split, so the In-the-Wild test stays untouched for a clean before/after.

(Third+ generators like Bark/Tortoise are heavier/slower; add later if the two-
generator jump warrants it.)
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from generate_trainfakes import PHRASES

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from tqdm import tqdm  # noqa: E402

DSET = config.DATA / "dataset"
DSET_CSV = config.DATA / "dataset.csv"
N_YOURTTS = 300


def main() -> None:
    # stdlib csv (the .venv-tts env has no pandas)
    with open(DSET_CSV, newline="", encoding="utf-8") as f:
        rows_in = list(csv.DictReader(f))
    refs = [r["path"] for r in rows_in
            if r["split"] == "train" and r["kind"] == "real"
            and "libri_" in r["path"]]
    if not refs:
        raise SystemExit("No train real libri clips found — run build_trainset.py.")

    from TTS.api import TTS
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading YourTTS on {dev}; generating {N_YOURTTS} clones ...")
    tts = TTS("tts_models/multilingual/multi-dataset/your_tts").to(dev)

    out_dir = DSET / "train" / "fake_modern"
    rows = []
    for i in tqdm(range(N_YOURTTS)):
        ref = config.ROOT / refs[i % len(refs)]
        spk = Path(ref).stem.split("_")[1] if "_" in Path(ref).stem else "x"
        text = PHRASES[i % len(PHRASES)]
        tmp = out_dir / f"_tmp_yt_{i}.wav"
        out = out_dir / f"yourtts_{spk}_{i:03d}.wav"
        try:
            tts.tts_to_file(text=text, speaker_wav=str(ref),
                            language="en", file_path=str(tmp))
        except Exception:
            continue
        wav, _ = librosa.load(str(tmp), sr=config.SAMPLE_RATE, mono=True)
        sf.write(str(out), wav.astype(np.float32), config.SAMPLE_RATE)
        tmp.unlink(missing_ok=True)
        rows.append({"path": out.relative_to(config.ROOT).as_posix(),
                     "label": "fake", "kind": "modern", "split": "train",
                     "speaker": f"yourtts_{spk}"})

    with open(DSET_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["path", "label", "kind", "split", "speaker"]).writerows(rows)
    print(f"\nAdded {len(rows)} YourTTS clones to TRAIN. "
          f"Next: python src/train_detector.py  then  src/cross_eval.py")


if __name__ == "__main__":
    main()
