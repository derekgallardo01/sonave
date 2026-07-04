"""
generate_trainfakes.py — generate the modern-fake (XTTS-v2) clones for training.

Runs in the ISOLATED TTS env (.venv-tts):

    .venv-tts\\Scripts\\activate
    python src/generate_trainfakes.py

Reads data/xtts_to_generate.csv (written by build_trainset.py), clones each listed
LibriSpeech reference with XTTS-v2 speaking VARIED text, and appends the results to
data/dataset.csv as kind=modern. Test-speaker clones stay in the test split, so the
detector is later judged on modern fakes from speakers it never trained on.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from tqdm import tqdm  # noqa: E402

DSET = config.DATA / "dataset"
DSET_CSV = config.DATA / "dataset.csv"
TOGEN_CSV = config.DATA / "xtts_to_generate.csv"
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# A larger, varied sentence pool so the model can't learn fixed phrasing as a tell.
PHRASES = [
    "Please confirm the wire transfer before the end of the business day.",
    "I authorize the payment to the account we discussed this morning.",
    "The quarterly figures look strong, let's move ahead with the deal.",
    "Can you hear me clearly? The connection has been a little rough today.",
    "Send the signed documents over and we'll finalize everything by Friday.",
    "I'm calling to verify the details on the invoice you sent yesterday.",
    "Let's schedule the follow-up meeting for early next week if possible.",
    "The board approved the budget, so we can proceed with hiring.",
    "Thanks for your patience, the update will be ready within the hour.",
    "Make sure the contract is reviewed by legal before we sign it.",
    "I appreciate you taking the time to walk me through the proposal.",
    "We should double-check those numbers before the client call.",
    "The weather has been unpredictable, so pack an umbrella just in case.",
    "History reminds us that progress is rarely a straight line.",
    "She placed the old photographs carefully back into the wooden box.",
    "Our team will present the findings at the conference next month.",
    "A gentle breeze carried the scent of pine across the quiet valley.",
    "Remember to back up your files before installing the new software.",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "He walked along the shoreline, listening to the crashing waves.",
    "Innovation often begins with a simple question that no one asked.",
    "The train departs at nine, so we should leave the house by eight.",
    "Their conversation drifted from politics to music to old memories.",
    "Please review the attached report and share your feedback soon.",
    "The children laughed as the kite climbed higher into the sky.",
    "Patience and persistence usually matter more than raw talent.",
    "We reserved a table for four at the restaurant near the river.",
    "The museum's new exhibit explores a century of scientific discovery.",
    "Could you forward that email to the rest of the department?",
    "Under the streetlight, the city felt calm and almost empty.",
]


def _append(rows: list[dict]) -> None:
    with open(DSET_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "kind", "split", "speaker"])
        w.writerows(rows)


def main() -> None:
    if not TOGEN_CSV.exists():
        raise SystemExit(f"{TOGEN_CSV} missing — run build_trainset.py first.")
    with open(TOGEN_CSV, newline="", encoding="utf-8") as f:
        todo = list(csv.DictReader(f))

    from TTS.api import TTS
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading XTTS-v2 on {device}; generating {len(todo)} clones ...")
    tts = TTS(XTTS_MODEL).to(device)

    rows = []
    for e in tqdm(todo):
        ref = config.ROOT / e["ref"]
        if not ref.exists():
            continue
        split, spk, idx = e["split"], e["speaker"], int(e["idx"])
        text = PHRASES[(hash(spk) + idx) % len(PHRASES)]
        out = DSET / split / "fake_modern" / f"xtts_{spk}_{idx:03d}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"_tmp_{spk}_{idx}.wav")
        try:
            tts.tts_to_file(text=text, speaker_wav=str(ref),
                            language="en", file_path=str(tmp))
        except Exception:
            continue
        wav, _ = librosa.load(str(tmp), sr=config.SAMPLE_RATE, mono=True)
        sf.write(str(out), wav.astype(np.float32), config.SAMPLE_RATE)
        tmp.unlink(missing_ok=True)
        rows.append({"path": out.relative_to(config.ROOT).as_posix(),
                     "label": "fake", "kind": "modern",
                     "split": split, "speaker": spk})

    _append(rows)
    n_tr = sum(1 for r in rows if r["split"] == "train")
    n_te = sum(1 for r in rows if r["split"] == "test")
    print(f"\nGenerated {len(rows)} modern clones "
          f"({n_tr} train / {n_te} test), appended to {DSET_CSV}")
    print("Next: python src/train_detector.py   (in .venv)")


if __name__ == "__main__":
    main()
