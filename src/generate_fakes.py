"""
generate_fakes.py — clone LibriSpeech speakers with XTTS-v2 (the controlled fakes).

Runs in the ISOLATED TTS env (.venv-tts), NOT the detector env:

    .venv-tts\\Scripts\\activate
    python src/generate_fakes.py

Why the split: Coqui/XTTS pins an older torch that conflicts with the detector's
cu128 torch. This step only writes WAV files, so the two envs never run together.

What it does: for each real LibriSpeech clip in the manifest, use that clip as a
voice reference and have XTTS-v2 speak *different* text in that speaker's voice.
The result is a genuine voice-clone of a real speaker — so within the controlled
track, a real clip and its fake differ only in being real vs synthetic (and later,
in compression). Fakes are resampled to 16 kHz mono to match the pipeline, and the
new rows are appended to data/manifest.csv (source=xtts, track=controlled).
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

# XTTS is under a non-commercial license (CPML) and prompts for agreement on
# first load; set this so it runs non-interactively. By running it you accept
# Coqui's CPML terms — fine for this internal validation experiment.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from tqdm import tqdm  # noqa: E402

# Varied, public-domain-ish sentences for the clones to speak. Different words
# than the reference clip => a realistic "someone cloned my voice" sample, not a
# reconstruction of the same utterance.
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
]

XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_MIN_REF_SECONDS = 3.0  # XTTS wants a few seconds of clean reference audio


def _load_libri_reals() -> list[dict]:
    """Read the manifest and return the real LibriSpeech rows to clone from."""
    if not config.MANIFEST.exists():
        raise SystemExit(
            f"No manifest at {config.MANIFEST}. Run prepare_data.py first "
            "(in the .venv detector env)."
        )
    with open(config.MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    libri = [r for r in rows if r["source"] == "libri" and r["label"] == "real"]
    if not libri:
        raise SystemExit("No LibriSpeech real rows in manifest — nothing to clone.")
    return libri


def _append_manifest(new_rows: list[dict]) -> None:
    """Append XTTS fake rows to the manifest without disturbing existing ones."""
    from prepare_data import MANIFEST_COLS  # same column contract

    with open(config.MANIFEST, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writerows(new_rows)


def main() -> None:
    config.ensure_dirs()
    libri = _load_libri_reals()

    from TTS.api import TTS  # coqui-tts fork exposes the same TTS API

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading XTTS-v2 on {device} ...")
    tts = TTS(XTTS_MODEL).to(device)

    n_target = min(config.N_XTTS_FAKE, len(libri))
    print(f"Cloning {n_target} fakes from {len(libri)} real LibriSpeech clips ...")

    new_rows: list[dict] = []
    for i in tqdm(range(n_target)):
        ref_row = libri[i % len(libri)]
        ref_path = config.ROOT / ref_row["path"]

        # XTTS needs a reference of a few seconds; skip clips that are too short.
        dur = librosa.get_duration(path=str(ref_path))
        if dur < XTTS_MIN_REF_SECONDS:
            continue

        text = PHRASES[i % len(PHRASES)]
        out = config.FAKE_XTTS_DIR / f"xtts_{ref_row['speaker']}_{i:03d}.wav"

        # Synthesize to a temp 24 kHz file, then resample to the pipeline rate.
        tmp = config.FAKE_XTTS_DIR / f"_tmp_{i:03d}.wav"
        tts.tts_to_file(
            text=text,
            speaker_wav=str(ref_path),
            language="en",
            file_path=str(tmp),
        )
        wav, _ = librosa.load(str(tmp), sr=config.SAMPLE_RATE, mono=True)
        sf.write(str(out), wav.astype(np.float32), config.SAMPLE_RATE)
        tmp.unlink(missing_ok=True)

        new_rows.append({
            "path": out.relative_to(config.ROOT).as_posix(),
            "label": "fake",
            "source": "xtts",
            "track": config.TRACK_CONTROLLED,
            "speaker": ref_row["speaker"],
        })

    _append_manifest(new_rows)
    print(f"\nGenerated {len(new_rows)} XTTS clones -> {config.FAKE_XTTS_DIR}")
    print(f"Appended {len(new_rows)} rows to {config.MANIFEST}")
    print("Next: run compress.py (back in the .venv detector env).")


if __name__ == "__main__":
    main()
