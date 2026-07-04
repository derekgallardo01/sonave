"""
prepare_asvspoof.py — add the IN-DISTRIBUTION track: ASVspoof 2019 LA eval.

Runs in the detector env (.venv):

    python src/prepare_asvspoof.py

Why this track exists: the OOD tracks (XTTS clones, In-the-Wild) never gave the
detector a strong CLEAN baseline — it's blind to modern XTTS, and In-the-Wild is
already-compressed with a weak baseline. You cannot measure "a clean-accurate
detector craters under compression" without a clean-accurate starting point.
ASVspoof 2019 LA is the detector's home turf (the "ASV" in its name), so clean EER
should be low here — a real baseline the Opus sweep can then degrade.

We only need ~300 clips, so we stream the protocol + extract just the sampled flac
straight out of the 7.6 GB LA.zip rather than unpacking all ~71k eval files.

After running: re-run compress.py (skips already-done clips) then evaluate.py.
The new track shows up automatically as track="asvspoof".
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402

from prepare_data import _download, _write_wav, MANIFEST_COLS  # reuse helpers

# Edinburgh DataShare (authoritative). If this 404s, download LA.zip by hand from
# https://datashare.ed.ac.uk/handle/10283/3336 and drop it at ASV_ZIP.
# DSpace-7 UUID bitstream link for LA.zip (~7.6 GB). Resolved from the record page
# https://datashare.ed.ac.uk/handle/10283/3336 — the old /bitstream/handle path now
# returns an HTML stub.
ASV_URL = "https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download"
ASV_ZIP = config.DOWNLOADS / "LA.zip"

PROTO_SUFFIX = "cm.eval.trl.txt"          # eval protocol file
FLAC_DIR_HINT = "ASVspoof2019_LA_eval/flac/"


def _read_flac_from_zip(z: zipfile.ZipFile, member: str) -> np.ndarray:
    """Extract one flac member fully into memory and load as 16 kHz mono."""
    raw = io.BytesIO(z.read(member))       # zip streams aren't seekable; buffer it
    wav, _ = librosa.load(raw, sr=config.SAMPLE_RATE, mono=True)
    return wav


def main() -> None:
    config.ensure_dirs()
    print("ASVspoof 2019 LA eval (in-distribution track)")
    if not _download(ASV_URL, ASV_ZIP):
        raise SystemExit(
            "ASVspoof LA.zip download failed. Download it manually from\n"
            "  https://datashare.ed.ac.uk/handle/10283/3336  (the LA.zip file)\n"
            f"and place it at {ASV_ZIP}, then re-run this script."
        )

    with zipfile.ZipFile(ASV_ZIP) as z:
        names = z.namelist()
        proto = next((n for n in names if n.endswith(PROTO_SUFFIX)), None)
        if not proto:
            raise SystemExit(f"Eval protocol ({PROTO_SUFFIX}) not found in zip.")

        # Protocol line: <speaker> <file> <-> <attack_id> <bonafide|spoof>
        bona, spoof = [], []
        for line in io.TextIOWrapper(z.open(proto), encoding="utf-8"):
            parts = line.split()
            if len(parts) < 5:
                continue
            fid, key = parts[1], parts[-1].lower()
            (bona if key == "bonafide" else spoof).append(fid)

        rng = np.random.default_rng(config.SEED)
        rng.shuffle(bona)
        rng.shuffle(spoof)
        bona = bona[: config.N_ASV_REAL]
        spoof = spoof[: config.N_ASV_FAKE]
        print(f"  sampled {len(bona)} bonafide / {len(spoof)} spoof from eval")

        # Map file id -> its flac member path inside the zip.
        flac_member = {
            Path(n).stem: n for n in names
            if FLAC_DIR_HINT in n and n.endswith(".flac")
        }

        rows = []
        for group, label, out_dir, prefix in (
            (bona, "real", config.REAL_DIR, "asv_real"),
            (spoof, "fake", config.FAKE_ASV_DIR, "asv_fake"),
        ):
            for fid in group:
                member = flac_member.get(fid)
                if not member:
                    continue
                wav = _read_flac_from_zip(z, member)
                out = out_dir / f"{prefix}_{fid}.wav"
                _write_wav(wav, out)
                rows.append({
                    "path": out.relative_to(config.ROOT).as_posix(),
                    "label": label,
                    "source": "asv",
                    "track": config.TRACK_INDIST,
                    "speaker": "asv",
                })

    _append_manifest(rows)
    n_real = sum(1 for r in rows if r["label"] == "real")
    n_fake = sum(1 for r in rows if r["label"] == "fake")
    print(f"  wrote {n_real} bonafide + {n_fake} spoof, appended to manifest")
    print("Next: python src/compress.py  (skips done clips)  then  evaluate.py")


def _append_manifest(new_rows: list[dict]) -> None:
    import csv
    with open(config.MANIFEST, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writerows(new_rows)


if __name__ == "__main__":
    main()
