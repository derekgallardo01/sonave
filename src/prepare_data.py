"""
prepare_data.py — build the labelled test set and write data/manifest.csv.

Two tracks:
  controlled : real LibriSpeech utterances (fakes come later from generate_fakes.py)
  benchmark  : a slice of the In-the-Wild deepfake dataset (real + fake, trusted labels)

Run in the detector env (.venv):

    python src/prepare_data.py

LibriSpeech is a small, reliable download. In-the-Wild is ~7-8 GB and hosted on a
Fraunhofer share whose URL occasionally changes — so if the auto-download fails,
this script tells you exactly where to drop the zip by hand and continues with the
controlled track alone rather than dying. The founder chose "both" sources; this
degrades gracefully to "controlled only" instead of blocking the whole run.
"""
from __future__ import annotations

import csv
import io
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

# --- Dataset sources ---------------------------------------------------------
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"

# In-the-Wild (Müller et al., "Does Audio Deepfake Detection Generalize?").
# If this URL 404s, download release_in_the_wild.zip manually and drop it at
# ITW_ZIP; the script will pick it up. Mirror listing: https://deepfake-total.com/in_the_wild
ITW_URL = "https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa/download"
ITW_ZIP = config.DOWNLOADS / "release_in_the_wild.zip"
LIBRI_TAR = config.DOWNLOADS / "test-clean.tar.gz"


# --- Small download helper ---------------------------------------------------
def _download(url: str, dest: Path) -> bool:
    """Stream a URL to disk with a coarse progress print. Returns success."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already present: {dest.name} "
              f"({dest.stat().st_size / 1e6:.0f} MB)")
        return True
    print(f"  downloading {url}\n  -> {dest}")
    try:
        # Browser-like UA: some hosts (e.g. DataShare) serve an HTML stub to
        # unrecognized agents instead of the file.
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            chunk = 1 << 20  # 1 MB
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    pct = 100 * done / total
                    print(f"\r    {done/1e6:7.0f} / {total/1e6:.0f} MB "
                          f"({pct:4.1f}%)", end="", flush=True)
            print()
        return True
    except Exception as e:  # noqa: BLE001 — any failure => degrade gracefully
        print(f"\n  download failed: {e}")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def _write_wav(wav: np.ndarray, dest: Path) -> None:
    """Write mono float32 audio at the pipeline sample rate."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), wav.astype(np.float32), config.SAMPLE_RATE)


# --- LibriSpeech (controlled track, real) ------------------------------------
def prepare_librispeech() -> list[dict]:
    """Download test-clean, export N varied-speaker utterances as 16 kHz WAV."""
    print("\n[1/2] LibriSpeech test-clean (real, controlled track)")
    if not _download(LIBRISPEECH_URL, LIBRI_TAR):
        raise RuntimeError("LibriSpeech download failed — cannot build any track.")

    extract_root = config.DOWNLOADS / "LibriSpeech"
    if not (extract_root / "test-clean").exists():
        print("  extracting tar ...")
        with tarfile.open(LIBRI_TAR, "r:gz") as tar:
            tar.extractall(config.DOWNLOADS)

    # Group utterances by speaker so we can round-robin for variety.
    flacs = sorted((extract_root / "test-clean").rglob("*.flac"))
    by_speaker: dict[str, list[Path]] = {}
    for f in flacs:
        speaker = f.parts[-3]  # LibriSpeech/test-clean/<speaker>/<chapter>/x.flac
        by_speaker.setdefault(speaker, []).append(f)

    # Round-robin pick across speakers until we hit N_LIBRI_REAL.
    rng = np.random.default_rng(config.SEED)
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    picked: list[tuple[str, Path]] = []
    idx = 0
    while len(picked) < config.N_LIBRI_REAL:
        progressed = False
        for spk in speakers:
            pool = by_speaker[spk]
            if idx < len(pool):
                picked.append((spk, pool[idx]))
                progressed = True
                if len(picked) >= config.N_LIBRI_REAL:
                    break
        if not progressed:
            break  # ran out of utterances
        idx += 1

    rows = []
    for spk, flac in picked:
        wav, _ = librosa.load(str(flac), sr=config.SAMPLE_RATE, mono=True)
        out = config.REAL_DIR / f"libri_{spk}_{flac.stem}.wav"
        _write_wav(wav, out)
        rows.append({
            "path": out.relative_to(config.ROOT).as_posix(),
            "label": "real",
            "source": "libri",
            "track": config.TRACK_CONTROLLED,
            "speaker": spk,
        })
    print(f"  exported {len(rows)} real clips from "
          f"{len({r['speaker'] for r in rows})} speakers")
    return rows


# --- In-the-Wild (benchmark track, real + fake) ------------------------------
def prepare_in_the_wild() -> list[dict]:
    """Download In-the-Wild, sample a balanced real/fake slice as 16 kHz WAV."""
    print("\n[2/2] In-the-Wild (real + fake, benchmark track)")
    if not _download(ITW_URL, ITW_ZIP):
        print("  !! In-the-Wild unavailable. Continuing with the CONTROLLED track")
        print(f"     only. To add it later: download release_in_the_wild.zip to")
        print(f"     {ITW_ZIP}  then re-run this script.")
        return []

    # The zip contains a meta.csv (columns: file, speaker, label) + wav files.
    with zipfile.ZipFile(ITW_ZIP) as z:
        names = z.namelist()
        meta_name = next((n for n in names if n.endswith("meta.csv")), None)
        if not meta_name:
            print("  !! meta.csv not found in zip; skipping benchmark track.")
            return []
        with z.open(meta_name) as m:
            reader = csv.DictReader(io.TextIOWrapper(m, encoding="utf-8"))
            meta = list(reader)

        # Column names vary slightly across releases; normalize.
        def col(row, *cands):
            for c in cands:
                if c in row:
                    return row[c]
            return ""

        reals = [r for r in meta
                 if "bona" in col(r, "label", "class").lower()]
        fakes = [r for r in meta
                 if "spoof" in col(r, "label", "class").lower()
                 or "fake" in col(r, "label", "class").lower()]

        rng = np.random.default_rng(config.SEED)
        rng.shuffle(reals)
        rng.shuffle(fakes)
        reals = reals[: config.N_ITW_REAL]
        fakes = fakes[: config.N_ITW_FAKE]

        zip_dir = {Path(n).name: n for n in names if n.endswith(".wav")}

        rows = []
        for group, label, out_dir in (
            (reals, "real", config.REAL_DIR),
            (fakes, "fake", config.FAKE_ITW_DIR),
        ):
            for r in group:
                fname = Path(col(r, "file", "filename")).name
                zpath = zip_dir.get(fname)
                if not zpath:
                    continue
                with z.open(zpath) as wf:
                    wav, _ = librosa.load(wf, sr=config.SAMPLE_RATE, mono=True)
                out = out_dir / f"itw_{label}_{Path(fname).stem}.wav"
                _write_wav(wav, out)
                rows.append({
                    "path": out.relative_to(config.ROOT).as_posix(),
                    "label": label,
                    "source": "itw",
                    "track": config.TRACK_BENCHMARK,
                    "speaker": col(r, "speaker") or "unknown",
                })
    print(f"  exported {sum(1 for r in rows if r['label']=='real')} real / "
          f"{sum(1 for r in rows if r['label']=='fake')} fake from In-the-Wild")
    return rows


# --- Manifest ----------------------------------------------------------------
MANIFEST_COLS = ["path", "label", "source", "track", "speaker"]


def write_manifest(rows: list[dict]) -> None:
    with open(config.MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote manifest: {config.MANIFEST}  ({len(rows)} rows)")
    print("  NOTE: XTTS fakes are added later by generate_fakes.py (appends rows).")


def main() -> None:
    config.ensure_dirs()
    rows = prepare_librispeech()
    rows += prepare_in_the_wild()
    write_manifest(rows)
    n_real = sum(1 for r in rows if r["label"] == "real")
    n_fake = sum(1 for r in rows if r["label"] == "fake")
    print(f"\nDone. {n_real} real / {n_fake} fake so far "
          f"(controlled fakes pending generate_fakes.py).")


if __name__ == "__main__":
    main()
