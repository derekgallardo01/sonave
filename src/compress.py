"""
compress.py — push every clip through Google-Meet-style Opus and back to WAV.

Runs in the detector env (.venv):

    python src/compress.py            # compress all manifest clips at every bitrate
    python src/compress.py --check    # round-trip ONE clip at 24k and verify it loads

Meet / Zoom / WebRTC voice is Opus, mono, ~16-40 kbps. For each clip we encode to
.opus at each target bitrate, then decode straight back to 16 kHz mono WAV — the
exact "your voice survived a real call" degradation. The uncompressed CONTROL
condition is handled by evaluate.py (it scores the originals), so we don't copy
those here. Output layout is owned by config.compressed_path().
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def _run(cmd: list[str]) -> None:
    """Run ffmpeg quietly; raise with stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(cmd)}\n{proc.stderr[-800:]}")


def opus_roundtrip(src_wav: Path, out_wav: Path, bitrate: str) -> None:
    """Encode src_wav to Opus at `bitrate`, decode back to 16 kHz mono WAV."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp_opus = out_wav.with_suffix(".opus")
    # Encode: libopus, mono, target bitrate (VoIP-style voice).
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src_wav),
        "-c:a", "libopus", "-b:a", bitrate, "-ac", str(config.CHANNELS),
        "-application", "voip",
        str(tmp_opus),
    ])
    # Decode back to WAV at the pipeline sample rate.
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(tmp_opus),
        "-ar", str(config.SAMPLE_RATE), "-ac", str(config.CHANNELS),
        str(out_wav),
    ])
    tmp_opus.unlink(missing_ok=True)


def _load_manifest() -> list[dict]:
    if not config.MANIFEST.exists():
        raise SystemExit(f"No manifest at {config.MANIFEST}. Run prepare_data.py "
                         "(and generate_fakes.py) first.")
    with open(config.MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check() -> int:
    """Smoke-test the codec path on a single clip."""
    import librosa

    rows = _load_manifest()
    src = config.ROOT / rows[0]["path"]
    out = config.COMPRESSED_DIR / "24k" / f"_check_{src.name}"
    print(f"round-tripping {src.name} at 24k ...")
    opus_roundtrip(src, out, "24k")
    dur = librosa.get_duration(path=str(out))
    print(f"OK — decoded {out.name}, {dur:.2f}s, loads cleanly.")
    out.unlink(missing_ok=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="round-trip one clip at 24k and exit")
    args = ap.parse_args()
    if args.check:
        raise SystemExit(check())

    config.ensure_dirs()
    rows = _load_manifest()
    total = len(rows) * len(config.OPUS_BITRATES)
    print(f"Compressing {len(rows)} clips x {len(config.OPUS_BITRATES)} bitrates "
          f"= {total} round-trips ...")

    done = 0
    for row in rows:
        src = config.ROOT / row["path"]
        if not src.exists():
            print(f"  !! missing source, skipping: {src}")
            continue
        for bitrate in config.OPUS_BITRATES:
            out = config.compressed_path(row["path"], bitrate)
            # Skip clips already round-tripped so re-runs (e.g. after adding a new
            # data track) only process the new files.
            if out.exists() and out.stat().st_size > 0:
                done += 1
                continue
            opus_roundtrip(src, out, bitrate)
            done += 1
            if done % 50 == 0 or done == total:
                print(f"\r  {done}/{total}", end="", flush=True)
    print(f"\nDone. Compressed audio under {config.COMPRESSED_DIR}")
    print("Next: python src/evaluate.py")


if __name__ == "__main__":
    main()
