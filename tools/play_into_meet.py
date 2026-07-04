"""
play_into_meet.py — auto-play a folder of audio into a live Meeting for data capture.

Plays every audio file in a folder (sequentially, optionally shuffled/looped) through
your default output device. With the Sonave bot in the meeting and either (a) speakers
-> laptop mic, or (b) a virtual audio cable routed to the Meet mic, this feeds the
audio through the REAL Meet pipeline so the capture service records it — hands-off.

Use it two ways (label by SESSION, not by file):
  # REAL session: point at a folder of podcasts / varied real voices
  python tools/play_into_meet.py C:\\audio\\podcasts --shuffle --loop

  # FAKE session: point at your fake clips (put those captures in data/captured_fake/)
  python tools/play_into_meet.py data/corpus/mlaad/test --shuffle

Requires ffplay (ships with ffmpeg). Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
import time
from pathlib import Path

EXTS = ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg", "*.opus")


def _files(folder: str) -> list[str]:
    fs = []
    for pat in EXTS:
        fs += glob.glob(str(Path(folder) / "**" / pat), recursive=True)
    return sorted(fs)


def _play(f: str) -> bool:
    try:
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", f],
                       check=True)
        return True
    except KeyboardInterrupt:
        raise
    except Exception:
        return False


def _find_device(spec: str):
    """Resolve --device (index or name substring) to an output device index + samplerate."""
    import sounddevice as sd
    devs = sd.query_devices()
    if spec.isdigit():
        idx = int(spec)
    else:
        matches = [i for i, d in enumerate(devs)
                   if d["max_output_channels"] > 0 and spec.lower() in d["name"].lower()]
        if not matches:
            outs = [f'[{i}] {d["name"]}' for i, d in enumerate(devs) if d["max_output_channels"] > 0]
            raise SystemExit(f"no output device matching '{spec}'. Available:\n  " + "\n  ".join(outs))
        idx = matches[0]
    d = devs[idx]
    sr = int(d["default_samplerate"])
    print(f"playing into output device [{idx}] {d['name']} @ {sr} Hz")
    return idx, sr


def _play_device(f: str, device: int, sr: int) -> bool:
    """Decode a clip and stream it straight to a chosen output device (e.g. CABLE Input)."""
    import librosa
    import sounddevice as sd
    try:
        w = librosa.load(f, sr=sr, mono=True)[0]
        sd.play(w, sr, device=device)
        sd.wait()
        return True
    except KeyboardInterrupt:
        import sounddevice as sd
        sd.stop()
        raise
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder of audio to play into the meeting")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--loop", action="store_true", help="repeat forever")
    ap.add_argument("--gap", type=float, default=0.6, help="silence between clips (s)")
    ap.add_argument("--limit", type=int, default=0, help="max files per pass (0=all)")
    ap.add_argument("--device", default="", help="output device index or name substring "
                    "(e.g. 'CABLE Input') — routes straight to a virtual cable, no speakers")
    args = ap.parse_args()

    device_idx = device_sr = None
    if args.device:
        device_idx, device_sr = _find_device(args.device)

    files = _files(args.folder)
    if not files:
        raise SystemExit(f"no audio files under {args.folder}")
    print(f"{len(files)} files. Make sure the Sonave bot is in your meeting and your "
          f"output is routed into it (speakers->mic or a virtual cable). Ctrl+C to stop.\n")

    import random
    rng = random.Random(0)
    played = 0
    try:
        while True:
            order = files[:]
            if args.shuffle:
                rng.shuffle(order)
            if args.limit:
                order = order[: args.limit]
            for f in order:
                played += 1
                print(f"  [{played}] {Path(f).name}", flush=True)
                if device_idx is not None:
                    _play_device(f, device_idx, device_sr)
                else:
                    _play(f)
                if args.gap:
                    time.sleep(args.gap)
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\nstopped.")
    print(f"played {played} clips.")


if __name__ == "__main__":
    main()
