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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder of audio to play into the meeting")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--loop", action="store_true", help="repeat forever")
    ap.add_argument("--gap", type=float, default=0.6, help="silence between clips (s)")
    ap.add_argument("--limit", type=int, default=0, help="max files per pass (0=all)")
    args = ap.parse_args()

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
