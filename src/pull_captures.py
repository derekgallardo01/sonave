"""
pull_captures.py — download captured Meet audio from the Railway capture service.

    python src/pull_captures.py https://sonave-production-3ca2.up.railway.app
    # or set SONAVE_CAPTURE_URL and run with no arg

Fetches the /captures list and downloads any WAVs not already local into
data/captured/, ready for src/add_captured.py.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

OUT = config.DATA / "captured"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sonave/pull"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> None:
    import os
    base = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SONAVE_CAPTURE_URL", "")).rstrip("/")
    if not base:
        raise SystemExit("Usage: pull_captures.py <service_url>  (or set SONAVE_CAPTURE_URL)")
    OUT.mkdir(parents=True, exist_ok=True)

    listing = json.loads(_get(f"{base}/captures"))
    files = listing.get("files", [])
    print(f"{base}: {len(files)} capture(s) on the server")
    got = 0
    for f in files:
        name = f["name"]
        dest = OUT / name
        if dest.exists():
            continue
        try:
            dest.write_bytes(_get(f"{base}/download/{name}"))
            print(f"  downloaded {name} ({f.get('mb','?')} MB)")
            got += 1
        except Exception as e:  # noqa: BLE001
            print(f"  !! failed {name}: {repr(e)[:100]}")
    print(f"pulled {got} new file(s) -> {OUT}")


if __name__ == "__main__":
    main()
