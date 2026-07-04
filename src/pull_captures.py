"""
pull_captures.py — download captured Meet audio from the Railway capture service.

    # REAL session (default) -> data/captured/
    python src/pull_captures.py https://sonave-production-3ca2.up.railway.app
    # FAKE session -> data/captured_fake/
    python src/pull_captures.py https://sonave-production-3ca2.up.railway.app --fake
    # only pull a substring (e.g. today's session) with --match
    python src/pull_captures.py <url> --fake --match 178318
    # or set SONAVE_CAPTURE_URL and run with no arg

Fetches the /captures list and downloads any WAVs not already local into
data/captured/ (real) or data/captured_fake/ (--fake), ready for src/add_captured.py.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

SKIP = ("HealthCheck", "FIXCHECK", "WSTEST")


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sonave/pull"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> None:
    import os
    args = [a for a in sys.argv[1:]]
    fake = "--fake" in args
    match = ""
    if "--match" in args:
        i = args.index("--match")
        match = args[i + 1] if i + 1 < len(args) else ""
        del args[i:i + 2]
    args = [a for a in args if a != "--fake"]
    base = (args[0] if args else os.environ.get("SONAVE_CAPTURE_URL", "")).rstrip("/")
    if not base:
        raise SystemExit("Usage: pull_captures.py <service_url> [--fake] [--match SUBSTR]")
    out = config.DATA / ("captured_fake" if fake else "captured")
    other = config.DATA / ("captured" if fake else "captured_fake")
    out.mkdir(parents=True, exist_ok=True)
    other_names = {p.name for p in other.glob("*.wav")} if other.exists() else set()
    print(f"-> {out.name}/  ({'FAKE' if fake else 'REAL'} session)"
          + (f"  [{len(other_names)} already labelled in {other.name}/ will be skipped]" if other_names else ""))

    listing = json.loads(_get(f"{base}/captures"))
    files = listing.get("files", [])
    print(f"{base}: {len(files)} capture(s) on the server")
    got = 0
    for f in files:
        name = f["name"]
        if any(s in name for s in SKIP):
            continue
        if match and match not in name:
            continue
        if name in other_names:      # already labelled the other way — never double-label
            continue
        dest = out / name
        if dest.exists():
            continue
        try:
            dest.write_bytes(_get(f"{base}/download/{name}"))
            print(f"  downloaded {name} ({f.get('mb','?')} MB)")
            got += 1
        except Exception as e:  # noqa: BLE001
            print(f"  !! failed {name}: {repr(e)[:100]}")
    print(f"pulled {got} new file(s) -> {out}")


if __name__ == "__main__":
    main()
