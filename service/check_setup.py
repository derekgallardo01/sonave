"""
check_setup.py — validate the .env / Recall setup without leaking the key.

    python service/check_setup.py

Loads .env, confirms the Recall key is present (masked), and makes a single
read-only GET to the Recall API to confirm the key + region base URL are valid.
Prints only status — never the key.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env():
    env = {}
    f = ROOT / ".env"
    if not f.exists():
        return env
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def mask(k: str) -> str:
    return f"{k[:5]}...{k[-3:]} (len {len(k)})" if len(k) > 10 else "(too short?)"


def main():
    env = load_env()
    key = env.get("SONAVE_RECALL_API_KEY", "")
    base = env.get("SONAVE_RECALL_BASE", "https://us-west-2.recall.ai/api/v1")
    print(f".env loaded: {(ROOT / '.env').exists()}")
    if not key or key == "your_recall_api_key_here":
        print("  [FAIL] SONAVE_RECALL_API_KEY not set (still placeholder).")
        return 1
    print(f"  [OK] key present: {mask(key)}")
    print(f"  base URL: {base}")

    # read-only auth probe: list bots (harmless)
    req = urllib.request.Request(f"{base}/bot/",
                                 headers={"Authorization": f"Token {key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            body = json.loads(r.read())
            n = body.get("count", body if isinstance(body, list) else "?")
            print(f"  [OK] Recall auth OK (HTTP {r.status}). Existing bots: {n}")
            print("\nSetup valid — you can send a bot to a meeting:")
            print("  python service/recall_adapter.py <meeting_url>")
            return 0
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            print(f"  [FAIL] auth rejected (HTTP {e.code}). Key wrong, or not authorized.")
        elif e.code == 404:
            print(f"  [FAIL] 404 — likely wrong region base URL. Check SONAVE_RECALL_BASE "
                  "against your Recall dashboard (us-west-2 / eu-central-1 / ...).")
        else:
            print(f"  [FAIL] HTTP {e.code}: {e.reason}")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] connection failed: {repr(e)[:120]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
