"""
verdict_monitor.py — live real/fake verdict on the Meet stream, on your GPU.

    python tools/verdict_monitor.py https://sonave-production-3ca2.up.railway.app

Polls the Railway capture service for new audio chunks as they flush, scores each on
your local GPU, and prints a live-updating verdict + a rolling call-level verdict.
No tunnel needed — it reads the chunks the capture service already saves.

Latency = the capture flush interval (~2 min by default). For faster feedback, lower
CHUNK_SEC in railway/app.py (needs a redeploy).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402
import model_sls  # noqa: E402

MODEL = ROOT / "models" / "sonave_xlsr_rw"
POLL = 12
SKIP = ("HealthCheck", "FIXCHECK", "WSTEST")
TMP = Path(config.DATA / "_verdict_tmp.wav")


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sonave"}), timeout=60) as r:
        return r.read()


def _verdict(p: float) -> str:
    return "fake" if p >= 0.7 else "suspect" if p >= 0.4 else "real"


def main():
    import glob
    import librosa
    import torch
    base = (sys.argv[1] if len(sys.argv) > 1 else "").rstrip("/")
    if not base:
        raise SystemExit("Usage: verdict_monitor.py <railway_url>")

    # optional 2nd arg = model dir; default the Meet-adapted model if present
    if len(sys.argv) > 2:
        mdir = Path(sys.argv[2])
    else:
        mdir = ROOT / "models" / "sonave_xlsr_meet"
        if not mdir.exists():
            mdir = MODEL
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = model_sls.SLSDetector.load(mdir, dev)
    print(f"scoring on {dev} with {mdir.name}. Polling {base} every {POLL}s. Ctrl+C to stop.\n")

    def score_file(path):
        w = librosa.load(str(path), sr=16000, mono=True)[0]
        wins = [w[s:s + 64000] for s in range(0, len(w) - 64000, 32000)
                if np.sqrt(np.mean(w[s:s + 64000] ** 2)) >= 0.005]
        if not wins:
            return None
        ps = []
        for i in range(0, len(wins), 8):
            b = [model_sls.fit_length(x, False) for x in wins[i:i + 8]]
            with torch.no_grad():
                ps += torch.softmax(m(**model_sls.make_inputs(b, dev)), -1)[:, 1].cpu().numpy().tolist()
        return float(np.mean(ps))

    seen, roll = set(), None
    try:
        while True:
            try:
                files = json.loads(_get(f"{base}/captures")).get("files", [])
            except Exception as e:
                print(f"  (poll failed: {repr(e)[:60]})"); time.sleep(POLL); continue
            new = [f["name"] for f in files
                   if f["name"] not in seen and not any(s in f["name"] for s in SKIP)]
            for name in new:
                seen.add(name)
                try:
                    TMP.write_bytes(_get(f"{base}/download/{name}"))
                    p = score_file(TMP)
                except Exception:
                    continue
                if p is None:
                    continue
                roll = p if roll is None else 0.4 * p + 0.6 * roll
                v, rv = _verdict(p), _verdict(roll)
                tag = {"real": "  REAL ", "suspect": "SUSPECT", "fake": " FAKE!"}.get(v, v)
                print(f"  {time.strftime('%H:%M:%S')} | {name[-18:]} | this chunk: "
                      f"[{tag}] P(fake)={p:.2f} | call so far: {rv.upper()} ({roll:.2f})", flush=True)
                # push the verdict up to Railway so it shows on the page
                parts = name[:-4].split("_")
                speaker = "_".join(parts[1:-2]) if len(parts) >= 4 else name
                try:
                    body = json.dumps({"speaker": speaker, "p_fake": round(p, 3),
                                       "rolling": round(roll, 3), "verdict": rv}).encode()
                    urllib.request.urlopen(urllib.request.Request(
                        f"{base}/api/verdict", data=body,
                        headers={"Content-Type": "application/json"}), timeout=10)
                except Exception:
                    pass
            time.sleep(POLL)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
