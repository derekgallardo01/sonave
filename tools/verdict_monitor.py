"""
verdict_monitor.py — live real/fake verdict on the Meet stream.

    # score on your local GPU:
    python tools/verdict_monitor.py https://sonave-production-3ca2.up.railway.app
    # or score on the hosted Modal service (no local GPU / torch needed):
    python tools/verdict_monitor.py <railway_url> --remote https://<you>--sonave-detector-fastapi-app.modal.run

Polls the Railway capture service for new audio chunks as they flush, scores each
(locally or on Modal via /score_clip), prints a live verdict + a rolling call-level
verdict, and pushes it to the capture page. No tunnel needed — it reads the chunks the
capture service already saves.

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
# NB: model_sls (torch/transformers) is imported lazily, only in local-scoring mode,
# so --remote can run on a box with no GPU / no torch.

MODEL = ROOT / "models" / "sonave_xlsr_rw"
POLL = 12
SKIP = ("HealthCheck", "FIXCHECK", "WSTEST")
TMP = Path(config.DATA / "_verdict_tmp.wav")


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sonave"}), timeout=60) as r:
        return r.read()


def _verdict(p: float) -> str:
    return "fake" if p >= 0.7 else "suspect" if p >= 0.4 else "real"


def _post_clip(remote: str, path: Path):
    """POST a wav to the hosted /score_clip endpoint; return its P(fake) (or None)."""
    boundary = "----sonaveclip"
    data = Path(path).read_bytes()
    body = (
        f"--{boundary}\r\n".encode()
        + b'Content-Disposition: form-data; name="file"; filename="clip.wav"\r\n'
        + b"Content-Type: audio/wav\r\n\r\n"
        + data + b"\r\n"
        + f"--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        f"{remote}/score_clip", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        res = json.loads(r.read())
    p = res.get("p_fake")
    return None if p is None else float(p)


def main():
    import glob
    args = sys.argv[1:]
    remote = ""
    if "--remote" in args:
        i = args.index("--remote")
        remote = (args[i + 1].rstrip("/") if i + 1 < len(args) else "")
        del args[i:i + 2]
    base = (args[0] if args else "").rstrip("/")
    if not base:
        raise SystemExit("Usage: verdict_monitor.py <railway_url> [--remote <modal_url>] [local_model_dir]")

    if remote:
        # thin client: score on the hosted Modal service, no local model / GPU / torch
        print(f"scoring REMOTELY via {remote}/score_clip. Polling {base} every {POLL}s. Ctrl+C to stop.\n")

        def score_file(path):
            return _post_clip(remote, path)
    else:
        import librosa
        import torch
        import model_sls  # noqa: E402 — lazy: only local mode needs torch/transformers
        mdir = Path(args[1]) if len(args) > 1 else (ROOT / "models" / "sonave_xlsr_meet")
        if not mdir.exists():
            mdir = MODEL
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = model_sls.SLSDetector.load(mdir, dev)
        print(f"scoring LOCALLY on {dev} with {mdir.name}. Polling {base} every {POLL}s. Ctrl+C to stop.\n")

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
