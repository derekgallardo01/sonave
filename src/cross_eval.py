"""
cross_eval.py — does sonave_v0 catch fakes from generators it NEVER trained on?

Runs in the detector env (.venv):

    python src/cross_eval.py

sonave_v0 was trained only on XTTS-v2 clones (+ ASVspoof). This tests it on
DIFFERENT fake sources, which is the real proof of generalization (and rules out
"it just memorized our generation pipeline"):

  - in_the_wild : real-world deepfakes from unknown tools, an entirely external
                  pipeline (data/fake/itw). The strongest unseen test.
  - piper       : (if generated) fakes from Piper — a different TTS architecture.

For each generator set and each model we report the catch rate (fraction of fakes
flagged) and, on a matched real set, the real-voice accuracy.
"""
from __future__ import annotations

import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from eval_detector import _score_model, _eer  # reuse scorers

OURS = config.ROOT / "models" / "sonave_v1"

# Real reference sets that the model also never trained on.
LIBRI_TEST_REAL = [Path(p) for p in glob.glob(str(config.DATA / "dataset" / "test" / "real" / "libri_*.wav"))]
ITW_REAL = [Path(p) for p in glob.glob(str(config.REAL_DIR / "itw_real_*.wav"))]


def _sets() -> dict:
    s = {}
    itw_fake = [Path(p) for p in glob.glob(str(config.FAKE_ITW_DIR / "*.wav"))]
    if itw_fake:
        s["in_the_wild"] = {"fake": itw_fake, "real": ITW_REAL, "external": True}
    piper = [Path(p) for p in glob.glob(str(config.DATA / "fake" / "piper" / "*.wav"))]
    if piper:
        s["piper"] = {"fake": piper, "real": LIBRI_TEST_REAL, "external": False}
    return s


def main() -> None:
    if not OURS.exists():
        raise SystemExit("Train first: python src/train_detector.py")
    sets = _sets()
    if not sets:
        raise SystemExit("No cross-generator fake sets found (itw / piper).")

    models = {"commodity (Bisher)": config.DETECTOR_HF_MODEL,
              "ours (sonave_v1)": str(OURS)}

    rows = []
    for gen, d in sets.items():
        print(f"\n=== generator: {gen} "
              f"({len(d['fake'])} fake / {len(d['real'])} real) ===")
        for mname, mid in models.items():
            fs = _score_model(mid, d["fake"])
            rs = _score_model(mid, d["real"]) if d["real"] else np.array([])
            catch = float((fs >= 0.5).mean())
            real_acc = float((rs < 0.5).mean()) if len(rs) else float("nan")
            if len(rs):
                y = np.r_[np.zeros(len(rs)), np.ones(len(fs))]
                sc = np.r_[rs, fs]
                eer = _eer(y, sc) * 100
            else:
                eer = float("nan")
            rows.append({"generator": gen, "model": mname,
                         "catch_%": round(catch * 100, 1),
                         "real_acc_%": round(real_acc * 100, 1),
                         "eer_%": round(eer, 1)})
            print(f"  {mname:20s} catch={catch*100:5.1f}%  "
                  f"real_acc={real_acc*100:5.1f}%  eer={eer:4.1f}%")

    res = pd.DataFrame(rows)
    res.to_csv(config.RESULTS / "cross_eval.csv", index=False)
    print("\n=== SUMMARY (catch% = modern fakes flagged; higher=better) ===")
    print(res.to_string(index=False))
    print(f"\nwrote {config.RESULTS / 'cross_eval.csv'}")


if __name__ == "__main__":
    main()
