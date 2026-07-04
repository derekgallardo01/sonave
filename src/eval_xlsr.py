"""
eval_xlsr.py — score the XLS-R+SLS detector vs commodity on the honest test sets.

Runs in the detector env (.venv):

    python src/eval_xlsr.py --model models/sonave_xlsr

Reports, for commodity (Bisher) and ours (XLS-R+SLS):
  - in_the_wild : external real-world deepfakes (data/fake/itw vs itw_real)  [gold test]
  - heldout     : dataset.csv split=test — modern clones from UNSEEN speakers
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import model_sls  # noqa: E402
from eval_detector import _score_model, _eer  # commodity scorer + EER


def _score_ours(model_dir: Path, paths: list[Path]) -> np.ndarray:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_sls.SLSDetector.load(model_dir, device)
    s = model_sls.score_paths(model, paths, device)
    del model
    torch.cuda.empty_cache()
    return s


def _metrics(fake_scores, real_scores):
    catch = float((fake_scores >= 0.5).mean())
    real_acc = float((real_scores < 0.5).mean())
    y = np.r_[np.zeros(len(real_scores)), np.ones(len(fake_scores))]
    s = np.r_[real_scores, fake_scores]
    return {"catch_%": round(catch * 100, 1), "real_acc_%": round(real_acc * 100, 1),
            "eer_%": round(_eer(y, s) * 100, 1)}


def _eer_threshold(real_scores, fake_scores) -> float:
    """Score threshold at the equal-error operating point."""
    from sklearn.metrics import roc_curve
    y = np.r_[np.zeros(len(real_scores)), np.ones(len(fake_scores))]
    s = np.r_[real_scores, fake_scores]
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, s)
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return float(thr[i])


def _opus_copies(paths, tag):
    """Opus-24k round-trip each clip through the real Meet codec (cached)."""
    from compress import opus_roundtrip
    out_dir = config.DATA / "compressed" / "itw_test"
    out = []
    for p in paths:
        dest = out_dir / f"{tag}_{Path(p).stem}__opus24k.wav"
        if not dest.exists():
            try:
                opus_roundtrip(Path(p), dest, "24k")
            except Exception:
                continue
        out.append(dest)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(config.ROOT / "models" / "sonave_xlsr"))
    args = ap.parse_args()
    model_dir = Path(args.model)

    # Test sets (all unseen in training).
    itw_fake = [Path(p) for p in glob.glob(str(config.FAKE_ITW_DIR / "*.wav"))]
    itw_real = [Path(p) for p in glob.glob(str(config.REAL_DIR / "itw_real_*.wav"))]

    # Prefer the diverse corpus test split if present: its fakes come from 27
    # MLAAD generators HELD OUT of training (ElevenLabs, Cartesia, Gemini, ...) —
    # the real "unseen generator" test. Fall back to dataset.csv otherwise.
    corpus = config.DATA / "corpus.csv"
    src = corpus if corpus.exists() else (config.DATA / "dataset.csv")
    df = pd.read_csv(src)
    test = df[df["split"] == "test"]
    held_fake = [config.ROOT / p for p in test[test.label == "fake"]["path"]]
    held_real = [config.ROOT / p for p in test[test.label == "real"]["path"]]

    # "Real call" test: Opus-compress In-the-Wild through the actual Meet codec.
    itw_opus_fake = _opus_copies(itw_fake, "itw_fake")
    itw_opus_real = _opus_copies(itw_real, "itw_real")

    sets = {
        "in_the_wild": (itw_fake, itw_real),
        "in_the_wild_opus24k": (itw_opus_fake, itw_opus_real),
        "unseen_gens": (held_fake, held_real),
    }

    mlaad_unseen = None
    if "generator" in test.columns:
        mg = test[(test.label == "fake") & test.generator.str.startswith("mlaad", na=False)]
        if len(mg):
            mlaad_unseen = [config.ROOT / p for p in mg["path"]]

    scorers = {
        "commodity (Bisher)": lambda ps: _score_model(config.DETECTOR_HF_MODEL, ps),
        "ours (XLS-R+SLS)": lambda ps: _score_ours(model_dir, ps),
    }

    rows = []
    raw = {}   # (set, model) -> (fake_scores, real_scores), for threshold calibration
    for setname, (fk, rl) in sets.items():
        print(f"\n=== {setname}: {len(fk)} fake / {len(rl)} real ===")
        for mname, fn in scorers.items():
            fs, rs = fn(fk), fn(rl)
            raw[(setname, mname)] = (fs, rs)
            m = _metrics(fs, rs)
            m.update({"set": setname, "model": mname})
            rows.append(m)
            print(f"  {mname:20s} catch={m['catch_%']:5.1f}%  "
                  f"real_acc={m['real_acc_%']:5.1f}%  eer={m['eer_%']:4.1f}%")

    if mlaad_unseen:
        print(f"\n=== unseen MLAAD generators only: {len(mlaad_unseen)} fakes ===")
        for mname, fn in scorers.items():
            c = float((fn(mlaad_unseen) >= 0.5).mean()) * 100
            print(f"  {mname:20s} catch={c:5.1f}%")
            rows.append({"set": "mlaad_unseen_only", "model": mname,
                         "catch_%": round(c, 1), "real_acc_%": None, "eer_%": None})

    # --- Calibrated operating point (honest thresholding) ---------------------
    # Pick the threshold on the held-out unseen_gens set (our own domain), then
    # apply that SAME fixed threshold to real-world In-the-Wild — the realistic
    # deployment number, instead of an arbitrary 0.5.
    print("\n=== calibrated threshold (set on unseen_gens, applied to In-the-Wild) ===")
    for mname in scorers:
        fs, rs = raw[("unseen_gens", mname)]
        tau = _eer_threshold(rs, fs)
        for target in ("in_the_wild", "in_the_wild_opus24k"):
            tfs, trs = raw[(target, mname)]
            catch = float((tfs >= tau).mean()) * 100
            racc = float((trs < tau).mean()) * 100
            print(f"  {mname:20s} [{target}] @tau={tau:.3f}: "
                  f"catch={catch:5.1f}%  real_acc={racc:5.1f}%")
            rows.append({"set": f"{target}@calib", "model": mname,
                         "catch_%": round(catch, 1), "real_acc_%": round(racc, 1),
                         "eer_%": None})

    res = pd.DataFrame(rows)[["set", "model", "catch_%", "real_acc_%", "eer_%"]]
    res.to_csv(config.RESULTS / "xlsr_eval.csv", index=False)
    print("\n=== SUMMARY ===")
    print(res.to_string(index=False))
    print(f"\nwrote {config.RESULTS / 'xlsr_eval.csv'}")


if __name__ == "__main__":
    main()
