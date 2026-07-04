"""
eval_detector.py — did our detector learn to catch modern fakes it never saw?

Runs in the detector env (.venv):

    python src/eval_detector.py

Evaluates on split=="test" ONLY (held-out speakers; their XTTS clones were never
in training). Reports, for BOTH the commodity baseline and our fine-tuned model:

  - modern-fake catch rate (recall on kind=modern)  <- the headline
  - old-fake catch rate    (recall on kind=old; did we forget old attacks?)
  - real accuracy          (specificity; are we crying wolf on genuine speech?)
  - overall EER / AUC

The honest comparison is BEFORE (commodity `Bisher/...`) vs AFTER (our `models/
sonave_v0`) on the exact same held-out clips.

Caveat surfaced in the writeup: the modern test fakes are still XTTS-v2 (a
different SPEAKER split, not a different TTS system). True cross-architecture
generalization (an unseen TTS) is the next validation, not this run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

DSET_CSV = config.DATA / "dataset.csv"
OURS_DIR = config.ROOT / "models" / "sonave_v0"
MAX_LEN = int(4.0 * config.SAMPLE_RATE)


def _score_model(model_id: str, paths: list[Path]) -> np.ndarray:
    """Return P(fake) in [0,1] per clip for a given model dir/id."""
    import torch
    import librosa
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ext = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id).to(device).eval()
    fake_idx = _fake_index(model.config.id2label)

    out = []
    for p in paths:
        wav, _ = librosa.load(str(p), sr=config.SAMPLE_RATE, mono=True)
        if len(wav) >= MAX_LEN:                       # center 4 s, matches training
            s = (len(wav) - MAX_LEN) // 2
            wav = wav[s:s + MAX_LEN]
        inp = ext(wav, sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**inp).logits, -1)[0]
        out.append(float(probs[fake_idx]))
    del model
    torch.cuda.empty_cache()
    return np.array(out)


def _fake_index(id2label: dict) -> int:
    for i, lab in id2label.items():
        if any(w in str(lab).lower() for w in ("fake", "spoof", "deepfake")):
            return int(i)
    return 0


def _eer(y, s):
    from sklearn.metrics import roc_curve
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2


def _report(name: str, df: pd.DataFrame, scores: np.ndarray, thr: float) -> dict:
    df = df.copy()
    df["score"] = scores
    df["pred_fake"] = df["score"] >= thr
    from sklearn.metrics import roc_auc_score

    def catch(kind):
        sub = df[df["kind"] == kind]
        return float(sub["pred_fake"].mean()) if len(sub) else float("nan")

    real = df[df["label"] == "real"]
    y = (df["label"] == "fake").astype(int).to_numpy()
    row = {
        "model": name,
        "modern_catch": catch("modern"),
        "old_catch": catch("old"),
        "real_acc": float((~real["score"].ge(thr)).mean()),
        "auc": float(roc_auc_score(y, df["score"])) if len(np.unique(y)) > 1 else float("nan"),
        "eer": float(_eer(y, df["score"].to_numpy())),
        "thr": thr,
    }
    return row


def main() -> None:
    df = pd.read_csv(DSET_CSV)
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f"test clips: {len(test)}  "
          f"(real={ (test.label=='real').sum()}, "
          f"modern={(test.kind=='modern').sum()}, old={(test.kind=='old').sum()})")
    if (test.kind == "modern").sum() == 0:
        raise SystemExit("No modern test clips yet — run generate_trainfakes.py.")

    paths = [config.ROOT / p for p in test["path"]]

    rows = []
    for name, mid in [("commodity (Bisher)", config.DETECTOR_HF_MODEL),
                      ("ours (sonave_v0)", str(OURS_DIR))]:
        if name.startswith("ours") and not OURS_DIR.exists():
            print("  (skip ours — not trained yet)")
            continue
        print(f"\nscoring with {name} ...")
        scores = _score_model(mid, paths)
        # Fixed 0.5 threshold: both models output calibrated P(fake).
        rows.append(_report(name, test, scores, thr=0.5))

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    print("\n=== HELD-OUT TEST (unseen speakers) ===")
    print("catch rates = fraction correctly flagged as fake (higher=better); "
          "real_acc = genuine speech kept (higher=better)")
    show = res.copy()
    for c in ["modern_catch", "old_catch", "real_acc", "auc", "eer"]:
        show[c] = (show[c] * 100).round(1)
    print(show.to_string(index=False))

    (config.RESULTS).mkdir(exist_ok=True)
    res.to_csv(config.RESULTS / "detector_eval.csv", index=False)
    print(f"\nwrote {config.RESULTS / 'detector_eval.csv'}")


if __name__ == "__main__":
    main()
