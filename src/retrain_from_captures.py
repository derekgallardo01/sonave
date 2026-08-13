"""
retrain_from_captures.py — the collect -> retrain -> validate loop, one command.

    # pull latest captures from Railway, retrain, validate
    python src/retrain_from_captures.py --pull https://sonave-production-3ca2.up.railway.app

    # use already-downloaded captures in data/captured/
    python src/retrain_from_captures.py

Steps:
  1. (optional) pull new captures from the Railway service
  2. add_captured.py     -> windows + speaker-disjoint split -> data/corpus_meet.csv
  3. train_xlsr.py       -> models/sonave_xlsr_meet  (diverse corpus + your Meet voices)
  4. validate            -> held-out captured audio: OLD vs NEW P(fake). The fix works
                            when held-out real Meet voice drops toward 0 with the new model.

This is the exact loop that already took a real voice from 0.42 -> 0.003.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

PY = str((ROOT / ".venv" / "Scripts" / "python.exe"))
OLD_MODEL = ROOT / "models" / "sonave_xlsr_rw"
NEW_MODEL = ROOT / "models" / "sonave_xlsr_meet"


def _run(args: list[str]):
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run([PY, *args], check=True, cwd=str(ROOT))


def validate() -> bool:
    """Score the held-out captured_test windows, REAL and FAKE sides separately.

    Labels live in the filenames (add_captured.py writes {label}_{spk}_te_*.wav;
    legacy meet_* windows predate labeling and are real). A single blended mean
    is meaningless — a model that misses every fake would look "good" on it.
    Returns True when the NEW model passes BOTH gates.
    """
    import glob
    import numpy as np
    import librosa
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    import model_sls

    def score(md, wavs):
        m = model_sls.SLSDetector.load(Path(md), "cuda" if torch.cuda.is_available() else "cpu")
        ps = []
        for i in range(0, len(wavs), 8):
            b = [model_sls.fit_length(w, False) for w in wavs[i:i + 8]]
            with torch.no_grad():
                ps += torch.softmax(m(**model_sls.make_inputs(b, next(m.parameters()).device.type)), -1)[:, 1].cpu().numpy().tolist()
        del m
        torch.cuda.empty_cache()
        return np.array(ps)

    test = sorted(glob.glob(str(config.DATA / "corpus" / "captured_test" / "*.wav")))
    if not test:
        print("no held-out captured_test windows to validate on."); return True
    real_paths = [p for p in test if Path(p).name.startswith(("real_", "meet_"))]
    fake_paths = [p for p in test if Path(p).name.startswith("fake_")]
    skipped = len(test) - len(real_paths) - len(fake_paths)
    if skipped:
        print(f"note: skipping {skipped} test windows with unrecognized label prefix")
    real_wavs = [librosa.load(p, sr=16000, mono=True)[0] for p in real_paths]
    fake_wavs = [librosa.load(p, sr=16000, mono=True)[0] for p in fake_paths]

    print(f"\n=== VALIDATION on held-out Meet windows: {len(real_wavs)} real / {len(fake_wavs)} fake ===")
    new_good = True
    for name, md in [("OLD (no new data)", OLD_MODEL), ("NEW (retrained)", NEW_MODEL)]:
        if not Path(md).exists():
            continue
        parts, real_ok, fake_ok = [], True, True
        if real_wavs:
            s = score(md, real_wavs)
            real_ok = s.mean() < 0.3
            parts.append(f"real: mean={s.mean():.3f} acc@0.5={(s < 0.5).mean()*100:.0f}%")
        if fake_wavs:
            s = score(md, fake_wavs)
            fake_ok = (s >= 0.5).mean() > 0.8
            parts.append(f"fake: mean={s.mean():.3f} catch@0.5={(s >= 0.5).mean()*100:.0f}%")
        good = real_ok and fake_ok
        verdict = "GOOD - both sides pass" if good else \
            ("flags real voices" if not real_ok else "misses fakes")
        print(f"  {name:20}: {'  '.join(parts)}  ({verdict})")
        if name.startswith("NEW"):
            new_good = good
    return new_good


def main():
    args = sys.argv[1:]
    if args and args[0] == "--validate-only":
        sys.exit(0 if validate() else 1)
    if args and args[0] == "--pull":
        _run(["src/pull_captures.py", args[1]] if len(args) > 1 else ["src/pull_captures.py"])
    _run(["src/add_captured.py"])
    _run(["src/train_xlsr.py", "--manifest", "data/corpus_meet.csv",
          "--out", "models/sonave_xlsr_meet", "--augment", "--epochs", "7"])
    good = validate()
    print("\nNext, before deploying the new checkpoint:")
    print("  python src/eval_xlsr.py --model models/sonave_xlsr_meet")
    print("  python -m pytest -m gpu tests/test_model_regression.py")
    if not good:
        print("VALIDATION FAILED - do not ship this checkpoint.")
        sys.exit(1)


if __name__ == "__main__":
    main()
