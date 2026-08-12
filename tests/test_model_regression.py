"""Model-quality regression gate.

The fast suite asserts nothing about detection quality, so a retrain that halves
the catch rate would ship green. This gate compares the latest benchmark run
(results/xlsr_eval.csv, produced by `python src/eval_xlsr.py --model
models/sonave_xlsr_meet`) against the committed baseline
(results/benchmark_baseline.json). Marked `gpu` because producing the CSV needs
the local GPU — run it after every retrain, BEFORE deploying:

    python src/eval_xlsr.py --model models/sonave_xlsr_meet
    python -m pytest -m gpu tests/test_model_regression.py

If a drop is intentional (a deliberate tradeoff), update the baseline JSON in
the same commit and say why in the commit message.
"""
import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "results" / "benchmark_baseline.json"
EVAL_CSV = ROOT / "results" / "xlsr_eval.csv"
OURS = "ours (XLS-R+SLS)"


@pytest.mark.gpu
def test_deployed_model_has_not_regressed():
    if not EVAL_CSV.exists():
        pytest.skip("results/xlsr_eval.csv missing — run src/eval_xlsr.py first")
    base = json.loads(BASELINE.read_text())
    tol = float(base["tolerance_pts"])

    current: dict[str, dict[str, float]] = {}
    with EVAL_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] == OURS:
                current[row["set"]] = {k: float(v) for k, v in row.items()
                                       if k.endswith("%") and v not in ("", None)}

    failures = []
    for test_set, metrics in base["metrics"].items():
        assert test_set in current, f"benchmark set '{test_set}' missing from {EVAL_CSV}"
        for metric, expected in metrics.items():
            got = current[test_set].get(metric)
            assert got is not None, f"{test_set}/{metric} missing from {EVAL_CSV}"
            if got < expected - tol:
                failures.append(f"{test_set}/{metric}: {got} (baseline {expected}, tol -{tol})")
    assert not failures, "model regressed vs baseline:\n  " + "\n  ".join(failures)
