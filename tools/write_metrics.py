"""write_metrics.py — distill the latest benchmark into railway/model_metrics.json.

Run after src/eval_xlsr.py. The Railway service serves this via GET /api/model so
the console's Training panel shows real dates and numbers instead of hardcoded ones.
(The file lives under railway/ because Railway builds with Root Directory=railway.)
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_CSV = ROOT / "results" / "xlsr_eval.csv"
HEAD = ROOT / "models" / "sonave_xlsr_meet" / "head.pt"
OUT = ROOT / "railway" / "model_metrics.json"


def main() -> None:
    if not EVAL_CSV.exists():
        sys.exit(f"{EVAL_CSV} missing — run src/eval_xlsr.py first")
    rows = {(r["set"], r["model"]): r for r in csv.DictReader(EVAL_CSV.open(newline=""))}
    ours = "ours (XLS-R+SLS)"
    m = {
        "model": "sonave-xlsr-meet",
        "trained": dt.date.fromtimestamp(HEAD.stat().st_mtime).isoformat() if HEAD.exists() else None,
        "benchmarked": dt.date.today().isoformat(),
        # ITW figures are read at the deployed operating point (tau=0.72, the @calib
        # rows) so the console/API match production behavior and the public page —
        # not the raw-0.5-threshold rows, which production does not run.
        "unseen_tools_catch_pct": float(rows[("mlaad_unseen_only", ours)]["catch_%"]),
        "unseen_gens_eer_pct": float(rows[("unseen_gens", ours)]["eer_%"]),
        "opus_real_acc_pct": float(rows[("in_the_wild_opus24k@calib", ours)]["real_acc_%"]),
        "itw_catch_pct": float(rows[("in_the_wild@calib", ours)]["catch_%"]),
    }
    OUT.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: trained {m['trained']}, unseen-tools catch {m['unseen_tools_catch_pct']}%")


if __name__ == "__main__":
    main()
