"""
analyze_meeting.py — offline "feed it a recording, get a report" analyzer (MVP demo).

Runs in the detector env (.venv):

    # single stream (whole file as one speaker)
    python service/analyze_meeting.py meeting.wav --hop 2

    # PER-SPEAKER (the real product view): pass who-spoke-when segments
    python service/analyze_meeting.py meeting.wav --segments turns.csv

`turns.csv` = the speaker turns the capture layer gives us (Recall.ai supplies these
live; offline you can generate them with pyannote/speechbrain). Format:
    start,end,speaker
    0.0,8.0,alice
    8.0,16.0,bob
    ...

Each speaker gets their own rolling confidence + verdict, so you get the live
per-speaker RAG (real/suspect/fake) that the product is built around. Windows are
scored once and attributed to whichever speaker owns that time.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detector  # noqa: E402
import model_sls  # noqa: E402

WIN = model_sls.MAX_LEN            # 4 s
SILENCE_RMS = 0.005


def _score_windows(wav, hop):
    """Score every non-silent window; return list of (t_center, p_fake)."""
    hop_n = int(hop * model_sls.SR)
    starts, chunks = [], []
    for s in range(0, max(1, len(wav) - WIN + 1), hop_n):
        w = wav[s:s + WIN]
        if np.sqrt(np.mean(w ** 2)) < SILENCE_RMS:
            continue
        starts.append(s)
        chunks.append(w)
    if not chunks:
        return []
    res = detector.batch_score_arrays(chunks)
    return [((s + WIN // 2) / model_sls.SR, r["p_fake"])
            for s, r in zip(starts, res)]


def _rolling_verdict(pairs, hop):
    """EWMA rolling score + flagged stretches for one speaker's windows."""
    if not pairs:
        return {"windows": 0, "verdict": "no_speech"}
    pairs = sorted(pairs)
    alpha, roll, rolled = 0.35, pairs[0][1], []
    for _, p in pairs:
        roll = alpha * p + (1 - alpha) * roll
        rolled.append(roll)
    peak = float(np.max(rolled))
    overall = ("fake" if peak >= detector.TAU_FAKE
               else "suspect" if peak >= detector.TAU_REAL else "real")
    flags, cur = [], None
    for (t, _), r in zip(pairs, rolled):
        if r >= detector.TAU_REAL:
            cur = [t, t, r] if cur is None else [cur[0], t, max(cur[2], r)]
        elif cur is not None:
            flags.append({"start_s": round(cur[0], 1), "end_s": round(cur[1] + hop, 1),
                          "peak": round(cur[2], 3)}); cur = None
    if cur is not None:
        flags.append({"start_s": round(cur[0], 1), "end_s": round(cur[1] + hop, 1),
                      "peak": round(cur[2], 3)})
    return {"windows": len(pairs), "verdict": overall, "peak_rolling": round(peak, 3),
            "fraction_suspect": round(float(np.mean([r >= detector.TAU_REAL for r in rolled])), 3),
            "flagged_stretches": flags,
            "timeline": [{"t": round(t, 1), "rolling": round(r, 3)}
                         for (t, _), r in zip(pairs, rolled)]}


def _load_segments(path):
    segs = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            segs.append((float(r["start"]), float(r["end"]), r["speaker"]))
    return segs


def analyze(path: Path, hop: float, segments_path: str | None) -> dict:
    import librosa
    wav, _ = librosa.load(str(path), sr=model_sls.SR, mono=True)
    dur = len(wav) / model_sls.SR
    scored = _score_windows(wav, hop)   # [(t_center, p_fake)]

    base = {"file": path.name, "duration_s": round(dur, 1),
            "model_version": detector.MODEL_VERSION, "hop_s": hop}

    if not segments_path:
        base["mode"] = "single_stream"
        base.update(_rolling_verdict(scored, hop))
        return base

    # attribute each scored window to the speaker owning its center time.
    segs = _load_segments(segments_path)
    per = {}
    for t, p in scored:
        spk = next((s for a, b, s in segs if a <= t < b), None)
        if spk is not None:
            per.setdefault(spk, []).append((t, p))

    base["mode"] = "per_speaker"
    base["speakers"] = {spk: _rolling_verdict(pairs, hop)
                        for spk, pairs in per.items()}
    worst = max((v.get("peak_rolling", 0) for v in base["speakers"].values()),
                default=0)
    base["overall_verdict"] = ("fake" if worst >= detector.TAU_FAKE
                               else "suspect" if worst >= detector.TAU_REAL else "real")
    return base


def _print(rep):
    print(f"\n=== Sonave meeting analysis: {rep['file']} ===")
    print(f"duration {rep.get('duration_s')}s | model {rep.get('model_version')} "
          f"| mode {rep.get('mode')}")
    if rep.get("mode") == "per_speaker":
        print(f"OVERALL: {rep.get('overall_verdict','?').upper()}")
        for spk, v in rep["speakers"].items():
            tag = v.get("verdict", "?").upper()
            print(f"  speaker {spk:>8}: {tag:8} peak P(fake)={v.get('peak_rolling')} "
                  f"({v.get('windows')} windows)")
            for f in v.get("flagged_stretches", []):
                print(f"        flagged {f['start_s']}s-{f['end_s']}s peak={f['peak']}")
    else:
        print(f"OVERALL: {rep.get('verdict','?').upper()}  "
              f"(peak rolling P(fake) {rep.get('peak_rolling')})")
        for f in rep.get("flagged_stretches", []):
            print(f"  flagged {f['start_s']}s-{f['end_s']}s peak P(fake)={f['peak']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--hop", type=float, default=2.0)
    ap.add_argument("--segments", default=None,
                    help="CSV of speaker turns (start,end,speaker) for per-speaker mode")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rep = analyze(Path(args.audio), args.hop, args.segments)
    _print(rep)
    out = Path(args.out) if args.out else Path(args.audio).with_suffix(".sonave.json")
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
