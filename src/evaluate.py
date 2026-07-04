"""
evaluate.py — the Phase 0 payload: clean-vs-Opus accuracy + EER, plots, findings.

Runs in the detector env (.venv):

    python src/evaluate.py

Pipeline:
  1. Load manifest, score EVERY clip at EVERY condition (control + each bitrate)
     with detect.py. The same clip is scored at every condition, so any metric
     delta is the codec's doing and nothing else.
  2. Per (track, condition) compute:
       - EER (threshold-free), via sklearn roc_curve
       - accuracy at a fixed 0.5 threshold
       - accuracy at the EER threshold CALIBRATED ON CLEAN audio, then held fixed
         on compressed audio (the realistic deployment number: you calibrate once
         on good audio and eat the live degradation).
  3. Write results/metrics.csv, three plots, and a draft results/findings.md.

Score convention (from detect.py): P(fake) in [0,1], higher = more likely fake.
So y_true = 1 for fake, 0 for real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import detect  # noqa: E402


# --- Metrics -----------------------------------------------------------------
def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """
    Equal Error Rate and its threshold.

    EER is the operating point where the false-accept rate (fakes called real)
    equals the false-reject rate (reals called fake). Threshold-free summary of a
    detector's quality — lower is better, 0.5 == coin flip.
    """
    from sklearn.metrics import roc_curve

    # Degenerate case (only one class present) — undefined, report 0.5.
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return eer, float(thr[idx])


def accuracy_at(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """Fraction correct when predicting fake iff score >= threshold."""
    pred = (scores >= threshold).astype(int)
    return float((pred == y_true).mean())


# --- Scoring pass ------------------------------------------------------------
def score_everything(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Return a long-form frame: one row per (clip, condition) with its P(fake).

    Skips (with a warning) any compressed file that's missing so a partial run
    still produces something rather than crashing.
    """
    records = []
    for condition in config.CONDITIONS:
        paths, keep_idx = [], []
        for i, row in manifest.iterrows():
            p = config.compressed_path(row["path"], condition)
            if p.exists():
                paths.append(p)
                keep_idx.append(i)
            else:
                print(f"  !! missing {condition} file for {row['path']}")
        print(f"  scoring {len(paths)} clips @ {condition} ...")
        scores = detect.score_batch(paths)
        for i, s in zip(keep_idx, scores):
            r = manifest.loc[i]
            records.append({
                "path": r["path"], "label": r["label"], "track": r["track"],
                "source": r["source"], "condition": condition, "score": s,
                "y_true": 1 if r["label"] == "fake" else 0,
            })
    return pd.DataFrame.from_records(records)


# --- Metric table ------------------------------------------------------------
def build_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Per (track, condition): EER, acc@0.5, and acc@clean-calibrated-threshold.

    The clean-calibrated threshold is each track's EER threshold measured on the
    CONTROL condition, then applied unchanged to every bitrate.
    """
    tracks = list(scored["track"].unique()) + ["all"]
    rows = []
    for track in tracks:
        tsub = scored if track == "all" else scored[scored["track"] == track]

        # Calibrate the fixed threshold once, on clean audio for this track.
        clean = tsub[tsub["condition"] == config.CONTROL]
        _, clean_thr = compute_eer(clean["y_true"].to_numpy(),
                                   clean["score"].to_numpy())

        for condition in config.CONDITIONS:
            csub = tsub[tsub["condition"] == condition]
            if csub.empty:
                continue
            y = csub["y_true"].to_numpy()
            s = csub["score"].to_numpy()
            eer, _ = compute_eer(y, s)
            rows.append({
                "track": track,
                "condition": condition,
                "n": len(csub),
                "n_real": int((y == 0).sum()),
                "n_fake": int((y == 1).sum()),
                "eer": round(eer, 4),
                "acc_0.5": round(accuracy_at(y, s, 0.5), 4),
                "acc_clean_thr": round(accuracy_at(y, s, clean_thr), 4),
                "clean_thr": round(clean_thr, 4),
            })
    return pd.DataFrame(rows)


# --- Plots -------------------------------------------------------------------
def _bitrate_x(conditions: list[str]) -> list[str]:
    return conditions  # categorical x-axis: control, 16k, 24k, 32k


def make_plots(metrics: pd.DataFrame, scored: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.PLOTS.mkdir(parents=True, exist_ok=True)
    order = config.CONDITIONS

    # 1) Accuracy vs bitrate (clean-calibrated threshold — the honest number).
    plt.figure(figsize=(7, 4.5))
    for track in metrics["track"].unique():
        m = metrics[metrics["track"] == track].set_index("condition").reindex(order)
        plt.plot(order, m["acc_clean_thr"] * 100, marker="o", label=track)
    plt.axhline(70, ls="--", color="grey", lw=1, label="~70% commodity floor")
    plt.title("Accuracy vs Opus bitrate (clean-calibrated threshold)")
    plt.xlabel("condition"); plt.ylabel("accuracy (%)")
    plt.ylim(45, 102); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(config.PLOTS / "accuracy_vs_bitrate.png", dpi=130); plt.close()

    # 2) EER vs bitrate.
    plt.figure(figsize=(7, 4.5))
    for track in metrics["track"].unique():
        m = metrics[metrics["track"] == track].set_index("condition").reindex(order)
        plt.plot(order, m["eer"] * 100, marker="s", label=track)
    plt.title("Equal Error Rate vs Opus bitrate (lower is better)")
    plt.xlabel("condition"); plt.ylabel("EER (%)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(config.PLOTS / "eer_vs_bitrate.png", dpi=130); plt.close()

    # 3) Score-distribution shift: clean vs 24k (real vs fake histograms).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    for ax, cond in zip(axes, [config.CONTROL, "24k"]):
        sub = scored[scored["condition"] == cond]
        ax.hist(sub[sub["label"] == "real"]["score"], bins=25, alpha=0.6,
                label="real", color="#2a7fff")
        ax.hist(sub[sub["label"] == "fake"]["score"], bins=25, alpha=0.6,
                label="fake", color="#ff5555")
        ax.set_title(f"P(fake) distribution @ {cond}")
        ax.set_xlabel("P(fake)"); ax.legend(); ax.grid(alpha=0.3)
    axes[0].set_ylabel("count")
    fig.suptitle("Score separation: clean vs Opus 24k")
    fig.tight_layout()
    fig.savefig(config.PLOTS / "score_dist_clean_vs_24k.png", dpi=130)
    plt.close(fig)

    print(f"  wrote 3 plots -> {config.PLOTS}")


# --- Draft findings ----------------------------------------------------------
def write_findings(metrics: pd.DataFrame) -> None:
    """Auto-draft results/findings.md; a human tightens the prose after.

    EER is the PRIMARY metric: our detector's scores are heavily zero-inflated
    (most clean clips sit near 0), which makes any single-threshold accuracy
    fragile, while EER — a rank-based metric — stays robust. Accuracy@clean-thr is
    reported as secondary colour. The commodity ~71% accuracy the thesis cites
    corresponds to EER ~29%, so we frame the drop in EER terms too.
    """
    all_m = metrics[metrics["track"] == "all"].set_index("condition")
    clean_acc = all_m.loc[config.CONTROL, "acc_clean_thr"] * 100
    clean_eer = all_m.loc[config.CONTROL, "eer"] * 100

    # Worst Opus point is the one with the HIGHEST EER (worst detection).
    worst_cond = all_m.loc[config.OPUS_BITRATES, "eer"].idxmax()
    worst_acc = all_m.loc[worst_cond, "acc_clean_thr"] * 100
    worst_eer = all_m.loc[worst_cond, "eer"] * 100
    eer_rise = worst_eer - clean_eer          # points of EER added by compression
    drop = clean_acc - worst_acc              # secondary: accuracy points lost

    # EER-driven verdict (accuracy drop kept as a secondary trigger).
    if (eer_rise >= 8 and worst_eer >= 25) or drop >= 15:
        verdict = ("**GO — thesis confirmed.** Opus compression drives a large, "
                   "consistent rise in EER (detection collapsing toward the "
                   "commodity ~29% EER / ~71% accuracy floor). Proceed to Phase 1 "
                   "(codec-augmented fine-tuning to recover it).")
    elif eer_rise >= 4 or drop >= 7:
        verdict = ("**LEAN GO — moderate gap.** Real degradation but smaller than "
                   "hoped; the wedge exists — confirm it holds on a bigger/harder "
                   "set and stronger fakes before committing.")
    else:
        verdict = ("**NO-GO — wedge is thin.** Commodity detection already tolerates "
                   "Opus here (EER barely moves). Reconsider the premise before "
                   "building further.")

    def table(m: pd.DataFrame) -> str:
        lines = ["| track | condition | n | EER % | acc@clean-thr % | acc@0.5 % |",
                 "|---|---|---|---|---|---|"]
        for _, r in m.iterrows():
            lines.append(
                f"| {r['track']} | {r['condition']} | {r['n']} | "
                f"{r['eer']*100:.1f} | {r['acc_clean_thr']*100:.1f} | "
                f"{r['acc_0.5']*100:.1f} |")
        return "\n".join(lines)

    md = f"""# Sonave Phase 0 — Findings (auto-draft)

**Detector:** `{config.DETECTOR_HF_MODEL}`
**Conditions:** {", ".join(config.CONDITIONS)} (Opus, mono, VoIP)
**Primary metric: EER** (rank-based, robust). Scores are zero-inflated, so
single-threshold accuracy is secondary; where shown it uses the EER threshold
calibrated on clean audio and held fixed on compressed audio (the realistic
deployment number: calibrate once on good audio, eat the live degradation).

## Headline (combined, both tracks)

- Clean (control): **EER {clean_eer:.1f}%** ({clean_acc:.1f}% acc@clean-thr).
- Worst Opus point ({worst_cond}): **EER {worst_eer:.1f}%** ({worst_acc:.1f}% acc).
- **EER rise under compression: +{eer_rise:.1f} points** (secondary: {drop:.1f} pts accuracy lost).

## Verdict

{verdict}

## Full metrics

{table(metrics)}

## Plots

- `plots/accuracy_vs_bitrate.png` — accuracy vs Opus bitrate (per track).
- `plots/eer_vs_bitrate.png` — EER vs Opus bitrate.
- `plots/score_dist_clean_vs_24k.png` — how real/fake score separation collapses.

## How to read this

- **Big drop (~15-20+ pts, or accuracy sliding toward ~70%)** → thesis confirmed.
- **Small / no drop** → commodity detectors already handle compression; wedge is thin.
- The `controlled` track (LibriSpeech vs XTTS clones) and the `benchmark` track
  (In-the-Wild) are independent; a drop showing up in **both** is the credible signal.

---
*Auto-generated by evaluate.py. Numbers are real; tighten the prose, then STOP and
take this to a human for the Phase 0 go/no-go decision before touching Phase 1.*
"""
    config.FINDINGS.write_text(md, encoding="utf-8")
    print(f"  wrote {config.FINDINGS}")


# --- Main --------------------------------------------------------------------
def main() -> None:
    config.ensure_dirs()
    if not config.MANIFEST.exists():
        raise SystemExit(f"No manifest at {config.MANIFEST}. Run prepare_data.py, "
                         "generate_fakes.py, compress.py first.")
    manifest = pd.read_csv(config.MANIFEST)
    print(f"Loaded manifest: {len(manifest)} clips "
          f"({(manifest.label=='real').sum()} real / "
          f"{(manifest.label=='fake').sum()} fake)")

    print("\nScoring all clips at all conditions ...")
    scored = score_everything(manifest)
    scored.to_csv(config.RESULTS / "scores_long.csv", index=False)

    print("\nComputing metrics ...")
    metrics = build_metrics(scored)
    metrics.to_csv(config.METRICS_CSV, index=False)
    print(metrics.to_string(index=False))

    print("\nPlots + findings ...")
    make_plots(metrics, scored)
    write_findings(metrics)
    print("\nDone. See results/findings.md — then STOP for the human go/no-go.")


if __name__ == "__main__":
    main()
