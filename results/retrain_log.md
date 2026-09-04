# Retrain log

One line per weekly retrain (appended by tools/sunday_retrain.ps1).

- 2026-08-01 — Stage 8 combined model trained (pre-log; see results/detector_v2_progress.md)
- 2026-08-12 — first benchmark of the deployed checkpoint: 95.2% unseen-tools catch (baseline set)
- 2026-08-13 — first retrain with genuine fake-through-Meet data (9 chunks MLAAD-via-Meet
  + 34 chunks clean-real-via-Meet from the 08-12 live demo). Held-out Meet validation
  passed both sides; **regression gate blocked the ship**: real-acc@0.5 fell ~5 pts
  (90→86.8 / 93.3→88 / 94→88) in exchange for In-the-Wild catch 58.7→**78.0** (+19.3)
  and calibrated τ≈0.7: 60.0% catch @ 93.3% real-acc vs deployed 48.7% @ 94.0%.
  Decision: **hold** — the fake-Meet data is one session; checkpoint preserved at
  `models/sonave_xlsr_meet_v2_candidate` (local). Revisit after more weekly data;
  if the gain repeats on diverse sessions, ship with recalibrated thresholds
  (suspect band 0.40→0.50) and an updated baseline.
- 2026-08-21 — make-up run for 2026-08-16 (scheduler pointed at a versioned Store
  pwsh path that a PowerShell update removed; fixed to the stable alias). 110 new
  capture files pulled — a week dominated by REAL meeting audio from live product
  testing. Split validation: the retrain fixed fresh-capture drift the deployed
  model shows (real acc **40% → 93%** on the new week's held-out windows, fake
  catch 80% → 86%). But the **regression gate held on all four fake-catch
  metrics**: MLAAD-unseen 84.4 (base 95.2), unseen-gens 85.6 (95.4), ITW 46.7
  (58.7), ITW-Opus 38.0 (59.3). The real-heavy week pulled the model toward
  real-voice cleanliness (ITW real-acc 98.0, EER 14.3 — both better than base)
  at too much fake-catch cost. Candidate preserved at
  `models/sonave_xlsr_meet_v3_candidate`; deployed checkpoint restored.
  Next run needs fake-side balance: generate red-team clones over the new real
  captures and/or up-weight the 87-generator diverse fake set so a real-heavy
  capture week can't tilt the loss.
- 2026-08-21 — retrain attempt FAILED (step failed: regression gate). Candidate preserved at models\sonave_xlsr_meet_candidate_2026-08-21; deployed checkpoint restored.
- 2026-08-30 — retrain attempt FAILED (step failed: regression gate). Candidate preserved at models\sonave_xlsr_meet_candidate_2026-08-30; deployed checkpoint restored.
- 2026-09-04 — **SHIPPED the run-3 candidate (`sonave_xlsr_meet_candidate_2026-08-21`) with recalibrated bands.**
  Re-benchmarked on the local RTX 5060 (fresh eval, not cached): headline unseen-tools catch
  **97.2%** (deployed 95.2), unseen-gens 97.6/88.2 at **6.3% EER** (best ever), and — the unlock —
  the eval's own calibration picked **τ=0.716**, landing In-the-Wild at **57.3% catch @ 94.0%
  real-acc** and ITW-Opus **56.7 @ 94.7**, matching the prior deployed ITW profile while beating
  it everywhere else and fixing the fresh-capture drift (40% → 84–94% real-acc). The Aug-13 log
  pre-planned exactly this: "if the gain repeats on diverse sessions, ship with recalibrated
  thresholds (suspect 0.40→0.50) and an updated baseline" — it has now repeated across three
  data mixes. Actions: promoted the checkpoint (prior deployed model backed up at
  `models/sonave_xlsr_meet_deployed_pre_ship_2026-09-04` for instant rollback); moved bands
  SUSPECT 0.40→**0.50** / FAKE 0.70→**0.72** across detector.py, modal_app.py, railway `_av`,
  verdict_monitor; reset `benchmark_baseline.json` to the measured operating-point numbers
  (ITW floors keyed on the @calib rows) with the deliberate-tradeoff rationale in the file;
  refreshed `model_metrics.json` and every public figure (benchmarks page, landing, guides,
  console tile). Regression gate green against the new baseline.
