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
