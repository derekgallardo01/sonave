# Sonave — Adversarial robustness findings

`python src/robustness.py` applies common manipulations to the fake test set (does it
help fakes EVADE?) and to the real set (do the tricks cause FALSE ALARMS?). Model:
`sonave_xlsr_rw`, 80 fakes (unseen MLAAD generators) / 80 reals.

## Results

| Transform | Fake catch % | Δ vs clean | Real kept % |
|---|---|---|---|
| clean | 95.0 | — | 91.2 |
| gain +6 dB / −10 dB | 95.0 | 0 | 91.2 |
| **noise SNR 20** | 77.5 | **−17.5** | 77.5 |
| **noise SNR 10** | 65.0 | **−30.0** | 77.5 |
| mp3 64k | 96.2 | +1.2 | 83.8 |
| opus 16k | 96.2 | +1.2 | 82.5 |
| **pitch +1 semitone** | 100.0 | +5 | **35.0** |
| **time 1.06×** | 100.0 | +5 | **40.0** |
| lowpass 3.4 kHz | 98.8 | +4 | 83.8 |
| **reverb** | 96.2 | +1 | **43.8** |

## Two findings

1. **Evasion via noise.** Adding background noise drops catch 95% → 65%. An attacker
   can dirty-up a deepfake to slip past. (Robust to gain, mp3/opus, pitch, time.)
2. **False-alarm fragility (bigger).** The detector flags REAL voices as fake under
   almost any processing — **pitch (real kept 35%), time-stretch (40%), reverb (44%)**,
   noise (77%). It's using "this audio was processed" as a fake cue. **This is the root
   of the live Meet false-positive** — not Meet-specific; the model is broadly over-
   sensitive to real-audio manipulation.

## The fix (one recipe, both problems)

Augment training with these exact transforms — **pitch shift, time-stretch, reverb,
additive noise** (plus the existing RawBoost/codec) — applied to **both** real and fake:
- real voices stop false-flagging under processing (fixes Meet + these transforms),
- noise stops being an evasion (fakes-with-noise are now in training).

## HARDENING RESULT — augmented against the flagged transforms ✅

Extended `src/augment.py` with pitch-shift / time-stretch / reverb, retrained
`models/sonave_xlsr_hard`, re-ran the test. Before (`_rw`) vs after (`_hard`):

| Metric | Before | After |
|---|---|---|
| clean catch / real-kept | 95.0 / 91.2 | **97.5 / 92.5** |
| noise SNR10 catch (evasion) | 65.0 | **82.5** |
| noise SNR20 catch | 77.5 | 83.8 |
| **reverb** real-kept | 43.8 | **82.5** |
| **time 1.06×** real-kept | 40.0 | **63.7** |
| **pitch +1** real-kept | 35.0 | **58.8** |

**The loop worked:** noise-evasion and reverb false-alarms are largely fixed, and
clean performance improved rather than being traded away. Remaining headroom on pitch/
time-stretch (real-kept ~60%) — push further with stronger pitch/time augmentation
weight + more epochs. Model: `models/sonave_xlsr_hard`.
