# Sonave Phase 0 — Findings & Go/No-Go

**Question:** How much detection accuracy is lost when audio is degraded to Google
Meet's Opus codec conditions?

**One-line answer:** For the variable we isolated — **Opus bitrate (16–32 kbps)** —
a competent open-source detector loses **almost nothing**. On its home turf it holds
~**97–98% accuracy through every bitrate** (clean EER 1.7% → compressed EER ~0.3%).
The "compression craters detection toward ~70%" thesis, **as narrowly tested, is not
confirmed.** A more interesting vulnerability showed up instead — see §4.

**Detector:** `Bisher/wav2vec2_ASV_deepfake_audio_detection` (wav2vec2 SSL front-end,
ASVspoof-trained). Chosen after the obvious first pick (`MelodyMachine/…`) proved
overfit and non-discriminating — see §5.
**Compression:** Opus, mono, `-application voip`, at 16k / 24k / 32k, decoded back to
16 kHz WAV. Uncompressed control included. Same clips scored at every condition, so any
delta is purely the codec's.
**Primary metric: EER** (rank-based; robust to the detector's zero-inflated scores).

---

## 1. The valid test — ASVspoof 2019 LA (in-distribution)

This is the track that can actually answer the question, because here the detector has a
**strong clean baseline** (near-perfect separation: bonafide P(fake)=0.002, spoof=0.938,
clean AUC 0.999). Only from a high baseline can you observe a "crater."

| Condition | EER | Accuracy (@0.5) |
|---|---|---|
| **Clean (control)** | **1.7%** | 97.3% |
| Opus 16k | 0.3% | 98.3% |
| Opus 24k | 0.3% | 97.7% |
| Opus 32k | 0.3% | 97.3% |

**Compression did not degrade detection.** EER stayed ≤1.7% (actually improved slightly —
noise). Accuracy held ~97–98%. The thesis predicted a slide toward ~70%; the observed
drop is **≈0 points**.

## 2. Supporting tracks

| Track | Clean EER | Opus 16/24/32k EER | Reading |
|---|---|---|---|
| **asvspoof** (in-distribution) | 1.7% | 0.3 / 0.3 / 0.3% | Strong baseline, **no compression cliff** |
| **benchmark** (In-the-Wild) | 23.3% | 22.0 / 22.7 / 21.3% | Weak baseline, **flat under compression** |
| **controlled** (LibriSpeech vs XTTS-v2) | ~chance | ~chance | Detector **blind to modern clones** (see §4) |

The compression effect is flat in **every** track that has any signal at all. There is no
bitrate at which detection collapses. Full numbers: `results/metrics.csv`. Plots:
`results/plots/{accuracy,eer}_vs_bitrate.png`, `score_dist_clean_vs_24k.png`.

## 3. Verdict — on the compression thesis as framed

> **NO-GO on "Opus compression is the wedge."** A working detector holds ~97% accuracy
> through Meet-realistic Opus bitrates. Opus is a modern, high-quality codec that
> preserves the spectral cues these detectors rely on; squeezing bitrate to 16 kbps did
> not break it. Building a product around "we survive compression others don't" is not
> supported by this evidence.

**Important scope caveats (why this is a lean, not an absolute, no-go):**
- **Only the Opus codec/bitrate was isolated.** The full live-call chain — WebRTC AEC/NS/AGC,
  packet loss, jitter/PLC, resample artifacts — was **not** tested (it was the optional
  stretch in the brief). Real-call degradation, if any, may live *there*, not in bitrate.
- **Telephony codecs not tested.** G.711/AMR/narrowband (8 kHz) are harsher than Opus and
  are common in phone-bridge / dial-in legs. Untested here.
- **One detector.** A single (good) model, not a survey of "every commodity detector."

If the "~70%" figure in the original thesis came from a real source, it was likely under
one of those harsher conditions — not Opus bitrate alone.

## 4. The finding that actually matters — generational blindness

The **controlled** track surfaced something sharper than the compression question:

- The ASVspoof-trained detector scores **98% on 2019-era attacks** but is **completely blind
  to 2023 XTTS-v2 voice clones** — it rates them P(fake)=0.002, identical to genuine speech
  (see the controlled track: EER ≈ chance, 0% of XTTS clones flagged).
- This is **codec-independent** and far larger than any compression effect: a ~98-point gap
  between "attacks it was trained on" and "current-generation TTS."

**The real wedge that showed up is temporal, not acoustic:** commodity detectors are
obsolete against the TTS people can actually use today. That is a more defensible, more
urgent product thesis than compression — and it's testable cheaply as a Phase 0b.

## 5. Method notes / credibility

- **Detector selection was empirical, not assumed.** First pick `MelodyMachine/Deepfake-
  audio-detection-V2` scored *all* clean speech (LibriSpeech and XTTS) at P(fake)=0.000 —
  overfit, no generalization (EER ~46%). A shootout (`scratchpad/shootout.py`) picked the
  ASVspoof model, validated by the smoke-test gate (`src/detect.py`, AUC ≥ 0.96 on held-out
  clips) *before* the pipeline was trusted.
- **Same clips through the codec**, three independent data sources, EER as the headline
  (scores are zero-inflated, so single-threshold accuracy is secondary).
- **Pipeline sanity:** uncompressed control matches the clean baseline, so the pipeline adds
  no degradation of its own.

## 6. Recommendation (for the human decision — Phase 0 pauses here)

1. **Do NOT build the product around compression robustness** on this evidence. As tested,
   the cliff isn't there.
2. **Before fully closing the compression thesis, spend one more cheap day** on the *untested*
   degradations that actually differ from clean Opus: full WebRTC processing chain + packet
   loss, and narrowband telephony codecs (G.711/AMR @ 8 kHz). If detection survives *those*
   too, close it hard.
3. **Seriously consider pivoting the wedge to §4 — generational blindness.** "Commodity
   detectors can't see current-gen TTS (XTTS-v2, F5, StyleTTS2)" is a bigger, cleaner gap
   than compression, and Phase 0 already has the harness to quantify it (swap the eval set,
   reuse everything else).

---

*Phase 0 complete. Numbers are real and reproducible (`python src/evaluate.py`). Per the
brief: stopping here for a human go/no-go before any Phase 1 work.*
