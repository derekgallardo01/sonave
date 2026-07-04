# Sonave — Detector v0: catching modern fake voices

**Goal:** build a detector that catches *current-generation* AI voice clones, which
Phase 0 showed commodity detectors are completely blind to.

**Result (held-out test, speakers never seen in training):**

| Model | Modern-clone catch | Old-attack catch | Real-voice accuracy | AUC | EER |
|---|---|---|---|---|---|
| Commodity (`Bisher/wav2vec2_ASV`) | **0.0%** | 95.0% | 100% | 69.6 | 36.5% |
| **Ours (`sonave_v0`)** | **97.9%** | 98.0% | 99.3% | 100.0 | **0.8%** |

The commodity detector misses **every** modern clone. Ours catches **97.9%** of them
— on speakers it never trained on — while keeping old-attack detection (98%) and not
crying wolf on genuine speech (99.3% of real voices pass).

## How

- **Base:** `Bisher/wav2vec2_ASV_deepfake_audio_detection` (wav2vec2-base, 95M).
  Already ~98% on 2019-era attacks; blind to modern TTS.
- **Fine-tuned** (`src/train_detector.py`) on a mix that ADDS modern XTTS-v2 clones,
  while keeping ASVspoof spoof clips in the blend so it doesn't forget old attacks.
  Conv feature encoder frozen; transformer + head trained, 4 epochs, fp16, RTX 5060.
- **Data** (`src/build_trainset.py`, `src/generate_trainfakes.py`): 1,880 clips.
  - real: LibriSpeech + ASVspoof bonafide
  - fake_old: ASVspoof 2019 LA spoof
  - fake_modern: 480 XTTS-v2 clones
  - **Speaker-disjoint split:** 28 speakers train / 12 speakers test. No speaker and
    no clip is shared, so the test measures generalization, not memorization.

## The honest caveat (this is a v0, not a finished product)

- **Same TTS family.** The modern test fakes are still **XTTS-v2** — a different
  *speaker* split, but the same *generator*. So this proves the model generalizes to
  **unseen speakers**, not yet to an **unseen TTS system**. It may have learned
  "XTTS-v2 fingerprints" as much as "modern-fakeness" in general.
- **Pipeline-artifact risk.** Because we generate the clones ourselves, there's a
  standard risk the model latches onto a quirk of our audio pipeline rather than the
  fake itself. (The commodity model saw the identical pipeline and still scored 0%,
  which is reassuring but not conclusive.)
- **Both caveats are killed by the same next test.**

## Next validation (the one that matters)

Test `sonave_v0` on modern fakes from a **different generator it never saw** —
F5-TTS, StyleTTS2, Piper, or ElevenLabs samples. If it still catches them, the claim
"catches modern fakes, broadly" holds and the pipeline-artifact worry is ruled out.
If it drops, we retrain on a mix of *several* TTS systems (the known fix for
cross-generator generalization) and re-test.

## Cross-generator validation (the caveat, tested)

Tested `sonave_v0` on **In-the-Wild** — real-world deepfakes from unknown tools, an
entirely external generation pipeline the model never saw (neither the fakes nor a
single training clip came from it):

| On In-the-Wild (external) | Catch rate | Real-voice acc | EER |
|---|---|---|---|
| Commodity (`Bisher`) | 4.0% | 99.3% | 23.0% |
| **Ours (`sonave_v0`)** | **61.3%** | 92.0% | 19.3% |

**Reading — real but partial generalization.**
- It catches **61% of totally external deepfakes** vs the commodity model's 4%. That
  ~15× jump on fakes made by *other* tools **rules out the "just memorized our
  pipeline" worry** — if it had only learned XTTS/pipeline fingerprints, it would
  score ~0% here, not 61%.
- But 61% is well below the 98% it hits on in-family (XTTS) fakes. **Training on one
  generator gives strong in-family detection and only partial cross-generator reach.**
- Real-voice accuracy dips to 92% (slightly trigger-happy on noisy real-world audio).

**The known fix:** train on a *mix* of several TTS systems, not one. That is the
standard recipe for cross-generator generalization and the clear next build step.

## v1: two generators in training (XTTS + YourTTS)

Added a second, architecturally-different generator (YourTTS, flow-based VITS
voice-cloning) to the training mix — 300 clones — and retrained. Same untouched
In-the-Wild external test:

| On In-the-Wild (external) | Modern catch | Real-voice acc | EER |
|---|---|---|---|
| Commodity (`Bisher`) | 4.0% | 99.3% | 23.0% |
| v0 — XTTS only | 61.3% | 92.0% | 19.3% |
| **v1 — XTTS + YourTTS** | 60.0% | **96.7%** | **15.3%** |

**Honest reading — a second generator improved *reliability*, not *coverage*.**
- Catch rate held at ~60% (did **not** jump). Two modern neural voice-cloners
  (XTTS, YourTTS) are similar enough that the second added little new coverage of
  the diverse real-world fakes in In-the-Wild.
- But false alarms on real speech dropped (real acc 92% → 96.7%) and overall EER
  improved ~20% relative (19.3% → 15.3%). The detector got **better calibrated**.

**What this tells us about the path to production:**
- The commodity → ours gap is real and large (4% → 60% catch).
- Raising catch rate further is a **diversity** problem, not a "more of the same"
  problem. The next levers, in rough priority:
  1. **More *diverse* fake types** in training — not another neural cloner, but
     different families (diffusion TTS e.g. Bark/Tortoise, vocoder-based, converted
     speech, partial splices), ideally including samples resembling the deployment
     distribution.
  2. **A stronger base model** — e.g. XLS-R-SLS (2024 SOTA for unseen-attack
     generalization) instead of wav2vec2-base.
  3. **Target-like data** — a small, *labelled* slice of the actual channel we ship
     into (meeting audio) moves generalization more than any amount of synthetic
     variety.

## Reproduce

```
python src/build_trainset.py            # assemble + split (uses on-disk data)
python src/generate_trainfakes.py       # 480 XTTS clones   (in .venv-tts)
python src/train_detector.py            # fine-tune         (in .venv)
python src/eval_detector.py             # before/after on held-out test
```

Artifacts: `models/sonave_v0/`, `results/detector_eval.csv`.
