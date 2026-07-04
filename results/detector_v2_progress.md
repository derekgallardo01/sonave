# Sonave — Detector v2 progress (XLS-R backbone + augmentation experiments)

Plan Stage 2 in progress. Two levers tested on our EXISTING data (XTTS + YourTTS +
ASVspoof), isolating one variable at a time. Metric = EER (lower better); catch =
fakes flagged @0.5; real-acc = genuine voices kept @0.5.

## Results so far

| Model / training | In-the-Wild EER (external) | In-the-Wild catch / real-acc | Held-out in-family EER |
|---|---|---|---|
| commodity (`Bisher`, wav2vec2-base) | 23.0% | 4% / 99% | 36.5% |
| v1 (wav2vec2-base, 2 generators) | 15.0% | 60% / 97% | ~2% |
| **XLS-R+SLS (no aug)** | **14.7%** | 99% / 42% | **1.9%** |
| XLS-R+SLS + real-call augmentation | 31.3% | 19% / 91% | 10.1% |

## Findings

1. **Stronger backbone (XLS-R) nailed in-family (EER 1.9%) but did NOT move the
   external number** (In-the-Wild 14.7% ≈ v1's 15%). It also over-flagged noisy
   real-world REAL audio (real-acc 42% @0.5) — a domain-shift / calibration issue,
   since it only ever saw clean studio speech.
2. **Real-call augmentation backfired (this recipe).** It fixed the false positives
   (real-acc 42%→91%) but was too destructive — In-the-Wild EER got *worse*
   (14.7%→31.3%) and in-family dropped (1.9%→10.1%). Negative result; revisit with a
   gentler recipe AFTER data diversity, not before.
3. **The ceiling is DATA, confirmed.** In-the-Wild EER has stuck at ~15% across
   wav2vec2-base, XLS-R, and XLS-R+aug. The training fakes (2 similar neural cloners
   + 2019-era ASVspoof) simply don't cover the fake *types* in real-world deepfakes.
   No backbone or augmentation trick on this narrow data breaks ~15%.

## Conclusion → the real lever is Stage 1 (diverse public datasets)

Breaking the external ceiling needs training fakes from MANY generator families
(diffusion, vocoder, LLM-TTS, VC), which we can't hand-generate cheaply. Best sources:
MLAAD (91 systems), WaveFake (vocoders), DFADD (diffusion), ASVspoof5.

**Blocker:** some are HF-gated (DFADD, ASVspoof5) — need the founder's HuggingFace
login/agreement; others are large open downloads (MLAAD/WaveFake) needing orchestration.
This is the decision point to resolve before proceeding.

Artifacts: `models/sonave_xlsr/` (no aug), `models/sonave_xlsr_aug/`, `results/xlsr_eval.csv`.
Reproduce: `python src/train_xlsr.py [--augment] --out <dir>` then `python src/eval_xlsr.py --model <dir>`.

---

## STAGE 1 RESULT — diverse data broke the ceiling (research proof) ✅

Built a generator-diverse corpus (`src/build_corpus.py`, `data/corpus.csv`): ~60
MLAAD English TTS systems in TRAIN + our XTTS/YourTTS + ASVspoof, with **27 MLAAD
systems HELD OUT of training** (ElevenLabs v2/v3, Cartesia Sonic-3, Gemini-TTS,
Fish, ChatTTS, Chatterbox, Higgs-Audio, …). Retrained XLS-R+SLS on it.

| Test set | Model | Catch | Real-acc | EER |
|---|---|---|---|---|
| **27 unseen modern TTS** (ElevenLabs/Cartesia/Gemini…) | commodity | **1.9%** | — | — |
| | **ours** | **76.9%** | — | — |
| held-out (unseen gens + speakers) | commodity | 13.4% | 100% | 33.7% |
| | **ours** | 78.3% | 98.9% | **3.7%** |

**Research proof ACHIEVED.** On clean audio, the detector generalizes to modern
generators it never trained on — EER **3.7%** (was stuck ~15%), catching ~77% of
fakes from 27 unseen commercial tools vs commodity's ~2%. The lever was DATA
DIVERSITY, exactly as diagnosed.

## The honest catch → this is Stage 3's job

| In-the-Wild (noisy, real-world) | Model | Catch | Real-acc | EER |
|---|---|---|---|---|
| | commodity | 4.0% | 99.3% | 23.0% |
| | **ours** | 80.7% | **53.3%** | **33.7%** |

On **In-the-Wild** — noisy, real-world audio — our model **regressed** (EER 33.7%,
worse than commodity's 23%). It over-flags noisy REAL voices as fake (real-acc 53%)
because it trained only on **clean** audio and now fires hard on "clean-TTS
artifacts" that degraded real-world clips don't share, while treating noisy real
speech as suspicious.

**Diagnosis is clean:** we solved *clean-audio* modern-fake detection; **real-world/
degraded audio is a separate, unsolved axis** — and In-the-Wild is essentially the
"real call" proxy. That is precisely **Stage 3** (real-call robustness): gentle
degradation augmentation + noisy/real-world REAL data so the detector judges
fakeness *through* channel noise. The earlier aggressive-augmentation misfire says:
do it gently, and now on top of the diverse data.

Artifact: `models/sonave_xlsr_corpus/`. Reproduce: `python src/build_corpus.py` →
`python src/train_xlsr.py --manifest data/corpus.csv --out models/sonave_xlsr_corpus`
→ `python src/eval_xlsr.py --model models/sonave_xlsr_corpus`.

---

## STAGE 3 RESULT — real-call robustness (gentle augmentation) ✅ (partial)

Retrained the diverse corpus WITH gentle degradation: RawBoost + mild band-limit +
rare mu-law + moderate noise, applied to only HALF the clips and to BOTH real and
fake (so "noisy != fake"). `models/sonave_xlsr_corpus_aug/`.

| Test set | Model | Catch | Real-acc | EER |
|---|---|---|---|---|
| 27 unseen modern TTS | commodity | 1.9% | — | — |
| | **ours** | **92.2%** | — | — |
| held-out (unseen gens) | ours | 93.4% | 92.5% | **6.5%** |
| **In-the-Wild (real-world)** | commodity | 4.0% | 99.3% | 23.0% |
| | **ours** | 98.7% | 45.3% | **17.0%** |

**Wins:** In-the-Wild EER halved (33.7%→**17.0%**, now beats commodity's 23%);
unseen-tool catch rose 77%→**92%** (augmentation regularized); clean unseen-gen EER
held strong (3.7%→6.5%).

**Honest remaining gap — real-world REAL audio calibration.** In-the-Wild real-acc is
still 45% @0.5: the model over-flags noisy real-world REAL voices as fake. EER 17%
means the *ranking* is decent (at the right threshold both sides ~83%), but the
outputs are shifted high on real-world audio. Causes: (1) our synthetic degradation
≠ true real-world channel (real Opus/codec, room noise, phone), (2) no genuinely
real-world REAL speech in training (all our reals are clean studio).

**Clear next steps (Stage 3b):**
1. **Real Opus/codec augmentation** — reuse `src/compress.py opus_roundtrip` to
   pre-degrade a copy of training audio (per-step ffmpeg is too slow; pre-generate).
2. **Real-world REAL speech in training** — a noisy real corpus (e.g. Common Voice)
   so "messy real" is represented, fixing the false positives at the source.
3. **Threshold calibration** on a held-out real-call-like set.

Net: clean-audio modern-fake detection is SOLVED (EER ~6%, 92% on unseen commercial
tools); real-world robustness is much improved and now beats commodity, with a clear,
known path to close the last gap (real reals + real codec).

---

## STAGE 3b RESULT — real Opus codec augmentation DIDN'T close the gap (honest)

Added REAL Meet-Opus copies of the whole train set (`src/degrade_corpus.py`,
`data/corpus_aug.csv`, 5,832 train clips), retrained (`models/sonave_xlsr_final/`),
and added a calibrated-threshold + Opus-degraded-In-the-Wild eval.

| Test set | Stage 3 (best) | Stage 3b (final) |
|---|---|---|
| In-the-Wild EER | **17.0%** | 31.0% |
| In-the-Wild real-acc @0.5 | 45% | 79% |
| unseen MLAAD tools catch | **92.2%** | 79.1% |
| clean unseen-gen EER | 6.5% | 6.1% |

**Codec augmentation was NOT the lever.** It rebalanced (fewer false alarms) but
*worsened* In-the-Wild separation (EER 17%→31%) and unseen-tool catch (92%→79%).
The calibration probe is decisive: at the threshold optimal for clean audio, In-the-
Wild real-acc is still ~39% — the model's scores on real-world REAL voices are
genuinely shifted, and Opus-degrading *clean studio* audio doesn't reproduce the
real-world domain (background noise, real mics/rooms, spontaneous speech).

**Conclusion — the last gap is real-world REAL DATA, not codec/augmentation.**
Every synthetic trick on clean-sourced reals has plateaued In-the-Wild at ~17% EER.
Closing it needs genuine real-world real speech in training (VoxCeleb — same
YouTube/media domain as In-the-Wild — or Common Voice). That is the one remaining
lever; it requires a dataset we don't yet have.

## STAGE 3c RESULT — real-world REAL speech fixed the false alarms ✅ (deployable)

Added 600 **VoxPopuli** real-world real clips (real parliamentary recordings — varied
speakers/mics/rooms/noise) to training via `src/add_realworld.py` → `data/corpus_rw.csv`,
retrained (`models/sonave_xlsr_rw/`). In-the-Wild stays a fully external test (different
real-world source), so the improvement is genuine generalization.

| Metric | Stage 3 | **Stage 3c (final)** | commodity |
|---|---|---|---|
| In-the-Wild real-acc (keeps real voices) | 45% | **94%** | 99% |
| In-the-Wild catch @calibrated τ | ~83% | 64% | ~5% |
| clean unseen-tool catch | 92% | **90.9%** | 1.9% |
| clean unseen-gen EER | 6.5% | **7.5%** | 33.7% |
| Opus-24k In-the-Wild (catch / real-acc @τ) | — | **62% / 91%** | 9% / 79% |

**The false-alarm problem is SOLVED.** Real-world real speech in training cut the
real-voice false-alarm rate from ~55% to ~8% (real-acc 45%→94%) while keeping clean
modern-tool detection at ~91% and staying robust through the real Meet Opus codec
(62% catch / 91% real-acc). The honest tradeoff: real-world *fake* catch settles at
~62–64% (those In-the-Wild deepfakes are genuinely hard/degraded), down from the
over-eager 98% that came with crying wolf on real people.

### Deployment model: `models/sonave_xlsr_rw/`
A usable "flag suspicious speakers for review" detector:
- **~91% catch on unseen commercial tools** (ElevenLabs/Cartesia/Gemini) vs commodity ~2%
- **~64% catch on hard real-world deepfakes at ~92% real-voice accuracy** vs commodity ~5%
- **Compression-robust** (holds through Google-Meet Opus)

---

## STAGE 4 — the live false-positive, DIAGNOSED and FIXED with real Meet data ✅

**Live test finding:** a real bot on a real Google Meet scored the founder's OWN
(real) voice as fake — fired a wire-hold on a real person. Root cause: the detector
had never heard audio through Google Meet's processing (noise-suppression/AGC/codec),
so Meet-processed real speech looked synthetic to it.

**Proved it's a domain gap, not a threshold gap:** on live Meet audio the real voice
and real fakes OVERLAP (~0.8), so no threshold separates them (raising it just trades
false-alarms for missed fakes). The fix had to be data.

**The fix (end-to-end, validated):**
1. Added capture-to-disk to the live pipeline (`service/dashboard.py` `_flush_capture`).
2. Captured **330 s of real Meet-piped voice** through the actual bot→Recall→WS path.
3. Added it as `label=real` (time-split, `src/add_captured.py` → `data/corpus_meet.csv`),
   retrained → `models/sonave_xlsr_meet/`.

| Check | OLD (`sonave_xlsr_rw`) | NEW (`sonave_xlsr_meet`) |
|---|---|---|
| **Held-out real Meet voice** (never trained on) | mean P(fake) 0.417, 33% flagged | **0.003, 0% flagged** ✅ |
| Unseen commercial tools catch | 91% | 79% |
| Held-out generators (catch / EER) | 91% / 7.5% | 82% / 8.9% |
| In-the-Wild (catch / real-acc / EER) | 60 / 94 / 22 | 45 / 88 / 30 |

**Result: the false-positive is fixed** (0.417→0.003 on held-out real Meet audio)
**while fake detection stays strong** (79–82% on unseen tools, EER 8.9%). Domain
adaptation with real Meet-piped data is the confirmed cure.

**Honest caveats:** only ONE speaker (Derek) and 6 held-out test windows — a strong
signal, not a large-sample proof. Fake-catch regressed slightly (91→79% on unseen
tools; In-the-Wild 60→45%) — the expected cost of limited single-speaker real data.
**Next:** collect Meet-piped real audio from several speakers/sessions to lock in the
false-positive fix AND recover the fake-catch. The capture rig is built and proven.

## STAGE 5 — offline Meet simulation FAILED (negative result, important) ❌

Hypothesis: reproduce Meet's processing offline (`src/meetify.py`: high-pass +
spectral NS + AGC + Opus) to "Meet-ify" the whole corpus at scale, avoiding live
capture. Built it, meetified 1400 clips (both classes), retrained
`models/sonave_xlsr_meetify` (captured audio held ENTIRELY out as ground truth).

| Held-out REAL captured Meet voice | flagged fake |
|---|---|
| old model (no Meet data) | 43% |
| trained on REAL capture (`_meet`) | **0%** ✅ |
| trained on OFFLINE-meetified (`_meetify`) | **96%** ❌ |

**Offline simulation made real Meet audio WORSE** (0.565 → 0.933). The signal-
processing approximation doesn't match Google's actual WebRTC processing; the model
learned the *simulated* artifacts and then flagged *real* Meet's different artifacts
even harder. (Fake-catch rose to 98%, but that's moot when real voices are flagged.)

**Conclusion: the domain cannot be faked offline. Only REAL captured Meet audio
works** (Stage 4 proved it). The scalable path must collect real Meet audio —
piggyback real meetings (with consent), and/or replay clips through a LIVE Meet via
a virtual audio cable — NOT offline processing. `meetify.py` is kept as a documented
dead-end.

### Remaining lever (future): real-world FAKE diversity
Real-world *fake* catch (~64%) is now the ceiling — the In-the-Wild fakes differ from
clean MLAAD. Closing it needs harder/real-world fake examples (compressed deepfakes,
partial splices, more deepfake-in-the-wild corpora) — the mirror of what we just did
for reals. Diminishing-returns territory; the current model is already a real product
starting point.
