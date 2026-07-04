# Sonave

**Real-time deepfake-voice detection for video meetings — engineered to work on the
compressed, processed audio real calls actually produce.**

Sonave flags AI-cloned voices on live Google Meet / Zoom calls (wire-fraud / finance
vertical). A Recall bot captures meeting audio → a fine-tuned **XLS-R + SLS** detector
scores each speaker → a live **REAL / SUSPECT / FAKE** verdict.

---

## Status: working detector + live capture/scoring pipeline

Full history and every honest caveat live in
[`results/detector_v2_progress.md`](results/detector_v2_progress.md). Highlights:

- **Modern-fake detection (clean audio):** ~91% catch on *unseen* commercial tools
  (ElevenLabs / Cartesia / Gemini) vs commodity ~2%; unseen-generator EER ~7.5%.
- **Real-call robustness:** holds through Google-Meet Opus; real-world false-alarms
  cut ~55% → ~8% by adding real-world real speech to training.
- **Meet domain (Stage 6):** a *balanced* real+fake corpus captured through a **live
  Meet** (VB-CABLE) gives **99% balanced accuracy** on held-out Meet windows (98%
  real-correct, 100% fake-caught) — fixing both the real false-positive and the
  fake-blindness. *Caveat: same-source held-out; cross-source validation is next.*
- **Product:** hosted CPU-only capture service ([`railway/`](railway/)) with a live
  authenticity badge; local GPU scoring via [`tools/verdict_monitor.py`](tools/verdict_monitor.py).

### The original Phase 0 thesis — DISPROVEN, then pivoted
Sonave began as a go/no-go test of one thesis: *Opus compression craters commodity
detectors.* **Phase 0 disproved it** — a competent detector held ~97% through every
bitrate ([`results/findings.md`](results/findings.md)) — but surfaced a bigger, real
gap: commodity detectors are **blind to current-generation voice clones** (~0–4%
catch). Everything since has built the detector that closes that gap *and* survives
real-call audio. The Phase 0 experiment below is preserved as historical record.

---

## Phase 0 (historical) — Compression Robustness Validation

The original go/no-go experiment: *how much detection accuracy is lost when audio is
degraded to Google Meet's Opus conditions?* Verdict: little — see `results/findings.md`.

### What the Phase 0 pipeline does

1. **Real speech** — 150 LibriSpeech utterances across 40 speakers (`controlled` track).
2. **Fake speech** — two independent sources:
   - `controlled`: XTTS-v2 **voice-clones of the same LibriSpeech speakers**, so a
     real clip and its fake differ only in real-vs-synthetic (+ later, compression).
   - `benchmark`: a slice of the **In-the-Wild** labelled deepfake dataset (trusted
     ground truth, independent of our generation).
3. **Compression** — every clip is round-tripped through **Opus** (mono, VoIP) at
   **16k / 24k / 32k**, plus an uncompressed control.
4. **Scoring** — an open-source SSL anti-spoofing model scores every clip at every
   condition. The *same* clips are scored clean and compressed, so any delta is
   purely the codec's doing.
5. **Metrics** — accuracy + **EER** per (track, bitrate). Headline accuracy uses the
   EER threshold **calibrated on clean audio, held fixed on compressed** — the
   realistic deployment number. Results land in `results/`.

---

## Setup

Two isolated virtual environments (the TTS stack pins a torch that fights the
detector's). **The RTX 50-series / Blackwell needs the cu128 torch wheels** — the
default `pip install torch` gives a build that can't use the GPU.

```powershell
# Detector + evaluation env
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Isolated XTTS generation env
python -m venv .venv-tts
.venv-tts\Scripts\Activate.ps1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-tts.txt
```

Requires **ffmpeg with libopus** on PATH.

## Run Phase 0

```powershell
# 1. real data + In-the-Wild benchmark slice + manifest.csv   (in .venv)
python src\prepare_data.py

# 2. XTTS-v2 clones of the LibriSpeech speakers               (in .venv-tts)
python src\generate_fakes.py

# 3. Opus compression sweep                                   (in .venv)
python src\compress.py            # or: --check to smoke-test one clip

# 4. score clean vs compressed -> metrics, plots, findings    (in .venv)
python src\evaluate.py

# detector smoke test (needs one real + one fake present)
python src\detect.py
```

**Output:** `results/findings.md` (go/no-go verdict), `results/metrics.csv`, and
three plots in `results/plots/`.

## Layout

```
config.py            central paths / knobs / bitrate sweep
src/prepare_data.py  download + subset LibriSpeech & In-the-Wild -> manifest.csv
src/generate_fakes.py  XTTS-v2 voice clones (isolated .venv-tts)
src/compress.py      Opus round-trip via ffmpeg
src/detect.py        load model -> score_wav(path) -> P(fake); smoke test
src/evaluate.py      clean-vs-compressed accuracy/EER + plots + findings draft
results/findings.md  the go/no-go writeup
```

## Reading the result

- **Big drop (~15-20+ pts, or accuracy sliding toward ~70%)** → thesis confirmed,
  proceed to Phase 1 (codec-augmented fine-tuning to recover accuracy).
- **Small / no drop** → commodity detectors already tolerate compression; the wedge
  is thin. Stop and reconsider the premise.
- A drop appearing in **both** the controlled and benchmark tracks is the credible
  signal — it rules out "this one fake generator just happens to be fragile."

*Phase 0 is cheap insurance: spend a weekend to save months. Respect the pause at
the end of it.*
