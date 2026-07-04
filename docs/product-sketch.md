# Sonave — Product Architecture Sketch

*From working detector → shippable product. Grounds the original brief (Recall.ai
capture → detection → orchestration → wire-fraud workflow) in what we've actually
built: `models/sonave_xlsr_rw/`.*

---

## 0. The moat we already have

The product's core asset exists and is validated:

- **Catches ~91% of fakes from unseen modern commercial tools** (ElevenLabs, Cartesia,
  Gemini) where commodity detectors catch ~2%.
- **~64% catch / ~92% real-voice accuracy on hard real-world deepfakes** at a
  calibrated threshold (commodity ~5%).
- **Compression-robust** — holds through the Google-Meet Opus codec (our detector was
  trained on real Meet-codec + real-world audio).
- **Cheap to run** — it's OUR wav2vec2/XLS-R model on a GPU (~50–150 ms per 4 s clip),
  not a commodity API. This is the economic moat (see §4).

Everything below wraps this asset. Nothing here requires a new detector.

---

## 1. System at a glance

```
 Calendar ──▶ Recall.ai bot ──▶ audio stream ──▶ Orchestrator ──▶ Detection API ──▶ scores
 (triggers)   (joins meeting)   (per-speaker)     (windows,        (our model)        │
                                                   active-speaker,                     ▼
                                                   rolling score) ◀───────────── verdict/conf
                                                        │
                            ┌───────────────────────────┼───────────────────────────┐
                            ▼                            ▼                            ▼
                     Live dashboard             Alerts (Slack/SMS)          Forensic log +
                     (per-speaker RAG)          on threshold cross          exportable report
                                                        │
                                                        ▼
                                          Wire-fraud hook: "hold the
                                          transaction / require re-auth"
```

Four layers, matching the brief's Phases 2–5:

| Layer | Phase | Build vs rent | Status |
|---|---|---|---|
| Detection service | 2 | **Build** (wrap our model) | model done; API next |
| Capture (meeting bot) | 3 | **Rent** (Recall.ai) | not started |
| Orchestration & cost | 4 | **Build** | not started |
| Product & workflow | 5 | **Build** | not started |

---

## 2. Detection service (Phase 2) — the first thing to build

A small, containerized service wrapping `sonave_xlsr_rw`. Stateless per request; the
orchestrator owns all state.

**API**
```
POST /score
  body: { audio: <wav/opus bytes, ~4s, 16kHz mono>, speaker_id: str, ts: float }
  resp: { p_fake: 0.0–1.0, verdict: "real"|"suspect"|"fake",
          confidence: 0.0–1.0, model_version: "sonave_xlsr_rw", latency_ms: int }

GET /healthz    → model loaded, GPU available
GET /version    → model + threshold-policy version
```

- `verdict` uses a **calibrated threshold policy** (not raw 0.5). From our eval,
  τ≈0.4 gives ~64% catch / ~92% real-acc on real-world audio; expose τ as config so
  the false-alarm tolerance is a product knob, not a hardcode.
- Reuse `src/model_sls.py` (`SLSDetector.load`, `score_paths`) directly.
- Batches concurrent chunks for GPU efficiency. FastAPI + a single worker owning the
  model; Docker image with the cu128 torch base.

**Why this first:** it's buildable *now* from what we have, and it's the seam every
other layer plugs into. Ship it behind a URL and the rest is integration.

---

## 3. Capture layer (Phase 3) — rented, not built

Per the brief: **do NOT build a headless-Chrome bot.** Use **Recall.ai**.

- Calendar connect (Google/M365) → bot auto-joins scheduled Meet/Zoom/Teams calls.
- Streams audio out; prefer **per-participant audio** if the platform exposes it, else
  mixed audio + Recall's speaker labels / our own diarization.
- Use **signed-in bots** (skip waiting rooms) and **login groups** for concurrency.
- Start with **Meet + Zoom**.

Output to the orchestrator: a live, labelled audio stream `(speaker_id, pcm_chunk, ts)`.

---

## 4. Orchestration & cost control (Phase 4)

The brief called cost "the product" because continuous *commodity-API* detection runs
~$400/meeting-hour. **We changed that equation by owning the model** — but smart
orchestration still matters for scale and latency.

**Pipeline**
1. **VAD / active-speaker** — only score the speaker currently talking (from Recall
   labels or a light VAD). No point scoring silence.
2. **Windowing** — 4 s windows (matches training), ~2 s hop for responsiveness.
3. **Rolling per-speaker score** — EWMA over a speaker's windows → a stable confidence
   that shrugs off single-window noise (important given per-clip variance).
4. **Trigger-on-suspicion** — because our model is cheap we can run it near-continuously
   as the cheap first pass; *escalation* (denser windowing, longer context, human/2nd-model
   review, and the wire-hold hook) fires only when a speaker's rolling score crosses amber/red.
5. **Cost-per-meeting-hour** is a tracked, first-class metric on the dashboard.

**Cost sanity check (the moat, quantified):** one active speaker, 4 s windows @ 2 s hop
= ~1,800 inferences/meeting-hour × ~100 ms GPU each ≈ **~3 GPU-minutes/meeting-hour** —
cents, not $400. One modest GPU covers many concurrent meetings. *Owning the detector
is the cost control.*

---

## 5. Product & workflow (Phase 5) — the wedge vertical

**Live surface**
- Per-speaker **RAG status** (green/amber/red) + rolling confidence, updating live.
- **Alerts**: Slack / email / SMS the moment a speaker crosses red.

**Forensic trail (compliance-grade)**
- Every window: score, verdict, timestamp, short audio snippet → immutable log.
- One-click **exportable report** per meeting (who/when/scores/snippets) — the audit
  artifact a finance/compliance team needs.

**First vertical: wire-transfer / finance authorization calls**
- Integration hook: **"hold the wire / require re-auth"** — when a speaker on an
  approval call trends red, Sonave can (a) surface a hard warning to the approver and
  (b) fire a webhook into the payment/approval system to pause the transaction pending
  a second factor. This is the concrete value: stop a deepfaked-CEO wire fraud mid-call.

---

## 6. MVP build sequence (recommended order)

1. **Detection microservice** (Phase 2) — wrap `sonave_xlsr_rw` in FastAPI + Docker,
   calibrated verdict, `/score`. *Buildable today.*
2. **Offline meeting analyzer** — feed a *recorded* meeting file → per-speaker timeline
   of scores + exportable report. Proves the full detection→scoring→report chain on
   real recordings **before** touching live infra. Cheap, high-signal demo.
3. **Recall.ai live integration** (Phase 3) — calendar → bot → stream → detection API →
   live rolling scores. Meet first.
4. **Alerting + live dashboard** (Phase 5 core).
5. **Wire-fraud workflow** (Phase 5 vertical) — the hold-the-wire webhook + forensic
   export, packaged for a design-partner finance team.

Steps 1–2 need nothing but what we have. Step 3 is where external cost/integration
starts. Each step is independently demoable.

---

## 7. Open decisions & risks

- **Per-participant vs mixed audio.** Per-participant is far cleaner for per-speaker
  scoring; confirm which platforms Recall exposes it for, else lean on diarization.
- **Operating threshold / false-alarm tolerance.** τ is a policy dial: finance wants
  high recall (catch fakes) but can't cry wolf on execs. Tune per design partner; the
  rolling-score + escalation design softens single-window false alarms.
- **Real-world fake catch (~64%).** Honest current ceiling on the *hardest* real-world
  fakes. Mitigations: (a) rolling score over a whole call raises effective catch vs
  per-clip, (b) add harder real-world fake training data later (the mirror of the
  VoxPopuli fix we did for reals).
- **Latency budget.** Target sub-second per verdict; batching + 2 s hop should hold it.
- **Privacy/consent.** Recording + biometric-ish analysis of meeting audio has legal/
  consent implications, especially finance — bake consent + data-retention policy in
  early.

---

*Grounding: detector results in `results/detector_v2_progress.md`; model in
`models/sonave_xlsr_rw/`; reusable pieces — `src/model_sls.py` (scoring),
`src/compress.py` (Opus handling), `src/eval_xlsr.py` (threshold calibration).*
