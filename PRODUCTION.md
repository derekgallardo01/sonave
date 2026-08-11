# Sonave — Production Deployment Guide (Google Meet)

This guide walks through deploying Sonave for **live Google Meet deepfake detection** in production. The architecture is two services:

- **Railway** (CPU-only) — capture service that sends a Recall.ai bot into meetings, receives real-time audio, and displays the live dashboard.
- **Modal** (GPU, scale-to-zero) — detection microservice that scores audio chunks and returns REAL / SUSPECT / FAKE verdicts.

---

## 1. Prerequisites

| Requirement | How to get it |
|-------------|--------------|
| **Recall.ai API key** | [recall.ai](https://recall.ai) — bot joins Meet/Zoom/Teams and streams audio |
| **Modal account** | [modal.com](https://modal.com) — serverless GPU hosting |
| **Railway account** | [railway.app](https://railway.app) — CPU web service hosting |
| **GitHub repo** | Fork or push this repo to GitHub |

---

## 2. Configure secrets

### 2.1 Local `.env`

Copy `.env.example` to `.env` (gitignored) and fill in:

```bash
cp .env.example .env
```

Required for production:

```env
# Recall.ai
SONAVE_RECALL_API_KEY=your_recall_api_key_here
SONAVE_RECALL_BASE=https://us-west-2.recall.ai/api/v1

# Shared auth token (generate a strong random string)
SONAVE_API_TOKEN=your_long_random_token_here

# Modal deploy will read SONAVE_API_TOKEN from your shell env
```

Optional but recommended:

```env
# Slack alert webhook for wire-hold incidents
SONAVE_ALERT_WEBHOOK=https://hooks.slack.com/services/...

# Scoring cadence (faster = more responsive, more GPU cost)
SONAVE_SCORE_SEC=10
SONAVE_SCORE_WIN_SEC=8
SONAVE_SCORE_EMA=0.6
```

### 2.2 GitHub Repository Secrets (for auto-deploy)

Go to **GitHub repo → Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Value | How to get it |
|--------|-------|---------------|
| `MODAL_TOKEN_ID` | Your Modal token ID | `modal token new` then `modal token list` |
| `MODAL_TOKEN_SECRET` | Your Modal token secret | Same as above |
| `SONAVE_API_TOKEN` | Same token as in `.env` | `python -c "import secrets;print(secrets.token_urlsafe(32))"` |
| `HF_TOKEN` *(optional)* | HuggingFace token | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — speeds up backbone download |

**Railway** auto-deploys via its own GitHub integration (no GitHub secrets needed).

---

Copy `.env.example` to `.env` (gitignored) and fill in:

```bash
cp .env.example .env
```

Required for production:

```env
# Recall.ai
SONAVE_RECALL_API_KEY=your_recall_api_key_here
SONAVE_RECALL_BASE=https://us-west-2.recall.ai/api/v1

# Shared auth token (generate a strong random string)
SONAVE_API_TOKEN=your_long_random_token_here

# Modal deploy will read SONAVE_API_TOKEN from your shell env
```

Optional but recommended:

```env
# Slack alert webhook for wire-hold incidents
SONAVE_ALERT_WEBHOOK=https://hooks.slack.com/services/...

# Scoring cadence (faster = more responsive, more GPU cost)
SONAVE_SCORE_SEC=10
SONAVE_SCORE_WIN_SEC=8
SONAVE_SCORE_EMA=0.6
```

---

## 3. Deploy the GPU scorer (Modal)

The Modal app bakes the XLS-R backbone into the image and mounts the trained model.

```bash
# One-time auth
pip install modal
modal token new

# Deploy (persistent URL)
modal deploy modal_app.py

# Test
export SONAVE_API_TOKEN=your_token_here
curl -H "X-Sonave-Token: $SONAVE_API_TOKEN" https://<you>--sonave-detector-fastapi-app.modal.run/healthz
curl -F "file=@test.wav" -H "X-Sonave-Token: $SONAVE_API_TOKEN" https://<you>--sonave-detector-fastapi-app.modal.run/score_clip
```

**Modal URL:** `https://<you>--sonave-detector-fastapi-app.modal.run` — save this for the Railway env var.

---

## 4. Deploy the capture service (Railway)

### 4.1 Create project

1. Railway dashboard → **New Project** → **Deploy from GitHub repo** (your `sonave` repo)
2. In service **Settings → Root Directory**, set: `railway`
3. Add a **Volume** mounted at `/data` so captures survive redeploys

### 4.2 Environment variables

| Variable | Value |
|----------|-------|
| `SONAVE_RECALL_API_KEY` | Your Recall key |
| `SONAVE_RECALL_BASE` | `https://us-west-2.recall.ai/api/v1` (match your region) |
| `SONAVE_SCORER_URL` | Modal URL from Step 3, e.g. `https://<you>--sonave-detector-fastapi-app.modal.run` |
| `SONAVE_API_TOKEN` | Same token you set for Modal |
| `SONAVE_DATA_DIR` | `/data/captured` |
| `SONAVE_ENROLL_DIR` | `/data/enrollments` (voiceprint persistence) |
| `SONAVE_MODEL_CACHE` | `/data/models/ecapa` (ECAPA model cache, optional) |
| `SONAVE_PUBLIC_DOMAIN` | Auto-set by Railway; only override if using a custom domain |
|----------|-------|
| `SONAVE_RECALL_API_KEY` | Your Recall key |
| `SONAVE_RECALL_BASE` | `https://us-west-2.recall.ai/api/v1` (match your region) |
| `SONAVE_SCORER_URL` | Modal URL from Step 3, e.g. `https://<you>--sonave-detector-fastapi-app.modal.run` |
| `SONAVE_API_TOKEN` | Same token you set for Modal |
| `SONAVE_DATA_DIR` | `/data/captured` |
| `SONAVE_PUBLIC_DOMAIN` | Auto-set by Railway; only override if using a custom domain |

### 4.3 Verify deploy

Open `https://<your-railway-domain>/`:

- You should see the Sonave dashboard
- If `SONAVE_API_TOKEN` is set, it prompts for the token
- Paste a Google Meet link → **Send bot**
- The bot joins and audio starts streaming

---

## 5. Speaker Enrollment (Voiceprint Verification)

Enrollment adds an **independent signal** to deepfake detection: "is this the person it's supposed to be?" Two independent checks → far fewer false positives and stronger catches, especially for wire-fraud scenarios where the caller claims an identity.

### How it works

1. **Capture real audio** — Send the bot into a meeting where the target speaker talks normally
2. **Enroll** — The Railway dashboard shows a speaker enrollment panel; click **Enroll** on the speaker's name
3. **Live fusion** — From then on, every scored window is checked against the stored voiceprint on Modal's GPU
4. **Dashboard** — Shows enrollment status, voiceprint match %, and the fused verdict

### Enrollment API

| Endpoint | Method | Body | Response |
|----------|--------|------|----------|
| `/api/enroll` | `POST` | `{"speaker_id": "Derek", "clip_names": ["meet_Derek_123_000.wav"]}` or omit `clip_names` to auto-select | `{"ok": true, "clips": 3}` |
| `/api/enrolled` | `GET` | — | `{"enrolled": [{"speaker_id": "Derek", ...}]}` |
| `/api/enroll/{speaker}` | `DELETE` | — | `{"ok": true}` |

### Voiceprint fusion scoring

When a speaker is enrolled, Railway automatically sends their voiceprint (base64-encoded numpy embedding) inline with every scoring request to Modal. Modal:

1. Scores the audio for deepfake (XLS-R detector)
2. Embeds the live audio via ECAPA-TDNN and compares to the enrolled voiceprint
3. **Fuses** the two signals: `risk = max(damped_deepfake, mismatch_risk)`
4. Returns the fused verdict with match confidence

The voiceprint itself is a small numpy array (~192 floats, ~1 KB base64) and is loaded from Railway's persistent `/data/enrollments/` volume.

### Architecture note

- **Railway** handles enrollment storage (`/data/enrollments/*.npy`) and builds voiceprints with CPU torch (one-time, lightweight)
- **Modal** receives voiceprint embeddings inline with scoring requests, does the fusion on GPU
- No sync needed — the voiceprint travels with each request

---

## 6. End-to-end smoke test

1. **Start a Google Meet** (or join an existing one)
2. **Copy the Meet URL** and paste into the Railway dashboard
3. **Send bot** — you should see:
   - Bot appears in the meeting as "Sonave"
   - Dashboard shows speaker cards with audio quality meters
   - After ~10 seconds, authenticity badges appear (REAL / SUSPECT / FAKE)
4. **Test with a fake**: play an AI-generated voice clip into the meeting → badge should flip to FAKE and a wire-hold alert fires

---

## 7. Health checks & monitoring

| Endpoint | Purpose |
|----------|---------|
| Modal `GET /healthz` | Platform probe — returns `{"status":"ok", "device":"cuda", ...}` |
| Modal `GET /ready` | Deep readiness — confirms model loaded and can score |
| Railway `GET /` | Dashboard page |
| Railway `GET /api/quality` | Per-speaker audio quality + authenticity verdicts |
| Railway `GET /api/incidents` | Open/acknowledged fake-voice incidents |

**Recommended**: Set your platform's health probe to Modal's `/healthz` (open) and Railway's `/` (auth-gated, so configure probe to send the token).

---

## 8. Security checklist

- [ ] `SONAVE_API_TOKEN` is set and identical on Modal + Railway
- [ ] `.env` is gitignored and never committed
- [ ] Recall key has appropriate scope (bot creation only)
- [ ] Railway volume at `/data` persists captures across deploys
- [ ] Railway volume at `/data/enrollments` persists voiceprints across deploys
- [ ] Modal `SONAVE_MAX_UPLOAD_MB` limits upload size (default 25 MB)
- [ ] Meeting URL allowlist restricts to Google Meet / Zoom / Teams only
- [ ] Incident DB (`incidents.db`) lives on the persistent volume
- [ ] `.env` is gitignored and never committed
- [ ] Recall key has appropriate scope (bot creation only)
- [ ] Railway volume at `/data` persists captures across deploys
- [ ] Modal `SONAVE_MAX_UPLOAD_MB` limits upload size (default 25 MB)
- [ ] Meeting URL allowlist restricts to Google Meet / Zoom / Teams only
- [ ] Incident DB (`incidents.db`) lives on the persistent volume

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Bot sent but no audio appears | Wrong `SONAVE_RECALL_WS` or domain | Check Railway domain is public and `wss://` URL is correct |
| "scoring…" never updates | `SONAVE_SCORER_URL` not set or Modal cold start | Verify Modal URL; first request after idle may take 10–15 s |
| Real voices flagged fake | Detector hasn't seen Meet-processed real audio | Collect real Meet audio via VB-CABLE and retrain (see `results/detector_v2_progress.md`) |
| High false-positive rate | Threshold too aggressive | Raise `SONAVE_TAU_FAKE` (default 0.70) or use voiceprint enrollment |
| Enrollment fails | Not enough captured clips (need ≥1) or speaker name mismatch | Check captures exist for that speaker; names are case-sensitive |
| Voiceprint match shows 0% | ECAPA model download failed on Railway first run | Check `/data/models/ecapa/` exists on the volume; retry enrollment |
| Capture files missing | Volume not mounted at `/data` | Add Railway volume at `/data` |
| Capture files missing | Volume not mounted at `/data` | Add Railway volume at `/data` |

---

## 10. Updating the model

1. Retrain locally: `python src/train_xlsr.py --manifest data/corpus_meet.csv --out models/sonave_xlsr_meet`
2. Update `modal_app.py` model path if the directory name changed
3. Redeploy Modal: `modal deploy modal_app.py`
4. The Railway capture service needs no changes

---

## 11. Auto-deploy behavior

The repo includes `.github/workflows/ci-cd.yml`. On every **Pull Request**, it runs the fast test suite. On every **push to `main`**, it:

1. Runs tests
2. If tests pass → auto-deploys Modal
3. Railway auto-deploys independently via its GitHub integration

You can monitor deploys in **GitHub → Actions**.

---

## 12. Costs (rough)

| Component | Cost |
|-----------|------|
| Railway (CPU, always-on) | ~$5–10/mo |
| Modal GPU (T4, scale-to-zero) | ~$0.50–2/hr active, $0 idle |
| Recall.ai bot | ~$0.05–0.10/min of meeting |

A 1-hour Meet with continuous scoring ≈ **$1–3 total** vs. ~$400/hr for always-on GPU.
