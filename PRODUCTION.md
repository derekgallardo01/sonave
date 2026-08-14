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

### 4.2b Multi-user: Google sign-in (optional but required for public users)

Create an OAuth client at **console.cloud.google.com → APIs & Services → Credentials →
Create credentials → OAuth client ID → Web application**, authorized redirect URI
`https://<your-railway-domain>/auth/callback` (add `http://localhost:8000/auth/callback` for dev).

| Variable | Value |
|----------|-------|
| `SONAVE_GOOGLE_CLIENT_ID` / `SONAVE_GOOGLE_CLIENT_SECRET` | From the OAuth client |
| `SONAVE_SESSION_SECRET` | `python -c "import secrets;print(secrets.token_urlsafe(32))"` |
| `SONAVE_ADMIN_EMAILS` | Comma list; these Google accounts become admin (unlimited, see all data) |
| `SONAVE_SIGNUP_MODE` | `open` (default) or `closed` (unknown emails rejected) |
| `SONAVE_APP_DB` | default `/data/app.db` (users/bots/billing; on the volume) |

On the first admin sign-in, pre-existing single-tenant data (flat captures,
voiceprints, incidents) migrates into that admin's workspace automatically
(idempotent; marker file `/.tenancy_migrated` on the volume).

### 4.2b Admin observability (optional pushes)

The console's **Admin** view (visible to admin accounts and the operator token)
shows overview tiles, a per-user rollup, and a live activity feed backed by the
`events` table — no setup needed. Growth pushes (new signup, subscription
added/canceled) are optional and env-gated; unset = silently off:

| Variable | Value |
|----------|-------|
| `SONAVE_ADMIN_WEBHOOK` | Slack incoming-webhook URL; growth events post here |
| `SONAVE_ADMIN_EMAIL` | Where growth emails go (e.g. the founder's Gmail) |
| `SONAVE_SMTP_HOST` / `SONAVE_SMTP_PORT` | SMTP relay; port default 587 (STARTTLS) |
| `SONAVE_SMTP_USER` / `SONAVE_SMTP_PASS` | SMTP login. Gmail: enable 2FA, then create an **App password** (Google Account → Security → App passwords) and use it as the password with `smtp.gmail.com` |

### 4.2b-2 OAuth Calendar auto-join (built, DARK until scope verification)

The OAuth variant of auto-protect ("Connect Google Calendar" — no URL pasting)
is fully implemented behind `SONAVE_CALENDAR_OAUTH` (default off). It uses the
**sensitive** `calendar.readonly` scope as an incremental, opt-in grant from
the console — never during sign-in. Enable ONLY after the Marketplace listing
is approved, in this order:

1. GCP → OAuth consent screen → **Data access** → add
   `https://www.googleapis.com/auth/calendar.readonly` and submit the scope
   verification (justification: "reads upcoming Google Meet events so the
   user's meetings can be protected automatically; read-only; opt-in").
2. Wait for scope approval (days; no CASA needed for sensitive-only).
3. Railway → set `SONAVE_CALENDAR_OAUTH=1` and redeploy. The "Connect Google
   Calendar" button appears in the console's Auto-protect card.

Disconnect (console button or revocation at myaccount.google.com) revokes the
refresh token at Google and deletes it from `oauth_tokens`; a revoked grant is
also detected by the join loop and cleaned up automatically. The zero-scope
iCal path keeps working independently either way.

### 4.2c Billing: Stripe metered (free 5 h/month, then $8/monitored-hour)

In the Stripe dashboard (test mode first): **Billing → Meters → Create meter**
(event name `sonave_monitored_minutes`, aggregation **sum**); **Product**
"Sonave" with a **usage-based Price** on that meter, per-unit
`$0.13333` per minute, monthly; **Developers → Webhooks → Add endpoint**
`https://<railway-domain>/api/billing/webhook` with events
`checkout.session.completed`, `customer.subscription.updated`,
`customer.subscription.deleted`.

| Variable | Value |
|----------|-------|
| `SONAVE_STRIPE_SECRET_KEY` | `sk_live_...` (or `sk_test_...`) |
| `SONAVE_STRIPE_WEBHOOK_SECRET` | `whsec_...` from the webhook endpoint |
| `SONAVE_STRIPE_PRICE_METERED` | `price_...` of the usage-based price |
| `SONAVE_STRIPE_METER_EVENT` | default `sonave_monitored_minutes` |
| `SONAVE_FREE_MINUTES` | default `300` (5 monitored hours/month free) |
| `SONAVE_MONTHLY_CAP_USD` | default `200` — bot launches blocked past this spend |
| `SONAVE_MAX_CONCURRENT_BOTS` | default `2` per non-admin user |

Unset Stripe vars = billing disabled: free tier is enforced, but no card flow
(users simply hit the quota). Note: the service must run as a **single worker**
(live state is in-memory; already the case with the default Procfile).

### 4.2d Google Meet add-on (preview — after brand verification clears)

The side panel is served at `https://usesonave.com/meet-addon` and works
standalone today. To surface it inside Google Meet:

1. Google Cloud console → enable the **Google Workspace Marketplace SDK** and
   **Google Workspace Add-ons API** on the Sonave project.
2. Marketplace SDK → App configuration → **Meet add-on**: side panel URL
   `https://usesonave.com/meet-addon`, logo `https://usesonave.com/og.png`
   (or the 120px icon), and the project number → Railway var
   `SONAVE_MEET_PROJECT_NUMBER`.
3. Publish to Marketplace (unlisted first for testing; public listing requires
   the completed brand verification).
4. Known limitation: inside Meet's iframe the session cookie is third-party —
   the panel shows a Sign-in button that opens the popup flow. Cookie
   partitioning (CHIPS) is the follow-up if browsers block it.

### 4.3 Verify deploy

- `https://<your-railway-domain>/` serves the public marketing landing page
- `https://<your-railway-domain>/console` is the operator console — it prompts for
  `SONAVE_API_TOKEN` (stored as a cookie for 30 days)
- In the console, paste a Google Meet link → **Deploy** — the bot joins, audio starts
  streaming, and the scale-to-zero scorer is pre-warmed in the background

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
2. **Copy the Meet URL** and paste into the console (`/console`)
3. **Deploy** — you should see:
   - Bot appears in the meeting as "Sonave" (deploying also pre-warms the Modal scorer,
     so the first verdict skips the cold start)
   - Console shows speaker cards with audio quality + risk meters
   - After ~10–20 seconds, authenticity badges appear (REAL / SUSPECT / FAKE) and the
     Median Latency tile shows measured GPU latency
4. **Test with a fake**: play an AI-generated voice clip into the meeting → the badge
   flips to FAKE, and after **3 consecutive** fake-band windows (`SONAVE_INCIDENT_STREAK`,
   ~30 s at the default cadence) the wire-hold incident fires

---

## 7. Health checks & monitoring

| Endpoint | Purpose |
|----------|---------|
| Modal `GET /healthz` | Platform probe — returns `{"status":"ok", "device":"cuda", ...}` |
| Modal `GET /ready` | Deep readiness — confirms model loaded and can score |
| Railway `GET /` | Public landing page (no auth — good probe target) |
| Railway `GET /console` | Operator console (page loads publicly; APIs behind the token) |
| Railway `GET /api/quality` | Per-speaker audio quality + authenticity verdicts (token) — includes a `_scorer` row showing whether `SONAVE_SCORER_URL` is configured |
| Railway `GET /api/incidents` | Open/acknowledged fake-voice incidents (token) |

**Recommended**: Set your platform's health probe to Modal's `/healthz` and Railway's `/` — both are public.

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

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Bot sent but no audio appears | Wrong `SONAVE_RECALL_WS` or domain | Check Railway domain is public and `wss://` URL is correct |
| Bot deploy fails after key rotation | New Recall key not saved to Railway | Update `SONAVE_RECALL_API_KEY` in the Railway service variables |
| Console shows "no scorer configured" | `SONAVE_SCORER_URL` not set | Set it to the Modal URL from Step 3 (the app also warns at startup) |
| Badges stay PENDING with scorer configured | Modal cold start or token mismatch | Deploying a bot pre-warms the scorer; verify the same `SONAVE_API_TOKEN` on both services |
| Real voices flagged fake | Detector hasn't seen Meet-processed real audio | Collect real Meet audio via VB-CABLE and retrain (see `results/detector_v2_progress.md`) |
| High false-positive rate | Threshold too aggressive | Raise `SONAVE_TAU_FAKE` (default 0.70) or use voiceprint enrollment |
| Enrollment fails | Not enough captured clips (need ≥1) or speaker name mismatch | Check captures exist for that speaker; names are case-sensitive |
| Voiceprint match shows 0% | ECAPA model download failed on Railway first run | Check `/data/models/ecapa/` exists on the volume; retry enrollment |
| Capture files missing | Volume not mounted at `/data` | Add Railway volume at `/data` |

---

## 10. Updating the model

1. Retrain locally: `python src/train_xlsr.py --manifest data/corpus_meet.csv --out models/sonave_xlsr_meet`
2. **Run the regression gate before deploying** — a retrain that quietly drops catch rate
   must not ship:
   ```bash
   python src/eval_xlsr.py --model models/sonave_xlsr_meet
   python -m pytest -m gpu tests/test_model_regression.py
   ```
   The gate compares against `results/benchmark_baseline.json` (±2 pts). If a drop is a
   deliberate tradeoff, update the baseline in the same commit and say why.
3. Update `modal_app.py` model path if the directory name changed
4. Commit the new `models/sonave_xlsr_meet/` checkpoint (it is version-controlled so CI
   can build the Modal image) and push — CI deploys Modal automatically
5. The Railway capture service needs no changes

---

## 11. Auto-deploy behavior

The repo includes `.github/workflows/ci-cd.yml`. On every **Pull Request**, it runs the fast test suite. On every **push to `main`**, it:

1. Runs tests
2. If tests pass → auto-deploys Modal
3. Verifies the live Modal `/openapi.json` version matches the repo — a failed
   verification fails the workflow, so a stale deploy can't go unnoticed
4. Railway auto-deploys independently via its GitHub integration

You can monitor deploys in **GitHub → Actions**.

---

## 12. Costs (rough)

| Component | Cost |
|-----------|------|
| Railway (CPU, always-on) | ~$5–10/mo |
| Modal GPU (T4, scale-to-zero) | ~$0.50–2/hr active, $0 idle |
| Recall.ai bot | ~$0.05–0.10/min of meeting |

A 1-hour Meet with continuous scoring ≈ **$1–3 total** vs. ~$400/hr for always-on GPU.
