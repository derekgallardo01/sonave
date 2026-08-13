# Sonave Capture Service — Railway deploy

A tiny, GPU-free service that **sends the bot into meetings, captures per-speaker
audio, and drives the live console**. Scoring is hosted: each speaker's rolling
window is POSTed to the Modal GPU scorer (`SONAVE_SCORER_URL`) and the verdict
lands on the console in seconds. Captured WAVs also feed retraining (pull them to
your GPU box with `../src/pull_captures.py`).

Why separate from the main repo: no torch / no model → the image builds in seconds
and runs on Railway's CPU boxes. It reproduces the *real* Meet processing (via a real
bot in a real call), which is the ONLY validated way to get this data — offline
simulation was proven to fail (see `../results/detector_v2_progress.md`).

## Deploy (Railway)

1. **New Project → Deploy from GitHub repo** (your `sonave` repo).
2. In the service **Settings → Root Directory**, set: `railway`
   (so Railway builds only this folder, not the heavy detector repo).
3. **Variables** (Settings → Variables):
   - `SONAVE_RECALL_API_KEY` = your Recall key
   - `SONAVE_RECALL_BASE` = `https://us-west-2.recall.ai/api/v1` (match your region)
   - `SONAVE_DATA_DIR` = `/data/captured`
   - `SONAVE_SCORER_URL` *(optional)* = a hosted detector, e.g. the Modal deploy
     (`https://<you>--sonave-detector-fastapi-app.modal.run`, see `../modal_app.py`).
     When set, Railway scores each flushed chunk on that GPU service in a background
     thread and shows the verdict on the page — **no local process / laptop needed**.
     Unset = capture-only (page shows "verdict pending").
   - `SONAVE_API_TOKEN` *(optional but recommended)* = a shared access token. **Unset =
     the service is open** (anyone with the URL can send bots on your Recall dime and
     download recorded audio). **Set it** and every sensitive endpoint requires it: the
     page prompts for the token (stored in a cookie), the Recall bot's WebSocket carries
     it (`?token=`), and `verdict_monitor`/Railway→Modal send it as a header. Use the
     **same** value on Modal (`../modal_app.py`) and in the shell running
     `verdict_monitor`. Generate one: `python -c "import secrets;print(secrets.token_urlsafe(32))"`.
4. **Add a Volume** (Settings → Volumes) mounted at `/data` so captures survive
   redeploys.
5. Railway gives you a public domain (e.g. `sonave.up.railway.app`). It sets
   `RAILWAY_PUBLIC_DOMAIN` automatically — the service uses it to tell Recall where to
   stream (`wss://<domain>/api/ws/audio`). No tunnel needed, ever.

## Use

- `https://<your-domain>/` is the public marketing landing page; the operator console
  lives at `https://<your-domain>/console` (prompts for `SONAVE_API_TOKEN`).
- In the console: paste a Meet/Zoom link → **Deploy** (this also pre-warms the
  scale-to-zero scorer).
- Talk / run the meeting. Each 2-min chunk of every speaker's audio is saved as it
  flushes (survives the bot leaving).
- The page shows **live stream quality** per speaker, a **live authenticity badge**
  (REAL / SUSPECT / FAKE — scored either hands-free via `SONAVE_SCORER_URL` (Railway →
  Modal) or by `../tools/verdict_monitor.py` on your local GPU), and **captures grouped
  by session** with inline play + download links.
- Pull captures to your GPU box (`../src/pull_captures.py`), fold in
  (`../src/add_captured.py`), and retrain.

## Collecting good data (the proven VB-CABLE workflow)
The reliable way to feed known-label audio through a *real* Meet — validated in
Stage 6 (`../results/detector_v2_progress.md`). Playing through speakers does **not**
work (the mic never picks it up at usable volume); a virtual cable is the unlock.

1. Install **VB-CABLE** (adds `CABLE Input` = a virtual speaker, `CABLE Output` = a
   virtual mic).
2. In your Meet tab: **Settings → Audio → Microphone → `CABLE Output`**, and un-mute.
3. Play the audio into the cable, full-volume and digital:
   `python ../tools/play_into_meet.py <folder> --shuffle --loop --device "CABLE Input"`
4. Send the bot; watch the page level go **GOOD**. It captures at full quality.

- **Real session:** play real human speech (e.g. LibriSpeech) → pull with
  `pull_captures.py <url>` → `data/captured/` (label real).
- **Fake session:** play AI-generated speech → pull with `pull_captures.py <url> --fake`
  → `data/captured_fake/` (label fake). The pull tool never double-labels a clip.
- **Balance matters:** collect comparable amounts of BOTH — real-only teaches "Meet =
  real" (goes blind to fakes), fake-only teaches "Meet = fake" (false-alarms on real).
- **Always** hold some captured audio out of training to validate — the ground truth.
- **Consent:** announce recording; required for the finance vertical.

## Fraud alerts, wire-hold & incidents
When a speaker's rolling verdict stays **fake** for `SONAVE_INCIDENT_STREAK` consecutive
scoring windows (default 3 — ~30 s at the default cadence), the service opens an **incident**
(persisted to SQLite next to the `/data` volume, so it survives redeploys), flags a
**wire-hold**, and — if `SONAVE_ALERT_WEBHOOK` is set — posts a Slack-formatted alert. The
page shows a red **⛔ WIRE HELD** banner with an **Acknowledge** button (which clears the hold
and closes the incident). This layer is torch-free (it runs on the verdicts the service
already receives from the scorer); GPU-side voiceprint fusion + auto-generated forensic
reports are the next follow-ups. Endpoints: `GET /api/incidents`, `POST /api/incidents/ack`.

## Legacy fallback: local-GPU scoring
The hosted path (`SONAVE_SCORER_URL` → Modal) is the primary way verdicts reach the
console. If you need to score without Modal, `../tools/verdict_monitor.py <url>` polls
this service for new capture chunks, scores them on your local GPU with
`models/sonave_xlsr_meet`, and `POST`s verdicts to `/api/verdict` — but only in
~2-minute steps (it works off the capture flush), vs seconds for the hosted path.

## Local test
```
cd railway
pip install -r requirements.txt
SONAVE_PUBLIC_DOMAIN=localhost:8000 uvicorn app:app --port 8000
# open http://localhost:8000
```
