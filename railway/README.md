# Sonave Capture Service — Railway deploy

A tiny, GPU-free service whose only job is to **collect real Meet-piped audio** for
training. Drop the bot into a meeting → it saves each participant's audio. Scoring
and retraining stay on your GPU box (download the WAVs and train locally).

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
4. **Add a Volume** (Settings → Volumes) mounted at `/data` so captures survive
   redeploys.
5. Railway gives you a public domain (e.g. `sonave.up.railway.app`). It sets
   `RAILWAY_PUBLIC_DOMAIN` automatically — the service uses it to tell Recall where to
   stream (`wss://<domain>/api/ws/audio`). No tunnel needed, ever.

## Use

- Open `https://<your-domain>/` → paste a Meet/Zoom link → **Send bot**.
- Talk / run the meeting. Each 2-min chunk of every speaker's audio is saved as it
  flushes (survives the bot leaving).
- The page shows **live stream quality** per speaker, a **live authenticity badge**
  (REAL / SUSPECT / FAKE — pushed up by `../tools/verdict_monitor.py` scoring on your
  GPU), and **captures grouped by session** with inline play + download links.
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

## Live authenticity verdict on the page
`../tools/verdict_monitor.py <url>` polls this service for new chunks, scores each on
your local GPU with `models/sonave_xlsr_meet`, and `POST`s the verdict to
`/api/verdict` so the page badge reads REAL / SUSPECT / FAKE in ~2-min steps. No
tunnel needed — it reads the chunks the capture service already saved.

## Local test
```
cd railway
pip install -r requirements.txt
SONAVE_PUBLIC_DOMAIN=localhost:8000 uvicorn app:app --port 8000
# open http://localhost:8000
```
