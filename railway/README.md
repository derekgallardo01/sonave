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
- Talk / run the meeting. When the bot leaves, each speaker's audio is saved.
- The page lists captures with **download** links. Pull them to your GPU box, add as
  `label=real` (see `../src/add_captured.py`), and retrain.

## Collecting good data
- **Real voices:** put the bot in your normal meetings (with consent) — diverse
  speakers/mics/rooms, all correctly "real".
- **Fakes in the Meet domain:** play your fake clips into a meeting via a virtual
  audio cable (VB-CABLE) so they go through real Meet, and label them fake.
- **Always** hold some captured audio out of training to validate — the ground truth.
- **Consent:** announce recording; required for the finance vertical.

## Local test
```
cd railway
pip install -r requirements.txt
SONAVE_PUBLIC_DOMAIN=localhost:8000 uvicorn app:app --port 8000
# open http://localhost:8000
```
