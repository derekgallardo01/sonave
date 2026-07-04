"""
Sonave capture service — deploys to Railway (CPU-only, no model needed).

Its ONE job: let you drop the Sonave bot into any meeting and save the real
Meet-piped audio for training. Scoring/retraining happens offline on your GPU box;
this just collects ground-truth domain data at scale.

Dependency-light on purpose: FastAPI + stdlib `wave` (no torch / numpy / soundfile),
so the Railway image is tiny and builds in seconds.

Endpoints:
  GET  /                 dashboard: send a bot, list/download captures
  POST /bot              {meeting_url} -> Recall bot streams audio here
  WS   /api/ws/audio     Recall real-time audio -> saved per speaker on disconnect
  GET  /captures         list saved files (JSON)
  GET  /download/{name}  download a capture
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
import wave
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# --- config (Railway env vars) ----------------------------------------------
RECALL_API_KEY = os.environ.get("SONAVE_RECALL_API_KEY")
RECALL_BASE = os.environ.get("SONAVE_RECALL_BASE", "https://us-west-2.recall.ai/api/v1")
# Railway sets RAILWAY_PUBLIC_DOMAIN automatically (e.g. sonave.up.railway.app).
PUBLIC_DOMAIN = os.environ.get("SONAVE_PUBLIC_DOMAIN") or os.environ.get("RAILWAY_PUBLIC_DOMAIN")
DATA_DIR = Path(os.environ.get("SONAVE_DATA_DIR", "/data/captured"))
SR = 16_000

app = FastAPI(title="Sonave Capture")


def _ws_url() -> str:
    if not PUBLIC_DOMAIN:
        raise RuntimeError("No public domain set (RAILWAY_PUBLIC_DOMAIN / SONAVE_PUBLIC_DOMAIN).")
    return f"wss://{PUBLIC_DOMAIN}/api/ws/audio"


# --- send a bot to a meeting -------------------------------------------------
class BotReq(BaseModel):
    meeting_url: str
    bot_name: str = "Sonave"


@app.post("/bot")
def send_bot(req: BotReq):
    if not RECALL_API_KEY:
        return {"error": "SONAVE_RECALL_API_KEY not set on the service"}
    payload = {
        "meeting_url": req.meeting_url,
        "bot_name": req.bot_name,
        "recording_config": {
            "audio_separate_raw": {},
            "realtime_endpoints": [
                {"type": "websocket", "url": _ws_url(),
                 "events": ["audio_separate_raw.data"]}
            ],
        },
    }
    r = urllib.request.Request(f"{RECALL_BASE}/bot", data=json.dumps(payload).encode(),
                               headers={"Authorization": f"Token {RECALL_API_KEY}",
                                        "Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(r, timeout=20).read())
        return {"ok": True, "bot_id": resp.get("id"), "ws": _ws_url()}
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code, "detail": e.read().decode()[:300]}


# --- real-time audio capture -------------------------------------------------
@app.websocket("/api/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    buffers: dict[str, bytearray] = {}
    session = int(time.time())
    try:
        while True:
            msg = await ws.receive_text()
            try:
                d = (json.loads(msg).get("data") or {}).get("data") or {}
                buf = d.get("buffer")
                if not buf:
                    continue
                spk = ((d.get("participant") or {}).get("name") or "unknown").replace(" ", "_")
                buffers.setdefault(spk, bytearray()).extend(base64.b64decode(buf))
            except Exception:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        _save(buffers, session)


def _save(buffers: dict[str, bytearray], session: int):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for spk, pcm in buffers.items():
        if len(pcm) < SR * 2:            # <1 s of 16-bit audio -> skip
            continue
        out = DATA_DIR / f"meet_{spk}_{session}.wav"
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)            # 16-bit PCM (S16LE, matches Recall)
            w.setframerate(SR)
            w.writeframes(bytes(pcm))
        print(f"[capture] saved {len(pcm)/2/SR:.1f}s of '{spk}' -> {out}", flush=True)


# --- retrieval ---------------------------------------------------------------
@app.get("/captures")
def captures():
    if not DATA_DIR.exists():
        return {"files": []}
    fs = sorted(DATA_DIR.glob("*.wav"))
    return {"files": [{"name": f.name, "mb": round(f.stat().st_size / 1e6, 2)} for f in fs]}


@app.get("/download/{name}")
def download(name: str):
    f = DATA_DIR / Path(name).name          # prevent path traversal
    return FileResponse(str(f)) if f.exists() else {"error": "not found"}


@app.get("/", response_class=HTMLResponse)
def index():
    domain = PUBLIC_DOMAIN or "(set RAILWAY_PUBLIC_DOMAIN)"
    key = "set" if RECALL_API_KEY else "MISSING — set SONAVE_RECALL_API_KEY"
    return f"""<!doctype html><meta charset=utf-8><title>Sonave Capture</title>
<style>body{{font:15px system-ui;max-width:720px;margin:40px auto;padding:0 16px;
background:#0f1420;color:#e8edf6}}input,button{{font:15px system-ui;padding:9px 12px;
border-radius:8px;border:1px solid #2a3446;background:#1a2130;color:#e8edf6}}
button{{background:#2f6df6;border:0;cursor:pointer}}a{{color:#5aa0ff}}
.row{{display:flex;gap:8px;margin:14px 0}}code{{color:#8b97ad}}</style>
<h2>Sonave — Meeting Audio Capture</h2>
<p>domain <code>{domain}</code> · Recall key <code>{key}</code></p>
<div class=row><input id=u placeholder="paste a Google Meet / Zoom link" style=flex:1>
<button onclick=send()>Send bot</button></div>
<p id=msg></p><h3>Captures</h3><div id=list>loading…</div>
<script>
async function send(){{let u=document.getElementById('u').value;
let r=await fetch('/bot',{{method:'POST',headers:{{'Content-Type':'application/json'}},
body:JSON.stringify({{meeting_url:u}})}});let d=await r.json();
document.getElementById('msg').textContent=d.ok?('bot sent: '+d.bot_id):('error: '+(d.detail||d.error||d.status));list()}}
async function list(){{let d=await(await fetch('/captures')).json();
document.getElementById('list').innerHTML=d.files.length?d.files.map(f=>
`<div>· <a href="/download/${{f.name}}">${{f.name}}</a> (${{f.mb}} MB)</div>`).join(''):'none yet';}}
list();setInterval(list,5000);
</script>"""
