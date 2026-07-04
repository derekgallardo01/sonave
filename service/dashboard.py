"""
dashboard.py — Sonave live meeting monitor (Phase 5 surface).

A self-contained web dashboard over the Orchestrator: per-speaker RAG status, rolling
P(fake) meters, an alert feed, and a live cost meter (the "$0.02 vs $400/hr" story).
No external assets — inline HTML/CSS/JS, polls /api/status.

Run (in .venv, after `pip install -r service/requirements.txt`):
    .venv/Scripts/python.exe -m uvicorn service.dashboard:app --port 8000
    # open http://localhost:8000
    # click "Run demo" (replays a stitched meeting with a fake stretch), or POST
    # /api/ingest live from the capture layer.

Live wiring: the Recall.ai adapter POSTs /api/ingest {speaker_id, audio_b64, ts};
the board reflects Orchestrator.status() in real time.
"""
from __future__ import annotations

import base64
import sys
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# make service/ + src/ + repo-root importable whether run as a script or as the
# `service.dashboard` module under uvicorn.
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "src", _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import detector
import model_sls
from orchestrator import Orchestrator

app = FastAPI(title="Sonave Live Monitor")

# --- shared live state -------------------------------------------------------
STATE = {"meeting": None, "started": None, "alerts": [], "inferences": 0}
GPU_COST_PER_HR = 1.50          # rough cloud GPU $/hr, for the cost meter
GPU_SEC_PER_INFER = 0.1         # ~100 ms per 4 s window


def _on_alert(event):
    STATE["alerts"].append({**event, "type": "alert"})


def _on_hold(event):
    STATE["alerts"].append({**event, "type": "wire_hold"})


ORCH = Orchestrator(on_alert=_on_alert, on_hold=_on_hold)


def _count_ingest(fn):
    def wrapped(*a, **k):
        STATE["inferences"] += 1
        return fn(*a, **k)
    return wrapped


ORCH.ingest = _count_ingest(ORCH.ingest)   # meter every inference


# --- API ---------------------------------------------------------------------
class Ingest(BaseModel):
    speaker_id: str
    audio_b64: str
    ts: float


@app.post("/api/ingest")
def ingest(body: Ingest):
    import io
    import soundfile as sf
    wav, sr = sf.read(io.BytesIO(base64.b64decode(body.audio_b64)))
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    return ORCH.ingest(body.speaker_id, np.asarray(wav, dtype="float32"), body.ts)


@app.post("/api/recall/audio")
async def recall_audio(event: dict):
    """LIVE real-time audio (WebSocket/realtime_endpoints on the bot) → orchestrator."""
    import recall_adapter
    if STATE["started"] is None:
        STATE["started"] = time.time()
        STATE["meeting"] = event.get("data", {}).get("meeting_url", "live meeting")
    res = recall_adapter.handle_audio_event(event, ORCH)
    return res or {"status": "ignored"}


# Async Svix webhooks (audio_separate.done, etc.). We CAPTURE the raw payload so the
# exact schema can be inspected, then hand it to the download+analyze handler.
RECALL_EVENTS: list = []


@app.post("/api/recall/webhook")
async def recall_webhook(event: dict):
    RECALL_EVENTS.append(event)
    del RECALL_EVENTS[:-25]                     # keep last 25
    import recall_adapter
    try:
        res = recall_adapter.handle_async_event(event, ORCH, STATE)
    except Exception as e:  # never NACK a webhook on our error
        res = {"status": "captured", "note": f"handler err: {repr(e)[:120]}"}
    return res or {"status": "captured"}


@app.get("/api/recall/events")
def recall_events():
    """Inspect the raw captured webhook payloads (to confirm the schema)."""
    return {"count": len(RECALL_EVENTS), "events": RECALL_EVENTS}


# Per-speaker raw-audio capture (for collecting real Meet-piped training data).
CAPTURE = {"on": True, "buffers": {}}
CAPTURE_DIR = Path(__file__).resolve().parent.parent / "data" / "captured"


@app.websocket("/api/ws/audio")
async def ws_audio(ws: WebSocket):
    """LIVE real-time audio: Recall's bot connects here and streams
    audio_separate_raw.data events (JSON). Each is scored AND (optionally) saved."""
    import json
    import recall_adapter
    await ws.accept()
    if STATE["started"] is None:
        STATE["started"] = time.time()
        STATE["meeting"] = "live meeting"
    try:
        while True:
            msg = await ws.receive_text()
            try:
                event = json.loads(msg)
                recall_adapter.handle_audio_event(event, ORCH)
                if CAPTURE["on"]:
                    _capture(event)
            except Exception:
                pass          # never drop the socket on a single bad frame
    except WebSocketDisconnect:
        pass
    finally:
        _flush_capture()      # write collected audio to disk on disconnect


def _capture(event: dict):
    d = (event.get("data") or {}).get("data") or {}
    buf = d.get("buffer")
    if not buf:
        return
    spk = ((d.get("participant") or {}).get("name") or "unknown").replace(" ", "_")
    pcm = np.frombuffer(base64.b64decode(buf), dtype=np.int16)
    CAPTURE["buffers"].setdefault(spk, []).append(pcm)


def _flush_capture():
    import soundfile as sf
    if not CAPTURE["buffers"]:
        return
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = int(STATE.get("started") or 0)
    for spk, chunks in CAPTURE["buffers"].items():
        pcm = np.concatenate(chunks).astype(np.float32) / 32768.0
        if len(pcm) < model_sls.SR:      # skip <1s
            continue
        out = CAPTURE_DIR / f"meet_{spk}_{stamp}.wav"
        sf.write(str(out), pcm, model_sls.SR)
        print(f"[capture] saved {len(pcm)/model_sls.SR:.1f}s of '{spk}' -> {out}")
    CAPTURE["buffers"].clear()


class EnrollReq(BaseModel):
    speaker_id: str


@app.post("/api/enroll")
def api_enroll(body: EnrollReq):
    """Enroll a voiceprint for a speaker from their captured Meet audio."""
    import glob
    import enroll
    key = body.speaker_id
    files = [p for p in glob.glob(str(CAPTURE_DIR / "*.wav"))
             if key.replace(" ", "_") in Path(p).name]
    if not files:
        return {"ok": False, "error": f"no captured audio matching '{key}' in {CAPTURE_DIR}"}
    enroll.enroll(key, files)
    return {"ok": True, "enrolled": enroll.list_enrolled(), "from": len(files)}


@app.get("/api/enrolled")
def api_enrolled():
    import enroll
    return {"enrolled": enroll.list_enrolled()}


@app.get("/api/status")
def status():
    elapsed = (time.time() - STATE["started"]) if STATE["started"] else 0
    cost = STATE["inferences"] * GPU_SEC_PER_INFER / 3600 * GPU_COST_PER_HR
    return {"meeting": STATE["meeting"], "elapsed_s": round(elapsed, 1),
            "speakers": ORCH.status(), "alerts": STATE["alerts"][-20:],
            "inferences": STATE["inferences"], "cost_usd": round(cost, 4),
            "tau_real": detector.TAU_REAL, "tau_fake": detector.TAU_FAKE}


@app.post("/api/demo/start")
def demo_start():
    """Replay a stitched demo meeting (real → real → ElevenLabs fake → real)."""
    _reset()
    STATE["meeting"] = "demo_meeting.wav"
    STATE["started"] = time.time()
    threading.Thread(target=_run_demo, daemon=True).start()
    return {"status": "started"}


def _reset():
    ORCH.speakers.clear()
    STATE["alerts"].clear()
    STATE["inferences"] = 0


def _run_demo():
    """Build (if needed) and replay a demo meeting at ~real time."""
    import glob
    import librosa
    import soundfile as sf

    root = Path(__file__).resolve().parent.parent
    demo = root / "data" / "_demo_meeting.wav"
    reals = sorted(glob.glob(str(root / "data" / "real" / "itw_real_*.wav")))
    fakes = sorted(glob.glob(str(root / "data" / "corpus" / "mlaad" / "test" /
                             "ElevenLabs-v3" / "*.wav"))) or \
        sorted(glob.glob(str(root / "data" / "fake" / "itw" / "*.wav")))

    sr = model_sls.SR

    def clip(p, sec=8):
        w, _ = librosa.load(p, sr=sr, mono=True)
        return np.pad(w, (0, max(0, sec * sr - len(w))))[:sec * sr]

    turns = [("Alice", "real", reals[0]), ("Bob", "real", reals[1]),
             ("Bob", "fake", fakes[0]), ("Alice", "real", reals[2])]
    meeting = np.concatenate([clip(p) for _, _, p in turns])
    sf.write(str(demo), meeting.astype(np.float32), sr)

    # speaker turns (8 s each)
    segs = []
    for i, (spk, _, _) in enumerate(turns):
        segs.append((i * 8.0, (i + 1) * 8.0, spk))

    WIN = model_sls.MAX_LEN
    hop = 2.0
    hop_n = int(hop * sr)
    for s in range(0, max(1, len(meeting) - WIN + 1), hop_n):
        t = (s + WIN // 2) / sr
        spk = next((sp for a, b, sp in segs if a <= t < b), None)
        chunk = meeting[s:s + WIN]
        if spk and np.sqrt(np.mean(chunk ** 2)) >= 0.005:
            ORCH.ingest(spk, chunk, t)
        time.sleep(hop)          # pace at ~real time for a live feel


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>Sonave — Live Monitor</title>
<style>
:root{--bg:#0f1420;--card:#1a2130;--tx:#e8edf6;--mut:#8b97ad;--line:#2a3446}
@media(prefers-color-scheme:light){:root{--bg:#f4f6fb;--card:#fff;--tx:#141a26;--mut:#5a6678;--line:#e2e7f0}}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--tx)}
.wrap{max-width:1000px;margin:0 auto;padding:24px}
h1{font-size:20px;margin:0 0 2px}.sub{color:var(--mut);font-size:13px}
.bar{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:14px 0 20px}
.pill{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 14px}
.pill b{font-size:18px}.pill span{color:var(--mut);font-size:12px;display:block}
button{background:#2f6df6;color:#fff;border:0;border-radius:9px;padding:10px 16px;font-size:14px;cursor:pointer}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}
.spk{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;position:relative;overflow:hidden}
.spk .name{font-weight:600;font-size:16px}.spk .st{font-size:13px;font-weight:700;letter-spacing:.04em;text-transform:uppercase}
.meter{height:8px;background:var(--line);border-radius:6px;margin:12px 0 6px;overflow:hidden}
.meter i{display:block;height:100%;border-radius:6px;transition:width .4s}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:middle}
.real{color:#38c172}.suspect{color:#e8a020}.fake{color:#ef4a4a}
.bg-real{background:#38c172}.bg-suspect{background:#e8a020}.bg-fake{background:#ef4a4a}
.spk.fake{border-color:#ef4a4a;box-shadow:0 0 0 1px #ef4a4a55;animation:pulse 1.4s infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 0 1px #ef4a4a55}50%{box-shadow:0 0 14px 1px #ef4a4a88}}
.feed{margin-top:22px;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 18px}
.feed h3{margin:0 0 8px;font-size:14px;color:var(--mut)}
.ev{padding:7px 0;border-bottom:1px solid var(--line);font-size:14px}.ev:last-child{border:0}
.tag{font-size:11px;font-weight:700;padding:2px 7px;border-radius:5px;margin-right:8px}
.tag.alert{background:#ef4a4a22;color:#ef4a4a}.tag.wire_hold{background:#e8a02022;color:#e8a020}
.empty{color:var(--mut);font-size:14px}
</style></head><body><div class="wrap">
<h1>Sonave — Live Meeting Monitor</h1>
<div class="sub" id="mt">no meeting active</div>
<div class="bar">
  <button onclick="demo()">▶ Run demo meeting</button>
  <div class="pill"><b id="el">0s</b><span>elapsed</span></div>
  <div class="pill"><b id="inf">0</b><span>inferences</span></div>
  <div class="pill"><b id="cost">$0.0000</b><span>GPU cost (vs ~$400/hr API)</span></div>
</div>
<div class="grid" id="spk"><div class="empty">Speakers appear here once a meeting starts.</div></div>
<div class="feed"><h3>Alerts &amp; actions</h3><div id="feed"><div class="empty">No alerts.</div></div></div>
</div><script>
async function demo(){await fetch('/api/demo/start',{method:'POST'})}
async function enroll(spk){let r=await fetch('/api/enroll',{method:'POST',
 headers:{'Content-Type':'application/json'},body:JSON.stringify({speaker_id:spk})});
 let d=await r.json();alert(d.ok?('enrolled '+spk+' from '+d.from+' clips'):('enroll failed: '+d.error))}
function cls(v){return v}
async function tick(){
 let r=await fetch('/api/status'),d=await r.json();
 document.getElementById('mt').textContent=d.meeting?('meeting: '+d.meeting):'no meeting active';
 document.getElementById('el').textContent=(d.elapsed_s||0).toFixed(0)+'s';
 document.getElementById('inf').textContent=d.inferences;
 document.getElementById('cost').textContent='$'+(d.cost_usd||0).toFixed(4);
 let g=document.getElementById('spk'),ks=Object.keys(d.speakers||{});
 if(!ks.length){g.innerHTML='<div class="empty">Speakers appear here once a meeting starts.</div>'}
 else{g.innerHTML=ks.map(k=>{let s=d.speakers[k],p=Math.round(s.rolling_p_fake*100);
  return `<div class="spk ${s.verdict}"><div class="name"><span class="dot bg-${s.verdict}"></span>${k}</div>
  <div class="st ${s.verdict}">${s.verdict}</div>
  <div class="meter"><i class="bg-${s.verdict}" style="width:${p}%"></i></div>
  <div class="sub">P(fake) ${s.rolling_p_fake} · peak ${s.peak} · ${s.windows} win${s.enrolled?` · <b>voiceprint ${s.voiceprint_sim==null?'…':s.voiceprint_sim}</b>`:` · <a href="#" onclick="enroll('${k}');return false">enroll</a>`}</div></div>`}).join('')}
 let f=document.getElementById('feed'),al=(d.alerts||[]).slice().reverse();
 if(!al.length){f.innerHTML='<div class="empty">No alerts.</div>'}
 else{f.innerHTML=al.map(a=>`<div class="ev"><span class="tag ${a.type}">${a.type=='wire_hold'?'WIRE HOLD':'ALERT'}</span>
  Speaker <b>${a.speaker_id}</b> ${a.type=='wire_hold'?'→ transaction held / re-auth required':'flagged FAKE'}
  at ${(a.ts||0).toFixed(1)}s (P(fake) ${a.rolling_p_fake})</div>`).join('')}
}
setInterval(tick,1000);tick();
</script></body></html>
"""
