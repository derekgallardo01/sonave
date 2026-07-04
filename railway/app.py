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

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# --- config (Railway env vars) ----------------------------------------------
RECALL_API_KEY = os.environ.get("SONAVE_RECALL_API_KEY")
RECALL_BASE = os.environ.get("SONAVE_RECALL_BASE", "https://us-west-2.recall.ai/api/v1")
DATA_DIR = Path(os.environ.get("SONAVE_DATA_DIR", "/data/captured"))
SR = 16_000

app = FastAPI(title="Sonave Capture")

# Inline favicon: white audio bars on the brand-blue rounded square (no file needed).
_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="7" fill="#2f6df6"/>'
    '<g fill="#fff">'
    '<rect x="6" y="13" width="3" height="6" rx="1.5"/>'
    '<rect x="11" y="9" width="3" height="14" rx="1.5"/>'
    '<rect x="16" y="5" width="3" height="22" rx="1.5"/>'
    '<rect x="21" y="10" width="3" height="12" rx="1.5"/>'
    '<rect x="26" y="14" width="3" height="4" rx="1.5"/>'
    '</g></svg>'
)
_FAVICON_B64 = base64.b64encode(_FAVICON_SVG.encode()).decode()


def _domain(request: Request | None = None) -> str:
    """Public hostname. Prefer an explicit env override, else the actual request
    Host header (works on Railway with zero config), else Railway's auto var."""
    env = os.environ.get("SONAVE_PUBLIC_DOMAIN") or os.environ.get("RAILWAY_PUBLIC_DOMAIN")
    if env:
        return env
    if request is not None:
        host = request.headers.get("host")
        if host:
            return host
    return ""


def _ws_url(request: Request) -> str:
    d = _domain(request)
    if not d:
        raise RuntimeError("Could not determine public domain from request.")
    return f"wss://{d}/api/ws/audio"


# --- send a bot to a meeting -------------------------------------------------
class BotReq(BaseModel):
    meeting_url: str
    bot_name: str = "Sonave"


@app.post("/bot")
def send_bot(req: BotReq, request: Request):
    if not RECALL_API_KEY:
        return {"error": "SONAVE_RECALL_API_KEY not set on the service"}
    ws = _ws_url(request)
    payload = {
        "meeting_url": req.meeting_url,
        "bot_name": req.bot_name,
        "recording_config": {
            "audio_separate_raw": {},
            "realtime_endpoints": [
                {"type": "websocket", "url": ws, "events": ["audio_separate_raw.data"]}
            ],
        },
    }
    r = urllib.request.Request(f"{RECALL_BASE}/bot", data=json.dumps(payload).encode(),
                               headers={"Authorization": f"Token {RECALL_API_KEY}",
                                        "Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(r, timeout=20).read())
        return {"ok": True, "bot_id": resp.get("id"), "ws": ws}
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code, "detail": e.read().decode()[:300]}


# --- real-time audio capture -------------------------------------------------
CHUNK_SEC = 120          # flush each speaker's audio every ~2 min (all-day safe)
_CHUNK_BYTES = CHUNK_SEC * SR * 2

# --- live stream-quality monitoring -----------------------------------------
QUALITY: dict[str, dict] = {}
# authenticity verdicts pushed up from the local GPU scorer (tools/verdict_monitor.py)
VERDICTS: dict[str, dict] = {}


def _quality(spk: str, pcm: bytes):
    """Update rolling audio-quality stats for a speaker from a raw PCM16 chunk.
    Uses stdlib array/math (audioop was removed in Python 3.13)."""
    import array
    import math
    s = array.array("h")
    s.frombytes(pcm if len(pcm) % 2 == 0 else pcm[:-1])
    n = len(s)
    if n == 0:
        return
    peak = max(abs(min(s)), abs(max(s))) / 32768.0
    step = max(1, n // 2000)                       # subsample for cheap RMS
    ss = sum(s[i] * s[i] for i in range(0, n, step))
    rms = math.sqrt(ss / (n // step + 1)) / 32768.0
    sec = n / SR
    q = QUALITY.setdefault(spk, {"level": 0.0, "peak": 0.0, "clips": 0,
                                 "speech_sec": 0.0, "total_sec": 0.0})
    q["level"] = 0.25 * rms + 0.75 * q["level"]       # smoothed current level
    q["peak"] = max(peak, q["peak"] * 0.99)            # decaying peak-hold
    if peak >= 0.99:
        q["clips"] += 1
    if rms > 0.01:
        q["speech_sec"] += sec
    q["total_sec"] += sec


def _quality_verdict(q: dict) -> str:
    if q["total_sec"] < 3:
        return "warming up"
    if q["peak"] >= 0.985 or q["clips"] > 5:
        return "CLIPPING — lower volume"
    if q["level"] < 0.01:
        return "TOO QUIET — raise volume"
    speech = q["speech_sec"] / max(q["total_sec"], 1e-6)
    if speech < 0.2:
        return "mostly silence — is audio playing?"
    return "good"


@app.websocket("/api/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    buffers: dict[str, bytearray] = {}
    idx: dict[str, int] = {}
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
                raw = base64.b64decode(buf)
                b = buffers.setdefault(spk, bytearray())     # CAPTURE FIRST (critical path)
                b.extend(raw)
                if len(b) >= _CHUNK_BYTES:               # periodic flush -> ~2 min files
                    _write(spk, bytes(b), session, idx.get(spk, 0))
                    idx[spk] = idx.get(spk, 0) + 1
                    b.clear()
                try:
                    _quality(spk, raw)                   # quality is best-effort, never breaks capture
                except Exception:
                    pass
            except Exception:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        for spk, b in buffers.items():                   # flush the remainder
            if len(b) >= SR * 2:
                _write(spk, bytes(b), session, idx.get(spk, 0))


def _write(spk: str, pcm: bytes, session: int, idx: int):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"meet_{spk}_{session}_{idx:03d}.wav"
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)            # 16-bit PCM (S16LE, matches Recall)
        w.setframerate(SR)
        w.writeframes(pcm)
    print(f"[capture] saved {len(pcm)/2/SR:.1f}s of '{spk}' -> {out}", flush=True)


# --- retrieval ---------------------------------------------------------------
@app.get("/favicon.ico")
@app.get("/favicon.svg")
def favicon():
    from fastapi.responses import Response
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml")


class VerdictReq(BaseModel):
    speaker: str
    p_fake: float
    rolling: float
    verdict: str


@app.post("/api/verdict")
def api_verdict(v: VerdictReq):
    """Local GPU scorer pushes authenticity verdicts here; the page shows them."""
    VERDICTS[v.speaker] = {"p_fake": round(v.p_fake, 3), "rolling": round(v.rolling, 3),
                           "verdict": v.verdict}
    return {"ok": True}


SKIP_SPEAKERS = ("HealthCheck", "FIXCHECK", "WSTEST", "deploycheck")


@app.get("/api/quality")
def api_quality():
    out = {}
    speakers = set(QUALITY) | set(VERDICTS)
    for spk in speakers:
        if any(s in spk for s in SKIP_SPEAKERS):
            continue
        q = QUALITY.get(spk)
        row = {"verdict": _quality_verdict(q) if q else "—"}
        if q:
            speech = q["speech_sec"] / max(q["total_sec"], 1e-6)
            row.update({"level": round(q["level"], 3), "peak": round(q["peak"], 3),
                        "clips": q["clips"], "speech_pct": round(speech * 100),
                        "total_sec": round(q["total_sec"])})
        av = VERDICTS.get(spk)
        if av:
            row["auth_verdict"] = av["verdict"]
            row["auth_p"] = av["rolling"]
        out[spk] = row
    return out


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
def index(request: Request):
    domain = _domain(request) or "(unknown — no Host header)"
    key = "set" if RECALL_API_KEY else "MISSING — set SONAVE_RECALL_API_KEY"
    return (_PAGE
            .replace("__DOMAIN__", domain)
            .replace("__KEY__", key)
            .replace("__FAVICON__", _FAVICON_B64))


_PAGE = r"""<!doctype html><html lang=en><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Sonave — Live Voice Authenticity</title>
<link rel="icon" href="data:image/svg+xml;base64,__FAVICON__">
<style>
:root{--bg:#0d111c;--card:#161c2b;--card2:#1b2233;--line:#28324a;--ink:#e9eef8;
--mut:#8b97ad;--dim:#616d84;--blue:#2f6df6;--green:#33c17f;--amber:#eaa021;--red:#ef4a4a}
*{box-sizing:border-box}
body{font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:840px;
margin:0 auto;padding:34px 18px 80px;background:var(--bg);color:var(--ink)}
a{color:#6aa5ff;text-decoration:none}a:hover{text-decoration:underline}
h1{font-size:21px;margin:0;letter-spacing:-.02em;display:flex;align-items:center;gap:9px}
.mark{width:22px;height:22px;border-radius:6px;background:linear-gradient(135deg,#2f6df6,#7a4dff);
display:inline-block;flex:none;box-shadow:0 2px 10px rgba(47,109,246,.4)}
.sub{color:var(--mut);font-size:13px;margin:5px 0 0}
.meta{color:var(--dim);font-size:12px;margin:3px 0 0}.meta code{color:var(--mut)}
.top{display:flex;justify-content:space-between;align-items:flex-start;gap:14px;flex-wrap:wrap}
.live{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--mut);
background:var(--card);border:1px solid var(--line);padding:6px 11px;border-radius:999px;white-space:nowrap}
.dot{width:8px;height:8px;border-radius:50%;background:var(--dim)}
.dot.on{background:var(--green);box-shadow:0 0 0 0 rgba(51,193,127,.6);animation:pulse 1.8s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(51,193,127,.5)}70%{box-shadow:0 0 0 7px rgba(51,193,127,0)}100%{box-shadow:0 0 0 0 rgba(51,193,127,0)}}
.row{display:flex;gap:8px;margin:20px 0 6px}
input,button{font:15px system-ui;padding:11px 13px;border-radius:9px;border:1px solid var(--line);
background:var(--card);color:var(--ink);outline:none}
input:focus{border-color:var(--blue)}
button{background:var(--blue);border:0;cursor:pointer;font-weight:600}
button:hover{filter:brightness(1.08)}button:disabled{opacity:.6;cursor:default}
.msg{font-size:13px;color:var(--mut);min-height:18px;margin:2px 0 0}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.09em;color:var(--dim);
margin:30px 0 4px;font-weight:700}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--mut);margin:0 0 12px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.pip{width:9px;height:9px;border-radius:3px;display:inline-block}
.card{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:15px 17px;margin:10px 0}
.chead{display:flex;justify-content:space-between;align-items:center;gap:12px}
.who{font-weight:700;font-size:16px}
.chip{color:#fff;padding:5px 12px;border-radius:8px;font-weight:800;font-size:13px;letter-spacing:.02em;white-space:nowrap}
.chip small{font-weight:600;opacity:.85;font-size:11px;margin-left:5px}
.pend{color:var(--dim);font-size:12px;font-style:italic}
.bar{height:7px;background:#0f1524;border-radius:5px;margin:11px 0 8px;overflow:hidden}
.bar i{display:block;height:100%;transition:width .3s}
.det{color:var(--mut);font-size:12.5px}.det b{color:var(--ink);font-weight:600}
.empty{color:var(--dim);font-size:13px;padding:18px 0}
.sess{border:1px solid var(--line);border-radius:11px;margin:9px 0;overflow:hidden;background:var(--card)}
.shead{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:11px 14px;
cursor:pointer;user-select:none}.shead:hover{background:var(--card2)}
.stitle{font-weight:600;font-size:14px}.smeta{color:var(--dim);font-size:12px}
.sbody{border-top:1px solid var(--line)}
.clip{display:flex;align-items:center;gap:10px;padding:8px 14px;font-size:13px;border-top:1px solid #202839}
.clip:first-child{border-top:0}
.play{width:28px;height:28px;flex:none;border-radius:50%;background:var(--card2);border:1px solid var(--line);
color:var(--ink);cursor:pointer;font-size:11px;padding:0;line-height:1}
.play:hover{background:var(--blue);border-color:var(--blue)}
.cname{color:var(--mut);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cdur{color:var(--dim);font-size:12px;font-variant-numeric:tabular-nums}
.caret{color:var(--dim);transition:transform .15s}.caret.open{transform:rotate(90deg)}
</style>
<div class=top>
 <div><h1><span class=mark></span>Sonave</h1>
  <p class=sub>Live deepfake-voice monitoring for meetings</p>
  <p class=meta>domain <code>__DOMAIN__</code> · Recall key <code>__KEY__</code></p></div>
 <div class=live><span id=dot class=dot></span><span id=livetxt>idle</span></div>
</div>
<div class=row><input id=u placeholder="paste a Google Meet / Zoom link" style=flex:1>
<button id=sendb onclick=send()>Send bot</button></div>
<p class=msg id=msg></p>

<h2>Live authenticity</h2>
<div class=legend>
 <span><i class=pip style=background:var(--green)></i>REAL &lt;0.40</span>
 <span><i class=pip style=background:var(--amber)></i>SUSPECT 0.40–0.70</span>
 <span><i class=pip style=background:var(--red)></i>FAKE ≥0.70</span>
</div>
<div id=quality><div class=empty>Waiting for audio… send a bot into a meeting to begin.</div></div>

<h2>Captures</h2>
<div id=list><div class=empty>No captures yet.</div></div>

<script>
var SKIP=/HealthCheck|FIXCHECK|WSTEST|deploycheck/i, open={}, lastAudio=null;
function avcolor(v){return v=='real'?'var(--green)':v=='fake'?'var(--red)':'var(--amber)'}
function qcolor(v){return v=='good'?'var(--green)':/CLIP/.test(v)?'var(--red)':/QUIET|silence/i.test(v)?'var(--amber)':'var(--dim)'}
function hms(s){s=Math.round(s);var h=s/3600|0,m=(s%3600)/60|0,x=s%60;
 return h?h+'h '+m+'m':m?m+'m '+(x<10?'0':'')+x+'s':x+'s'}
function tod(ms){var d=new Date(ms);var h=d.getHours(),m=d.getMinutes();var ap=h<12?'AM':'PM';
 h=h%12||12;return h+':'+(m<10?'0':'')+m+' '+ap}

async function quality(){
 var d=await(await fetch('/api/quality')).json();
 var ks=Object.keys(d).filter(k=>!SKIP.test(k)),active=0;
 if(!ks.length){document.getElementById('quality').innerHTML=
  '<div class=empty>Waiting for audio… send a bot into a meeting to begin.</div>';setlive(0);return}
 document.getElementById('quality').innerHTML=ks.map(k=>{
  var s=d[k],lv=Math.min(100,Math.round((s.level||0)*300)),c=qcolor(s.verdict);
  if(s.level>0.01)active++;
  var chip=s.auth_verdict
   ?'<span class=chip style=background:'+avcolor(s.auth_verdict)+'>'+s.auth_verdict.toUpperCase()
     +(s.auth_p!=null?' '+s.auth_p:'')+'<small>authenticity</small></span>'
   :'<span class=pend>scoring…</span>';
  var det=s.level!=null
   ?'<div class=bar><i style="width:'+lv+'%;background:'+c+'"></i></div>'
    +'<div class=det>audio <b>'+s.verdict.toUpperCase()+'</b> · '+s.speech_pct+'% speech · '
    +hms(s.total_sec)+' · '+s.clips+' clip'+(s.clips==1?'':'s')
    +(s.peak>=0.985?' · <span style=color:var(--red)>⚠ clipping</span>':'')+'</div>':'';
  return '<div class=card><div class=chead><span class=who>'+k+'</span>'+chip+'</div>'+det+'</div>'
 }).join('');
 setlive(active)
}
function setlive(n){var on=n>0;document.getElementById('dot').className='dot'+(on?' on':'');
 document.getElementById('livetxt').textContent=on?(n+' active'):'idle'}

function play(name,btn){
 if(lastAudio){lastAudio.pause();if(lastAudio._b)lastAudio._b.textContent='▶'}
 var a=new Audio('/download/'+name);a._b=btn;a.play();btn.textContent='❚❚';lastAudio=a;
 a.onended=function(){btn.textContent='▶';if(lastAudio==a)lastAudio=null}
}
function toggle(id){open[id]=!open[id];list(true)}
var _cache=null;
async function list(fromToggle){
 if(!fromToggle){_cache=(await(await fetch('/captures')).json()).files}
 var files=(_cache||[]).filter(f=>!SKIP.test(f.name));
 if(!files.length){document.getElementById('list').innerHTML='<div class=empty>No captures yet.</div>';return}
 var sess={};
 files.forEach(f=>{var p=f.name.replace(/\.wav$/,'').split('_');
  var ts=p.length>=3?+p[p.length-2]:0, spk=p.slice(1,-2).join('_')||'?';
  var key=spk+'@'+ts;(sess[key]=sess[key]||{spk:spk,ts:ts,items:[]}).items.push(f)});
 var groups=Object.values(sess).sort((a,b)=>b.ts-a.ts);
 document.getElementById('list').innerHTML=groups.map((g,i)=>{
  var id=g.spk+g.ts, mb=g.items.reduce((s,f)=>s+f.mb,0), dur=mb*1e6/32000;
  var isopen=i in open?open[id]:(i==0);open[id]=isopen;
  var body=isopen?'<div class=sbody>'+g.items.map(f=>{
   var d=f.mb*1e6/32000;
   return '<div class=clip><button class=play onclick="play(\''+f.name+'\',this)">▶</button>'
    +'<span class=cname>'+f.name+'</span><span class=cdur>'+hms(d)+' · '+f.mb+' MB</span></div>'
  }).join('')+'</div>':'';
  return '<div class=sess><div class=shead onclick="toggle(\''+id+'\')">'
   +'<span class=stitle><span class="caret'+(isopen?' open':'')+'">▸</span> '+g.spk
   +' · '+(g.ts?tod(g.ts*1000):'session')+'</span>'
   +'<span class=smeta>'+g.items.length+' clips · '+mb.toFixed(1)+' MB · '+hms(dur)+'</span></div>'
   +body+'</div>'
 }).join('')
}

async function send(){
 var b=document.getElementById('sendb'),u=document.getElementById('u').value.trim();
 if(!u){document.getElementById('msg').textContent='paste a meeting link first';return}
 b.disabled=true;b.textContent='sending…';
 try{var r=await fetch('/bot',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({meeting_url:u})});var d=await r.json();
  document.getElementById('msg').textContent=d.ok?('✓ bot sent · '+d.bot_id):('✕ '+(d.detail||d.error||d.status))}
 catch(e){document.getElementById('msg').textContent='✕ '+e}
 b.disabled=false;b.textContent='Send bot';list()
}
list();setInterval(()=>list(),5000);
quality();setInterval(quality,1500);
</script>"""
