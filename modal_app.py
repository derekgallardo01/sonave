"""
modal_app.py — deploy the Sonave detection API (service/app.py) on Modal, on a
scale-to-zero GPU. You pay per-second of actual scoring; idle costs nothing.

One-time:
    pip install modal
    modal token new                 # opens a browser to auth

Iterate / test (hot-reload, temporary URL):
    modal serve modal_app.py

Deploy (persistent URL):
    modal deploy modal_app.py
    # -> https://<you>--sonave-detector-fastapi-app.modal.run

Test it:
    curl https://<url>/healthz
    curl -F "file=@some.wav" https://<url>/score

Then point the product at it: set SONAVE_DETECTOR_URL to that base and have the
orchestrator POST 4 s chunks to /score (or /score_json with base64).

Why T4: XLS-R-300M frozen + a tiny head runs comfortably on a T4 (16 GB) — the
cheapest GPU that fits. Bump to gpu="L4" if you want a bit more headroom/throughput.
"""
import modal

BACKBONE = "facebook/wav2vec2-xls-r-300m"   # public, no HF token needed


def _cache_backbone():
    """Runs at image BUILD time: bake the 1.2 GB XLS-R backbone into the image so
    cold starts don't re-download it."""
    from transformers import AutoFeatureExtractor, Wav2Vec2Model
    Wav2Vec2Model.from_pretrained(BACKBONE)
    AutoFeatureExtractor.from_pretrained(BACKBONE)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")          # librosa / soundfile decoders
    .pip_install(
        "torch",                                    # PyPI CUDA build — matches Modal's T4/L4
        "transformers>=4.44",
        "huggingface_hub>=0.24",
        "soundfile>=0.12",
        "librosa>=0.10",
        "numpy>=1.26",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.29",
        "python-multipart>=0.0.9",
        "pydantic>=2.6",
    )
    .run_function(_cache_backbone)                  # bake backbone weights into the image
    .env({
        "SONAVE_MODEL": "/root/models/sonave_xlsr_meet",   # the balanced Meet model (Stage 6)
        "SONAVE_TAU_REAL": "0.40",
        "SONAVE_TAU_FAKE": "0.70",
        "PYTHONPATH": "/root",
    })
    # add_local_* MUST come last (Modal mounts these at startup, not a build step)
    .add_local_file("config.py", "/root/config.py")
    .add_local_dir("src", "/root/src")
    .add_local_dir("service", "/root/service")
    .add_local_dir("models/sonave_xlsr_meet", "/root/models/sonave_xlsr_meet")
)

app = modal.App("sonave-detector", image=image)


@app.function(
    gpu="T4",
    scaledown_window=300,   # stay warm 5 min after the last request (covers a live meeting), then -> 0
    timeout=600,
    min_containers=0,       # scale to zero when idle = $0
)
@modal.concurrent(max_inputs=8)   # one warm GPU serves several chunk requests at once
@modal.asgi_app()
def fastapi_app():
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from service.app import app as web   # FastAPI app; its startup event warms the model
    return web
