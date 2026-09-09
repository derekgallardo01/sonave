"""tests/test_hf_subscription.py — Tests for Hugging Face Voice Model Subscriptions & Webhooks."""
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from src.pipeline.hf_corpus_harvester import HFCorpusHarvester


def test_hf_discover_trending_models():
    harvester = HFCorpusHarvester()
    models = harvester.discover_trending_hf_models(limit=10)
    assert len(models) >= 5
    assert all("model_id" in m and "author" in m and "pipeline_tag" in m for m in models)


def test_hf_webhook_event_handling():
    harvester = HFCorpusHarvester()
    webhook_payload = {
        "event": "model.created",
        "repo": {"name": "test-org/new-flow-matching-voice"}
    }
    res = harvester.handle_hf_webhook_event(webhook_payload)
    assert res["ok"] is True
    assert res["status"] == "webhook_processed"
    assert res["model_id"] == "test-org/new-flow-matching-voice"


def test_hf_api_endpoints(monkeypatch, tmp_path):
    import sys
    _RAILWAY = Path(__file__).resolve().parent.parent / "railway"
    if str(_RAILWAY) not in sys.path:
        sys.path.insert(0, str(_RAILWAY))

    # Redirect the discovered-model registry to a temp file so the webhook's
    # record path never pollutes the real models/ registry.
    reg = tmp_path / "hf_discovered.json"
    reg.write_text(json.dumps({"discovered_models": [
        {"model_id": "seed/existing", "author": "seed"}], "total_tracked": 1}))
    monkeypatch.setenv("SONAVE_HF_REGISTRY", str(reg))

    from app import app
    client = TestClient(app)

    # 1. GET /api/hf/trending reads the (seeded) registry
    r = client.get("/api/hf/trending")
    assert r.status_code == 200
    assert len(r.json()["discovered_models"]) > 0

    # 2. POST /api/webhooks/hf-model-update — secret-gated (CASA hardening):
    #    unconfigured -> 404, wrong/missing secret -> 403, correct secret -> 200
    wh_payload = {"event": "repo.updated", "repo_id": "community/ultra-tts-v2"}
    assert client.post("/api/webhooks/hf-model-update", json=wh_payload).status_code == 404
    monkeypatch.setenv("SONAVE_HF_WEBHOOK_SECRET", "hf-test-secret")
    assert client.post("/api/webhooks/hf-model-update", json=wh_payload).status_code == 403
    hdr = {"X-Webhook-Secret": "hf-test-secret"}

    # HF verification ping is acknowledged (lets the webhook be enabled)
    r_ping = client.post("/api/webhooks/hf-model-update", json={"event": "ping"}, headers=hdr)
    assert r_ping.status_code == 200 and r_ping.json()["status"] == "ping_received"

    # a real model event is RECORDED (never 503) and surfaces via /api/hf/trending
    r_wh = client.post("/api/webhooks/hf-model-update", json=wh_payload, headers=hdr)
    assert r_wh.status_code == 200 and r_wh.json()["status"] == "recorded"
    assert r_wh.json()["model"] == "community/ultra-tts-v2"
    listed = client.get("/api/hf/trending").json()["discovered_models"]
    assert any(m["model_id"] == "community/ultra-tts-v2" for m in listed)
