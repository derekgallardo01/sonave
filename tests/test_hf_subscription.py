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


def test_hf_api_endpoints():
    import sys
    _RAILWAY = Path(__file__).resolve().parent.parent / "railway"
    if str(_RAILWAY) not in sys.path:
        sys.path.insert(0, str(_RAILWAY))

    from app import app
    client = TestClient(app)

    # 1. GET /api/hf/trending
    r = client.get("/api/hf/trending")
    assert r.status_code == 200
    data = r.json()
    assert "discovered_models" in data
    assert len(data["discovered_models"]) > 0

    # 2. POST /api/webhooks/hf-model-update
    wh_payload = {"event": "repo.updated", "repo_id": "community/ultra-tts-v2"}
    r_wh = client.post("/api/webhooks/hf-model-update", json=wh_payload)
    assert r_wh.status_code == 200
    assert r_wh.json()["ok"] is True
