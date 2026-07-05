"""Railway's off-path hosted-scoring hook (SONAVE_SCORER_URL -> Modal). Mocked HTTP —
verifies the verdict/rolling bookkeeping and that a missing URL is a safe no-op, and
that scoring failures never propagate (the capture path must never break)."""
import io
import json

import pytest


def test_av_thresholds(railway_mod):
    assert railway_mod._av(0.0) == "real"
    assert railway_mod._av(0.39) == "real"
    assert railway_mod._av(0.4) == "suspect"
    assert railway_mod._av(0.69) == "suspect"
    assert railway_mod._av(0.7) == "fake"
    assert railway_mod._av(1.0) == "fake"


def test_score_and_store_noop_without_url(railway_mod, tmp_path):
    railway_mod.SCORER_URL = ""
    f = tmp_path / "c.wav"
    f.write_bytes(b"RIFFxxxxWAVE")
    railway_mod._score_and_store("Derek", f)
    assert "Derek" not in railway_mod.VERDICTS


def _fake_urlopen(payload):
    class _Resp(io.BytesIO):
        def __enter__(self):
            return self
        def __exit__(self, *a):
            self.close()
    def _open(req, timeout=None):
        return _Resp(json.dumps(payload).encode())
    return _open


def test_score_and_store_updates_verdict(railway_mod, tmp_path, monkeypatch):
    railway_mod.SCORER_URL = "http://scorer.test"
    monkeypatch.setattr(railway_mod.urllib.request, "urlopen", _fake_urlopen({"p_fake": 0.8}))
    f = tmp_path / "c.wav"
    f.write_bytes(b"RIFFxxxxWAVE")
    railway_mod._score_and_store("Derek", f)
    v = railway_mod.VERDICTS["Derek"]
    assert v["p_fake"] == 0.8 and v["verdict"] == "fake"
    assert railway_mod.ROLL["Derek"] == pytest.approx(0.8)


def test_score_and_store_rolling_ema(railway_mod, tmp_path, monkeypatch):
    railway_mod.SCORER_URL = "http://scorer.test"
    f = tmp_path / "c.wav"
    f.write_bytes(b"RIFFxxxxWAVE")
    monkeypatch.setattr(railway_mod.urllib.request, "urlopen", _fake_urlopen({"p_fake": 1.0}))
    railway_mod.ROLL["Derek"] = 0.0
    railway_mod._score_and_store("Derek", f)
    assert railway_mod.ROLL["Derek"] == pytest.approx(0.4)  # 0.4*1 + 0.6*0


def test_score_and_store_swallows_errors(railway_mod, tmp_path, monkeypatch):
    railway_mod.SCORER_URL = "http://scorer.test"
    def _boom(req, timeout=None):
        raise ConnectionError("scorer down")
    monkeypatch.setattr(railway_mod.urllib.request, "urlopen", _boom)
    f = tmp_path / "c.wav"
    f.write_bytes(b"RIFFxxxxWAVE")
    railway_mod._score_and_store("Derek", f)  # must NOT raise
    assert "Derek" not in railway_mod.VERDICTS


def test_score_and_store_ignores_silence_response(railway_mod, tmp_path, monkeypatch):
    railway_mod.SCORER_URL = "http://scorer.test"
    monkeypatch.setattr(railway_mod.urllib.request, "urlopen", _fake_urlopen({"p_fake": None}))
    f = tmp_path / "c.wav"
    f.write_bytes(b"RIFFxxxxWAVE")
    railway_mod._score_and_store("Derek", f)
    assert "Derek" not in railway_mod.VERDICTS
