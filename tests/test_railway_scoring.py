"""Railway's off-path hosted-scoring hook (SONAVE_SCORER_URL -> Modal). Mocked HTTP —
verifies the verdict/rolling bookkeeping, retry-swallowing (capture must never break),
and that a missing URL is a safe no-op. `_score_and_store` takes WAV bytes (a live window)."""
import io
import json

import pytest

WAV = b"RIFFxxxxWAVEfmt "   # opaque bytes; the scorer is mocked, so contents don't matter


def test_av_thresholds(railway_mod):
    assert railway_mod._av(0.0) == "real"
    assert railway_mod._av(0.39) == "real"
    assert railway_mod._av(0.4) == "suspect"
    assert railway_mod._av(0.69) == "suspect"
    assert railway_mod._av(0.7) == "fake"
    assert railway_mod._av(1.0) == "fake"


def test_pcm_to_wav_is_valid_wav(railway_mod):
    import wave
    out = railway_mod._pcm_to_wav(b"\x00\x01" * 16000)   # 1 s of PCM
    w = wave.open(io.BytesIO(out), "rb")
    assert w.getframerate() == railway_mod.SR and w.getnchannels() == 1


def test_score_and_store_noop_without_url(railway_mod):
    railway_mod.SCORER_URL = ""
    railway_mod._score_and_store("admin", "Derek", WAV)
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


def test_score_and_store_updates_verdict(railway_mod, monkeypatch):
    railway_mod.SCORER_URL = "http://scorer.test"
    monkeypatch.setattr(railway_mod.urllib.request, "urlopen", _fake_urlopen({"p_fake": 0.9}))
    railway_mod._score_and_store("admin", "Derek", WAV)
    v = railway_mod.VERDICTS[("admin", "Derek")]
    a, pr = railway_mod.SCORE_EMA, railway_mod.SCORE_PRIOR
    # neutral-prior seed: one hot window shows SUSPECT, never an instant FAKE flash
    assert v["p_fake"] == 0.9 and v["verdict"] == "suspect"
    assert railway_mod.ROLL[("admin", "Derek")] == pytest.approx(a * 0.9 + (1 - a) * pr)
    railway_mod._score_and_store("admin", "Derek", WAV)      # sustained -> confirmed
    assert railway_mod.VERDICTS[("admin", "Derek")]["verdict"] == "fake"


def test_score_and_store_rolling_ema(railway_mod, monkeypatch):
    railway_mod.SCORER_URL = "http://scorer.test"
    monkeypatch.setattr(railway_mod.urllib.request, "urlopen", _fake_urlopen({"p_fake": 1.0}))
    railway_mod.ROLL[("admin", "Derek")] = 0.0
    railway_mod._score_and_store("admin", "Derek", WAV)
    a = railway_mod.SCORE_EMA
    assert railway_mod.ROLL[("admin", "Derek")] == pytest.approx(a * 1.0 + (1 - a) * 0.0)


def test_score_and_store_swallows_errors(railway_mod, monkeypatch):
    railway_mod.SCORER_URL = "http://scorer.test"
    monkeypatch.setattr(railway_mod.time, "sleep", lambda *_: None)   # skip retry backoff in test
    def _boom(req, timeout=None):
        raise ConnectionError("scorer down")
    monkeypatch.setattr(railway_mod.urllib.request, "urlopen", _boom)
    railway_mod._score_and_store("admin", "Derek", WAV)   # must NOT raise even after retries
    assert "Derek" not in railway_mod.VERDICTS


def test_score_and_store_retries_then_succeeds(railway_mod, monkeypatch):
    railway_mod.SCORER_URL = "http://scorer.test"
    monkeypatch.setattr(railway_mod.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): self.close()

    def _flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("429-ish")     # first attempt fails, retry succeeds
        return _Resp(json.dumps({"p_fake": 0.9}).encode())

    monkeypatch.setattr(railway_mod.urllib.request, "urlopen", _flaky)
    railway_mod._score_and_store("admin", "Derek", WAV)
    v = railway_mod.VERDICTS[("admin", "Derek")]
    assert calls["n"] == 2 and v["p_fake"] == 0.9                 # retried, then scored
    assert v["verdict"] == "suspect"                              # prior-seeded first window


def test_score_and_store_ignores_silence_response(railway_mod, monkeypatch):
    railway_mod.SCORER_URL = "http://scorer.test"
    monkeypatch.setattr(railway_mod.urllib.request, "urlopen", _fake_urlopen({"p_fake": None}))
    railway_mod._score_and_store("admin", "Derek", WAV)
    assert "Derek" not in railway_mod.VERDICTS


class _Inline:
    """Run the scoring thread synchronously so the WS test is deterministic."""
    def __init__(self, target=None, args=(), daemon=None):
        self._t, self._a = target, args

    def start(self):
        self._t(*self._a)


def test_ws_streams_a_verdict_well_before_the_2min_flush(railway_mod, tmp_path, monkeypatch):
    import base64
    from fastapi.testclient import TestClient
    monkeypatch.setattr(railway_mod, "DATA_DIR", tmp_path)          # don't write to /data
    railway_mod.SCORER_URL = "http://scorer.test"
    monkeypatch.setattr(railway_mod.urllib.request, "urlopen", _fake_urlopen({"p_fake": 0.9}))
    monkeypatch.setattr(railway_mod.threading, "Thread", _Inline)   # score inline
    frame = base64.b64encode(b"\x40\x00" * railway_mod.SR).decode()  # 1 s of PCM per frame
    n = railway_mod.SCORE_SEC + 3                                    # exceed the score hop
    with TestClient(railway_mod.app).websocket_connect("/api/ws/audio") as ws:
        for _ in range(n):
            ws.send_text(json.dumps({"data": {"data": {"buffer": frame,
                                                        "participant": {"name": "Derek"}}}}))
    # a verdict landed from seconds of audio — no 2-min file flush needed
    # (single hot window -> SUSPECT under the neutral-prior EMA seed)
    assert railway_mod.VERDICTS[("admin", "Derek")]["verdict"] == "suspect"
