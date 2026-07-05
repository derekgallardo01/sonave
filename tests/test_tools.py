"""The two operator tools: verdict_monitor (live scoring client) and play_into_meet
(the VB-CABLE feeder). Network and audio devices are mocked."""
import io
import json
import sys
import types

import pytest

import verdict_monitor as vm
import play_into_meet as pim


# --- verdict_monitor -------------------------------------------------------
def test_verdict_thresholds():
    assert vm._verdict(0.0) == "real"
    assert vm._verdict(0.39) == "real"
    assert vm._verdict(0.4) == "suspect"
    assert vm._verdict(0.69) == "suspect"
    assert vm._verdict(0.7) == "fake"


def test_post_clip_builds_multipart_and_returns_pfake(tmp_path, monkeypatch):
    f = tmp_path / "c.wav"
    f.write_bytes(b"RIFFdata")
    captured = {}

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): self.close()

    def _open(req, timeout=None):
        captured["url"] = req.full_url
        captured["ctype"] = req.headers.get("Content-type", "")
        captured["body"] = req.data
        return _Resp(json.dumps({"p_fake": 0.73}).encode())

    monkeypatch.setattr(vm.urllib.request, "urlopen", _open)
    p = vm._post_clip("http://scorer.test", f)
    assert p == 0.73
    assert captured["url"].endswith("/score_clip")
    assert "multipart/form-data; boundary=" in captured["ctype"]
    assert b'name="file"' in captured["body"] and b"RIFFdata" in captured["body"]


def test_post_clip_none_for_silence(tmp_path, monkeypatch):
    f = tmp_path / "c.wav"; f.write_bytes(b"x")

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): self.close()

    monkeypatch.setattr(vm.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp(json.dumps({"p_fake": None}).encode()))
    assert vm._post_clip("http://scorer.test", f) is None


# --- play_into_meet --------------------------------------------------------
def test_files_discovers_audio_recursively(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.mp3").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    found = pim._files(str(tmp_path))
    names = {p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in found}
    assert names == {"a.wav", "b.mp3"}


def _fake_sounddevice(devices):
    m = types.ModuleType("sounddevice")
    m.query_devices = lambda: devices
    return m


def test_find_device_by_name_substring(monkeypatch):
    devs = [
        {"name": "Speakers (Realtek)", "max_output_channels": 2, "default_samplerate": 48000},
        {"name": "CABLE Input (VB-Audio)", "max_output_channels": 2, "default_samplerate": 44100},
        {"name": "Mic", "max_output_channels": 0, "default_samplerate": 16000},
    ]
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sounddevice(devs))
    idx, sr = pim._find_device("CABLE Input")
    assert idx == 1 and sr == 44100


def test_find_device_by_index(monkeypatch):
    devs = [{"name": "A", "max_output_channels": 2, "default_samplerate": 48000}]
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sounddevice(devs))
    idx, sr = pim._find_device("0")
    assert idx == 0 and sr == 48000


def test_find_device_no_match_raises(monkeypatch):
    devs = [{"name": "Speakers", "max_output_channels": 2, "default_samplerate": 48000}]
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sounddevice(devs))
    with pytest.raises(SystemExit):
        pim._find_device("nonexistent-device")
