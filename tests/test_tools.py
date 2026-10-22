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
    assert vm._verdict(0.49) == "real"
    assert vm._verdict(0.5) == "suspect"
    assert vm._verdict(0.71) == "suspect"
    assert vm._verdict(0.72) == "fake"


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


# --- send_daily_outreach ---------------------------------------------------
def test_send_daily_outreach_dedup_and_persistence(tmp_path, monkeypatch):
    import csv
    import conftest

    sdo = conftest.load_module("send_daily_outreach", "scripts/send_daily_outreach.py")
    assert sdo.extract_root_domain("https://www.blog.trace3.com/post") == "trace3.com"
    assert sdo.extract_root_domain("Editor <editor@news.co.uk>") == "news.co.uk"
    assert sdo.extract_root_domain("news.substack.com/p/hello") == "news.substack.com"
    assert sdo.extract_email("Editor <Ed@SealedSite.com>") == "ed@sealedsite.com"
    assert sdo.extract_email("invalid-no-tld@localhost") == ""
    assert sdo.normalize_url("https://www2.Example.com/guide/") == "example.com/guide"
    assert sdo.normalize_url("www.Example.com/guide/#top") == "example.com/guide"

    tracker_csv = tmp_path / "tracker.csv"
    dispatch_log = tmp_path / "dispatch.json"
    sr_tracker = tmp_path / "sr.csv"

    with open(sr_tracker, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Platform", "Platform URL", "Contact Email"])
        w.writeheader()
        w.writerow({"Platform": "sealedsite.com", "Platform URL": "https://sealedsite.com/a", "Contact Email": "Editor <ed@sharededitor.org>"})

    fieldnames = [
        "Referring_Page_URL", "DR", "Dofollow", "Outreach_Status",
        "Primary_Email", "Email_Subject", "Email_Body"
    ]
    with open(tracker_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        # 1. Blocked by Sealed Rose tracker domain
        w.writerow({
            "Referring_Page_URL": "https://sealedsite.com/other",
            "DR": "70", "Dofollow": "Yes", "Outreach_Status": "Not Contacted",
            "Primary_Email": "info@sealedsite.com", "Email_Subject": "Subj", "Email_Body": "Body"
        })
        # 1b. Blocked by Sealed Rose angle-bracketed Contact Email on a different domain
        w.writerow({
            "Referring_Page_URL": "https://unseen-publication.org/story",
            "DR": "68", "Dofollow": "Yes", "Outreach_Status": "Not Contacted",
            "Primary_Email": "ed@sharededitor.org", "Email_Subject": "Subj", "Email_Body": "Body"
        })
        # 2. Empty body -> skipped
        w.writerow({
            "Referring_Page_URL": "https://emptybody.com/guide",
            "DR": "65", "Dofollow": "Yes", "Outreach_Status": "Not Contacted",
            "Primary_Email": "ed@emptybody.com", "Email_Subject": "Subj", "Email_Body": ""
        })
        # 3. Valid candidate 1
        w.writerow({
            "Referring_Page_URL": "https://validone.com/article",
            "DR": "60", "Dofollow": "Yes", "Outreach_Status": "Not Contacted",
            "Primary_Email": "Editor <ed@validone.com>", "Email_Subject": "Subj 1", "Email_Body": "Body 1"
        })
        # 4. Duplicate domain of candidate 1 -> skipped
        w.writerow({
            "Referring_Page_URL": "https://blog.validone.com/other",
            "DR": "60", "Dofollow": "Yes", "Outreach_Status": "Not Contacted",
            "Primary_Email": "other@validone.com", "Email_Subject": "Subj 2", "Email_Body": "Body 2"
        })

    monkeypatch.setattr(sdo, "TRACKER_CSV", tracker_csv)
    monkeypatch.setattr(sdo, "DISPATCH_LOG", dispatch_log)
    monkeypatch.setattr(sdo, "SEALED_ROSE_TRACKER", sr_tracker)
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
    monkeypatch.setattr(sdo, "send_via_resend", lambda key, to, subj, body: "re_receipt_123")
    monkeypatch.setattr(sys, "argv", ["send_daily_outreach.py", "--send", "--limit", "5"])

    sdo.main()

    dispatched = json.loads(dispatch_log.read_text(encoding="utf-8"))
    assert len(dispatched) == 1
    assert dispatched[0]["domain"] == "validone.com"
    assert dispatched[0]["to"] == "ed@validone.com"
    assert dispatched[0]["resendId"] == "re_receipt_123"
