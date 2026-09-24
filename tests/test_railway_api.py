"""The Railway capture service HTTP contract — page render, verdict push, quality
merge + test-speaker filtering, capture listing, and path-traversal safety."""
from fastapi.testclient import TestClient


def client(railway_mod):
    return TestClient(railway_mod.app)


def test_index_renders_with_no_leftover_placeholders(railway_mod):
    r = client(railway_mod).get("/")
    assert r.status_code == 200
    for ph in ("__DOMAIN__", "__KEY__", "__FAVICON__"):
        assert ph not in r.text
    assert "Sonave" in r.text


def test_post_verdict_stores_it(railway_mod):
    c = client(railway_mod)
    r = c.post("/api/verdict", json={"speaker": "Derek", "p_fake": 0.82,
                                     "rolling": 0.75, "verdict": "fake"})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert railway_mod.VERDICTS[("admin", "Derek")]["verdict"] == "fake"


def test_quality_merges_verdict_and_filters_test_speakers(railway_mod):
    railway_mod.VERDICTS[("admin", "Derek")] = {"p_fake": 0.1, "rolling": 0.1, "verdict": "real"}
    railway_mod.VERDICTS[("admin", "deploycheck")] = {"p_fake": 0.1, "rolling": 0.1, "verdict": "real"}
    railway_mod.VERDICTS[("admin", "HealthCheck")] = {"p_fake": 0.1, "rolling": 0.1, "verdict": "real"}
    out = client(railway_mod).get("/api/quality").json()
    assert "Derek" in out and out["Derek"]["auth_verdict"] == "real"
    assert "deploycheck" not in out and "HealthCheck" not in out  # SKIP_SPEAKERS


def test_quality_speaker_with_active_audio_under_4s(railway_mod):
    railway_mod.QUALITY[("admin", "Alice")] = {
        "level": 0.05, "peak": 0.1, "clips": 0, "speech_sec": 1.5, "total_sec": 2.0, "state": "speaking"
    }
    railway_mod.VERDICTS[("admin", "Alice")] = {"p_fake": 0.05, "rolling": 0.05, "verdict": "real"}
    out = client(railway_mod).get("/api/quality").json()
    assert "Alice" in out
    assert out["Alice"]["auth_verdict"] is None

    railway_mod.QUALITY[("admin", "Executive Baritone (AI Clone)")] = {
        "level": 0.05, "peak": 0.1, "clips": 0, "speech_sec": 1.5, "total_sec": 2.0, "state": "speaking"
    }
    railway_mod.VERDICTS[("admin", "Executive Baritone (AI Clone)")] = {"p_fake": 0.98, "rolling": 0.98, "verdict": "fake"}
    out2 = client(railway_mod).get("/api/quality").json()
    assert out2["Executive Baritone (AI Clone)"]["auth_verdict"] == "fake"


def test_data_progress_odometer(railway_mod, tmp_path, monkeypatch):
    monkeypatch.setattr(railway_mod, "DATA_DIR", tmp_path)
    ws = tmp_path / "admin"                                 # the machine principal's workspace
    ws.mkdir()
    body = b"\x00" * (44 + 32000 * 60)                      # 60 s of PCM16 mono 16k
    (ws / "meet_Derek_1755024300_000.wav").write_bytes(body)
    (ws / "meet_Ana_1755024900_000.wav").write_bytes(body)
    (ws / "meet_HealthCheck_1_000.wav").write_bytes(body)   # filtered test speaker
    out = client(railway_mod).get("/api/data_progress").json()
    assert out["files"] == 2 and out["speakers"] == 2 and out["sessions"] == 2
    assert out["hours"] == round(120 / 3600, 2)
    assert out["last_capture_ts"] == 1755024900
    assert out["m1_target_hours"] == 15


def test_download_rejects_path_traversal(railway_mod, tmp_path, monkeypatch):
    monkeypatch.setattr(railway_mod, "DATA_DIR", tmp_path)
    # encoded traversal never matches the single-segment {name} route -> file not served
    r = client(railway_mod).get("/download/..%2f..%2fconfig.py")
    assert r.status_code == 404
    # a bare name outside DATA_DIR is stripped by Path(name).name and simply not found
    r2 = client(railway_mod).get("/download/nope.wav")
    assert r2.status_code == 200 and r2.json() == {"error": "not found"}


def test_captures_returns_files_list(railway_mod, tmp_path, monkeypatch):
    monkeypatch.setattr(railway_mod, "DATA_DIR", tmp_path)
    (tmp_path / "admin").mkdir()
    (tmp_path / "admin" / "meet_Derek_1_000.wav").write_bytes(b"RIFF0000WAVE")
    out = client(railway_mod).get("/captures").json()
    assert out["files"] and out["files"][0]["name"] == "meet_Derek_1_000.wav"


def test_legal_pages_render(railway_mod):
    c = client(railway_mod)
    for path, marker in (("/privacy", "Privacy Policy"), ("/terms", "Terms of Service")):
        r = c.get(path)
        assert r.status_code == 200 and marker in r.text and "__FAVICON__" not in r.text


def test_favicon_served(railway_mod):
    r = client(railway_mod).get("/favicon.svg")
    assert r.status_code == 200 and "svg" in r.headers["content-type"]


def test_favicon_ico_served(railway_mod):
    r = client(railway_mod).get("/favicon.ico")
    assert r.status_code == 200
    assert "icon" in r.headers["content-type"]
    assert len(r.content) > 100


def test_apple_touch_icon_served(railway_mod):
    r = client(railway_mod).get("/apple-touch-icon.png")
    assert r.status_code == 200
    assert "png" in r.headers["content-type"]
    assert len(r.content) > 100


def test_user_telemetry_and_events(railway_mod):
    c = client(railway_mod)
    # Test _format_activity
    fmt = railway_mod._format_activity("u_test", "meet_media_connect", {"space": "abc-defg-hij", "ok": True, "email": "test@sonave.com"})
    assert "Meet Call Joined" in fmt and "abc-defg-hij" in fmt

    fmt_sim = railway_mod._format_activity("u_test", "simulation_run", {"voice_name": "ElevenLabs CEO", "is_real": False, "confidence": "98.5%", "email": "test@sonave.com"})
    assert "Threat Demo Tested" in fmt_sim and "ElevenLabs CEO" in fmt_sim

    fmt_ack = railway_mod._format_activity("u_test", "incident_ack", {"incident_id": 42, "speaker": "CEO", "email": "test@sonave.com"})
    assert "Wire Hold Cleared" in fmt_ack

    fmt_cloner = railway_mod._format_activity("u_test", "cloner_opened", {"email": "test@sonave.com"})
    assert "Live Mic Cloner Opened" in fmt_cloner

    # Test /api/telemetry/event endpoint
    r = c.post("/api/telemetry/event", json={"kind": "cloner_opened", "detail": {"source": "test"}})
    assert r.status_code == 204

    # Test db.get_user_telemetry
    user = railway_mod.db.upsert_google_user("sub_test_123", "telemetry_test@sonave.com", "Test User", "", "member")
    uid = user["id"]
    railway_mod.db.add_event(uid, "meet_media_connect", '{"space": "xyz-test-space", "ok": true}')
    railway_mod.db.add_event(uid, "simulation_run", '{"voice_name": "ElevenLabs CEO clone", "is_real": false}')

    telem = railway_mod.db.get_user_telemetry(uid)
    assert telem["meets_total"] >= 1
    assert telem["sims_total"] >= 1
    assert "xyz-test-space" in telem["recent_spaces"]
    assert "ElevenLabs CEO clone" in telem["recent_simulations"]

    # Test /api/admin/users/{user_id}/telemetry
    admin_user = railway_mod.db.upsert_google_user("sub_admin_456", "admin@sonave.com", "Admin User", "", "admin")
    admin_sess = railway_mod.auth.sign_session(admin_user["id"])
    r = c.get(f"/api/admin/users/{uid}/telemetry", headers={"Authorization": f"Bearer {admin_sess}"})
    assert r.status_code == 200
    data = r.json()
    assert "user" in data and "events" in data
    assert data["user"]["meets_total"] >= 1
    assert len(data["events"]) >= 2

