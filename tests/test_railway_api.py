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
    assert "Live authenticity" in r.text


def test_post_verdict_stores_it(railway_mod):
    c = client(railway_mod)
    r = c.post("/api/verdict", json={"speaker": "Derek", "p_fake": 0.82,
                                     "rolling": 0.75, "verdict": "fake"})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert railway_mod.VERDICTS["Derek"]["verdict"] == "fake"


def test_quality_merges_verdict_and_filters_test_speakers(railway_mod):
    railway_mod.VERDICTS["Derek"] = {"p_fake": 0.1, "rolling": 0.1, "verdict": "real"}
    railway_mod.VERDICTS["deploycheck"] = {"p_fake": 0.1, "rolling": 0.1, "verdict": "real"}
    railway_mod.VERDICTS["HealthCheck"] = {"p_fake": 0.1, "rolling": 0.1, "verdict": "real"}
    out = client(railway_mod).get("/api/quality").json()
    assert "Derek" in out and out["Derek"]["auth_verdict"] == "real"
    assert "deploycheck" not in out and "HealthCheck" not in out  # SKIP_SPEAKERS


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
    (tmp_path / "meet_Derek_1_000.wav").write_bytes(b"RIFF0000WAVE")
    out = client(railway_mod).get("/captures").json()
    assert out["files"] and out["files"][0]["name"] == "meet_Derek_1_000.wav"


def test_favicon_served(railway_mod):
    r = client(railway_mod).get("/favicon.svg")
    assert r.status_code == 200 and "svg" in r.headers["content-type"]
