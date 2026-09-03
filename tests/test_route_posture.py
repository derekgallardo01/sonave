"""Anonymous route-posture sweep (CASA hardening regression net).

Invariant: an unauthenticated caller must get exactly 200 on the public
marketing surface and NEVER a 200 or a 5xx anywhere else. This is the test
that catches the next accidentally-open or crash-on-probe endpoint before a
security scanner does.
"""
import pytest
from fastapi.routing import APIRoute, APIWebSocketRoute
from fastapi.testclient import TestClient

import conftest

TOKEN = "machine-token-123"

# Anonymous GETs that must serve content.
PUBLIC_200 = {
    "/", "/benchmarks", "/guides", "/privacy", "/terms", "/onboarding",
    "/llms.txt", "/robots.txt", "/sitemap.xml", "/console", "/meet-addon",
    "/favicon.ico", "/favicon.svg", "/og.png", "/console-shot.png",
    "/icon-120.png", "/icon-128.png",
}

# Dummy values for parameterized paths.
PARAMS = {"slug": "nope", "incident_id": "1", "name": "x.wav",
          "speaker": "x", "key_id": "1"}


@pytest.fixture
def mod(monkeypatch, tmp_path):
    monkeypatch.setenv("SONAVE_SCORER_URL", "")
    monkeypatch.setenv("SONAVE_API_TOKEN", TOKEN)       # secured mode: no open-dev fallback
    monkeypatch.setenv("SONAVE_SESSION_SECRET", "sec")
    monkeypatch.setenv("SONAVE_RECALL_API_KEY", "test-key")
    return conftest.load_module("rwapp_posture", "railway/app.py")


def _fill(path: str) -> str:
    for k, v in PARAMS.items():
        path = path.replace("{" + k + "}", v)
    return path


def test_no_route_is_open_or_crashy_to_anonymous(mod):
    c = TestClient(mod.app, base_url="https://testserver")
    failures = []
    for route in mod.app.router.routes:
        if not isinstance(route, APIRoute):
            continue
        path = _fill(route.path)
        for method in route.methods - {"HEAD", "OPTIONS"}:
            r = c.request(method, path, json={} if method in ("POST", "PUT") else None,
                          follow_redirects=False)
            if route.path in PUBLIC_200 and method == "GET":
                if r.status_code != 200:
                    failures.append(f"{method} {route.path}: public page returned {r.status_code}")
            else:
                if r.status_code in (200, 201):
                    failures.append(f"{method} {route.path}: OPEN to anonymous ({r.status_code})")
                elif r.status_code >= 500:
                    failures.append(f"{method} {route.path}: 5xx on anonymous probe ({r.status_code})")
    assert not failures, "route posture violations:\n  " + "\n  ".join(failures)


def test_websockets_reject_anonymous(mod):
    for path in [r.path for r in mod.app.router.routes if isinstance(r, APIWebSocketRoute)]:
        with pytest.raises(Exception):
            with TestClient(mod.app).websocket_connect(path) as ws:
                ws.receive_text()
