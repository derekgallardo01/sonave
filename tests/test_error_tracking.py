"""test_error_tracking.py — tests for client error telemetry and server error handling."""
import json
from fastapi.testclient import TestClient
import railway.db as db


def client(railway_mod):
    return TestClient(railway_mod.app)


def test_client_error_telemetry_stores_event(railway_mod):
    c = client(railway_mod)
    payload = {
        "page": "meet-addon",
        "message": "TypeError: Cannot read properties of undefined",
        "source": "https://meet.google.com/add-on.js",
        "lineno": 42,
        "colno": 12,
        "stack": "TypeError: Cannot read properties...\n    at render (add-on.js:42:12)",
        "meet_code": "abc-defg-hij",
        "user_agent": "Mozilla/5.0 TestBrowser"
    }
    r = c.post("/api/telemetry/client-error", json=payload)
    assert r.status_code == 204

    events = db.list_events(kind="client_error", limit=10)
    assert len(events) >= 1
    ev = events[0]
    detail = json.loads(ev["detail"])
    assert detail["page"] == "meet-addon"
    assert "TypeError" in detail["message"]
    assert detail["lineno"] == 42
    assert detail["meet_code"] == "abc-defg-hij"


def test_global_exception_handler_catches_server_error(railway_mod):
    # Add a temporary crashy endpoint to verify the 500 handler
    @railway_mod.app.get("/api/test-crash-handler")
    def crashy_route():
        raise RuntimeError("Simulated unhandled internal bug")

    c = TestClient(railway_mod.app, raise_server_exceptions=False)
    r = c.get("/api/test-crash-handler")
    assert r.status_code == 500
    data = r.json()
    assert data["ok"] is False
    assert data["error"] == "Internal server error"
    assert "error_id" in data

    events = db.list_events(kind="server_error", limit=10)
    assert len(events) >= 1
    ev = events[0]
    detail = json.loads(ev["detail"])
    assert detail["path"] == "/api/test-crash-handler"
    assert "Simulated unhandled internal bug" in detail["error"]
    assert "RuntimeError" in detail["traceback"]


def test_http_exception_not_logged_as_server_error(railway_mod):
    c = client(railway_mod)
    init_errs = len(db.list_events(kind="server_error", limit=100))
    r = c.get("/api/nonexistent-endpoint-xyz-123")
    assert r.status_code == 404
    after_errs = len(db.list_events(kind="server_error", limit=100))
    assert after_errs == init_errs


def test_multi_kind_event_filtering(railway_mod):
    db.add_event("test_user", "client_error", json.dumps({"message": "test client err"}))
    db.add_event("test_user", "server_error", json.dumps({"error": "test server err"}))
    db.add_event("test_user", "signin", json.dumps({"email": "test@example.com"}))

    both = db.list_events(kind="client_error,server_error", limit=20)
    kinds = {e["kind"] for e in both}
    assert "client_error" in kinds
    assert "server_error" in kinds
    assert "signin" not in kinds
