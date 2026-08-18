"""Tests for Enterprise API Keys and HMAC Webhook Signatures:
  - Scoped API key generation, prefixing, database lookup, and token revocation
  - Authentication via `Authorization: Bearer snv_live_...`
  - Webhook secret provisioning, rotation, and signature verification (X-Sonave-Signature)
  - Endpoints: GET/POST/DELETE /api/keys, GET/POST /api/settings/webhook-secret
"""
import time
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_mod(railway_mod, tmp_path, monkeypatch):
    import db
    monkeypatch.setattr(db, "_db_path", lambda: tmp_path / "test_keys.db")
    return railway_mod, TestClient(railway_mod.app)


def test_api_key_lifecycle_and_db_methods(tmp_path, monkeypatch):
    import db
    monkeypatch.setattr(db, "_db_path", lambda: tmp_path / "keys_db.db")
    
    # Create user first
    u = db.upsert_google_user("sub_123", "alice@example.com", "Alice", "", "member")
    uid = u["id"]

    # Create API key
    record, raw_token = db.create_api_key(uid, "Treasury Bot", ["read:verdicts", "read:incidents"])
    assert record["name"] == "Treasury Bot"
    assert raw_token.startswith("snv_live_")
    assert "..." in record["prefix"]
    assert "read:verdicts" in record["scopes"]
    assert record["is_active"] == 1

    # Verify key
    user_info, scopes = db.verify_api_key(raw_token)
    assert user_info is not None and user_info["id"] == uid
    assert "read:incidents" in scopes

    # List keys
    keys = db.list_api_keys(uid)
    assert len(keys) == 1
    assert keys[0]["id"] == record["id"]

    # Revoke key
    assert db.revoke_api_key(record["id"], uid) is True
    user_info_after, _ = db.verify_api_key(raw_token)
    assert user_info_after is None  # Revoked key fails verification


def test_webhook_secret_lifecycle_and_signature_verification(tmp_path, monkeypatch):
    import db
    import webhook_dispatcher
    monkeypatch.setattr(db, "_db_path", lambda: tmp_path / "webhook_sec.db")

    u = db.upsert_google_user("sub_wh", "bob@example.com", "Bob", "", "member")
    uid = u["id"]

    sec1 = db.get_or_create_webhook_secret(uid)
    assert sec1.startswith("whsec_")
    assert db.get_or_create_webhook_secret(uid) == sec1  # Idempotent

    sec2 = db.rotate_webhook_secret(uid)
    assert sec2.startswith("whsec_")
    assert sec2 != sec1

    payload_bytes = b'{"event":"voice.alert","confidence":0.95}'
    sig_header, ts = webhook_dispatcher.compute_webhook_signature(payload_bytes, sec2)
    assert f"t={ts}" in sig_header and "v1=" in sig_header

    # Verify valid signature
    assert webhook_dispatcher.verify_webhook_signature(payload_bytes, sig_header, sec2) is True
    
    # Tampered payload fails
    tampered = b'{"event":"voice.alert","confidence":0.10}'
    assert webhook_dispatcher.verify_webhook_signature(tampered, sig_header, sec2) is False

    # Wrong secret fails
    assert webhook_dispatcher.verify_webhook_signature(payload_bytes, sig_header, "whsec_wrong") is False


def test_api_keys_http_endpoints_and_authentication(client_mod, monkeypatch):
    mod, client = client_mod
    monkeypatch.setenv("SONAVE_API_TOKEN", "test-enforce-token")
    u = mod.db.upsert_google_user("sub_http", "charlie@example.com", "Charlie", "", "admin")
    uid = u["id"]
    sess = mod.auth.sign_session(uid)
    client.cookies.set("sonave_session", sess)

    # 1. Create a key via POST /api/keys
    r_create = client.post("/api/keys", json={"name": "SIEM Integration", "scopes": ["read:verdicts", "read:incidents"]})
    assert r_create.status_code == 200
    res = r_create.json()
    assert res["ok"] is True
    raw_token = res["token"]
    key_id = res["key"]["id"]

    # 2. List keys via GET /api/keys
    r_list = client.get("/api/keys")
    assert r_list.status_code == 200
    assert any(k["id"] == key_id for k in r_list.json()["keys"])

    # 3. Authenticate with Bearer snv_live_... on an authenticated endpoint
    api_client = TestClient(mod.app)
    r_auth = api_client.get("/api/quality", headers={"Authorization": f"Bearer {raw_token}"})
    assert r_auth.status_code == 200

    # 4. Get Webhook Secret via GET /api/settings/webhook-secret
    r_sec = client.get("/api/settings/webhook-secret")
    assert r_sec.status_code == 200
    assert r_sec.json()["webhook_secret"].startswith("whsec_")

    # 5. Rotate Webhook Secret via POST /api/settings/webhook-secret/rotate
    r_rot = client.post("/api/settings/webhook-secret/rotate")
    assert r_rot.status_code == 200
    assert r_rot.json()["webhook_secret"] != r_sec.json()["webhook_secret"]

    # 6. Revoke key via DELETE /api/keys/{id}
    r_del = client.delete(f"/api/keys/{key_id}")
    assert r_del.status_code == 200
    assert r_del.json()["ok"] is True

    # 7. Using revoked key now returns 401
    r_revoked = api_client.get("/api/quality", headers={"Authorization": f"Bearer {raw_token}"})
    assert r_revoked.status_code == 401
