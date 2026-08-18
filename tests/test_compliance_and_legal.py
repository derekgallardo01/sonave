"""Tests for Enterprise Compliance Suite & Legal Features:
  - Compliance Cloud Vault Archiver & Manifest integrity checks
  - Legal Chain-of-Custody Certificate of Acoustic Authenticity
  - Executive Risk Posture Analytics & CSV Export
  - Endpoints: /certificate/{id}, /api/incidents/{id}/vault/*, /api/analytics/*
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_app(railway_mod, tmp_path, monkeypatch):
    import db
    import incidents
    import compliance_vault
    monkeypatch.setattr(db, "_db_path", lambda: tmp_path / "comp_app.db")
    monkeypatch.setattr(incidents, "DB_PATH", tmp_path / "comp_incidents.db")
    monkeypatch.setattr(compliance_vault, "_vault_dir", lambda: tmp_path / "vault")
    return railway_mod, TestClient(railway_mod.app)


def test_compliance_vault_archiving_and_manifest(tmp_path, monkeypatch):
    import compliance_vault
    monkeypatch.setattr(compliance_vault, "_vault_dir", lambda: tmp_path / "v_test")

    inc = {
        "id": 505,
        "user_id": "u_treasury_99",
        "speaker": "Impersonated_CFO",
        "first_ts": 1723900000.0,
        "last_ts": 1723900050.0,
        "rolling": 0.962,
        "model": "sonave-xlsr-meet-v2",
        "status": "open",
        "hold": 1
    }
    secret = b"audit-secret-key-xyz"
    raw_audio = b"\x00\x02" * 1600

    manifest = compliance_vault.archive_incident(inc, secret, domain="usesonave.com", audio_pcm=raw_audio)
    assert manifest["incident_id"] == 505
    assert manifest["user_id"] == "u_treasury_99"
    assert "metadata.json" in manifest["assets"]
    assert "report.html" in manifest["assets"]
    assert "audio_snippet.raw" in manifest["assets"]
    assert len(manifest["integrity"]["canonical_digest_sha256"]) == 64
    assert len(manifest["integrity"]["signature_hmac_sha256"]) == 64

    # Retrieve manifest from vault
    fetched = compliance_vault.get_vault_manifest(505, user_id="u_treasury_99")
    assert fetched is not None
    assert fetched["speaker"] == "Impersonated_CFO"
    assert fetched["assets"] == manifest["assets"]


def test_legal_certificate_generation():
    import legal_certificate
    inc = {
        "id": 777,
        "user_id": "u_corp_legal",
        "speaker": "Executive_Impersonator_Target",
        "first_ts": 1723900000.0,
        "last_ts": 1723900080.0,
        "rolling": 0.985,
        "model": "sonave-xlsr-meet-v2",
        "status": "open",
        "hold": 1
    }
    secret = b"legal-cert-key-456"
    html = legal_certificate.generate_legal_certificate_html(inc, domain="usesonave.com", secret=secret)
    assert "Certificate of Acoustic Authenticity" in html
    assert "SNV-CERT-000777" in html
    assert "Executive_Impersonator_Target" in html
    assert "98.5% FAKE" in html
    assert "MANDATORY WIRE HOLD ISSUED" in html
    assert "SHA-256 EVIDENCE DIGEST:" in html


def test_compliance_and_analytics_http_endpoints(client_app):
    mod, client = client_app
    u = mod.db.upsert_google_user("sub_comp", "risk@corp.com", "Risk Director", "", "admin")
    uid = u["id"]
    sess = mod.auth.sign_session(uid)
    client.cookies.set("sonave_session", sess)

    import incidents
    inc = incidents.record("Fraudster_Speaker", 0.94, "sonave-xlsr-meet-v2", user_id=uid)
    assert inc is not None
    inc_id = inc["id"]

    # 1. Legal Certificate Endpoint
    r_cert = client.get(f"/certificate/{inc_id}")
    assert r_cert.status_code == 200
    assert "text/html" in r_cert.headers["content-type"]
    assert "Certificate of Acoustic Authenticity" in r_cert.text
    assert "Fraudster_Speaker" in r_cert.text

    # 2. Vault Archive Endpoint
    r_arch = client.post(f"/api/incidents/{inc_id}/vault/archive")
    assert r_arch.status_code == 200
    assert r_arch.json()["ok"] is True
    assert "manifest" in r_arch.json()

    # 3. Vault Manifest Endpoint
    r_man = client.get(f"/api/incidents/{inc_id}/vault/manifest")
    assert r_man.status_code == 200
    assert r_man.json()["manifest"]["incident_id"] == inc_id

    # 4. Executive Analytics Summary Endpoint
    r_exec = client.get("/api/analytics/executive")
    assert r_exec.status_code == 200
    summary = r_exec.json()["summary"]
    assert "total_wire_holds_triggered" in summary
    assert "estimated_fraud_loss_prevented_usd" in summary
    assert summary["compliance_status"] == "SOC2 / FINRA Compliant"

    # 5. Executive CSV Audit Report Export Endpoint
    r_csv = client.get("/api/analytics/executive-report.csv")
    assert r_csv.status_code == 200
    assert "text/csv" in r_csv.headers["content-type"]
    assert "Incident_ID,Timestamp_UTC,Speaker" in r_csv.text
    assert "Fraudster_Speaker" in r_csv.text
