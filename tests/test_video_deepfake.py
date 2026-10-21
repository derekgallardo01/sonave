"""Tests for multimodal video deepfake detection (face-swap, neural avatar, and AV lip-sync)
across src/model_video.py, service/app.py, railway/attribution.py, railway/forensics.py,
railway/meet_media_ingest.py, and railway/app.py."""
from __future__ import annotations

import base64
import struct
import zlib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import load_module, pcm16


def _encode_rgb_png(rgb: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 array as a valid standard PNG using only stdlib."""
    h, w, _ = rgb.shape
    raw_rows = bytearray()
    for y in range(h):
        raw_rows.append(0)  # filter type 0 (None)
        raw_rows.extend(rgb[y].tobytes())

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw_rows), level=6)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def _make_natural_face_rgb(h: int = 128, w: int = 128, seed: int = 42, mouth_open: float = 0.0) -> np.ndarray:
    """Create a realistic textured RGB frame with a natural skin-toned face and organic micro-texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Soft studio background with organic grain
    bg_r = 70.0 + 20.0 * (yy / h) + rng.normal(0, 6.5, (h, w))
    bg_g = 75.0 + 18.0 * (xx / w) + rng.normal(0, 6.5, (h, w))
    bg_b = 85.0 + 15.0 * (yy / h) + rng.normal(0, 6.5, (h, w))

    # Elliptical face region with smooth natural transition and rich high-frequency pore texture
    cy, cx = h * 0.5, w * 0.5
    ry, rx = h * 0.34, w * 0.28
    dist = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
    alpha = np.clip((1.15 - dist) / 0.35, 0.0, 1.0)

    tex = rng.normal(0, 7.5, (h, w)) + 4.0 * np.sin(xx * 0.9) * np.cos(yy * 0.9)
    skin_r = 198.0 - 14.0 * dist + tex
    skin_g = 152.0 - 12.0 * dist + tex * 0.95
    skin_b = 124.0 - 10.0 * dist + tex * 0.9

    r = bg_r * (1.0 - alpha) + skin_r * alpha
    g = bg_g * (1.0 - alpha) + skin_g * alpha
    b = bg_b * (1.0 - alpha) + skin_b * alpha

    # Mouth region modulation when speaking
    if mouth_open > 0:
        m_mask = (np.abs(yy - h * 0.64) < (2.5 + mouth_open * 6.0)) & (np.abs(xx - w * 0.5) < w * 0.10)
        r[m_mask] = 90.0 + mouth_open * 40.0
        g[m_mask] = 45.0
        b[m_mask] = 50.0

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _make_faceswap_rgb(h: int = 128, w: int = 128) -> np.ndarray:
    """Create a frame exhibiting classic face-swap artifacts:
    1. Sharp rectangular splice seam at the face bounding box boundary
    2. Over-smoothed (GAN/autoencoder low-pass) inner face lacking high-frequency skin texture
    3. Chrominance bleed across the blending mask boundary."""
    rng = np.random.default_rng(99)
    # High-texture background
    rgb = rng.integers(30, 220, size=(h, w, 3), dtype=np.uint8).astype(np.float32)

    y0, y1 = int(h * 0.22), int(h * 0.78)
    x0, x1 = int(w * 0.24), int(w * 0.76)

    # Flat over-smoothed inner face patch (zero high-frequency variance)
    rgb[y0:y1, x0:x1, 0] = 205.0
    rgb[y0:y1, x0:x1, 1] = 148.0
    rgb[y0:y1, x0:x1, 2] = 118.0

    # Sharp high-contrast step seam + chroma bleed ring at boundary
    for b_pad in range(-2, 3):
        rgb[max(0, y0 + b_pad), x0:x1, :] = [255.0, 20.0, 240.0]
        rgb[min(h - 1, y1 + b_pad), x0:x1, :] = [255.0, 20.0, 240.0]
        rgb[y0:y1, max(0, x0 + b_pad), :] = [20.0, 250.0, 240.0]
        rgb[y0:y1, min(w - 1, x1 + b_pad), :] = [20.0, 250.0, 240.0]

    return np.clip(rgb, 0, 255).astype(np.uint8)


# --- 1. Core src/model_video.py unit tests ------------------------------------
def test_decode_frame_and_face_detection():
    import model_video

    nat_rgb = _make_natural_face_rgb(96, 96)
    png_bytes = _encode_rgb_png(nat_rgb)
    decoded = model_video.decode_frame(png_bytes)
    assert decoded is not None
    assert decoded.shape == (96, 96, 3)
    assert np.allclose(decoded * 255.0, nat_rgb, atol=1.5)

    face_info = model_video.detect_face_region(decoded)
    assert face_info["face_detected"] is True
    assert len(face_info["bbox"]) == 4


def test_spatial_frequency_distinguishes_natural_vs_faceswap():
    import model_video

    det = model_video.VideoDeepfakeDetector()
    nat_png = _encode_rgb_png(_make_natural_face_rgb(128, 128, seed=7))
    swap_png = _encode_rgb_png(_make_faceswap_rgb(128, 128))

    r_nat = det.score_frame(nat_png, speaker_id="alice_nat", audio_rms=0.01, timestamp=100.0)
    r_swap = det.score_frame(swap_png, speaker_id="bob_swap", audio_rms=0.01, timestamp=100.0)

    assert r_nat["video_p"] < 0.45, f"Natural frame scored too high: {r_nat}"
    assert r_nat["verdict"] == "real"
    assert r_swap["video_p"] >= 0.65, f"Face-swap frame scored too low: {r_swap}"
    assert r_swap["verdict"] == "fake"
    assert r_swap["spatial_seam_score"] > r_nat["spatial_seam_score"]


def test_av_lip_sync_detects_frozen_face_during_speech():
    import model_video

    det = model_video.VideoDeepfakeDetector()
    frozen_frame = _encode_rgb_png(_make_natural_face_rgb(96, 96, seed=11, mouth_open=0.0))

    # Feed 6 consecutive identical frames while audio_rms indicates loud active speech
    last_res = None
    for i in range(6):
        last_res = det.score_frame(
            frozen_frame, speaker_id="frozen_avatar", audio_rms=0.28, timestamp=200.0 + i * 0.2
        )

    assert last_res is not None
    assert last_res["av_sync"]["desync_flag"] is True
    assert last_res["av_sync_score"] < 0.55


def test_multimodal_fusion_verdict():
    import model_video

    # Voice real + Video fake -> fused FAKE
    f1 = model_video.fuse_multimodal_verdict(audio_p=0.04, video_p=0.91)
    assert f1["verdict"] == "fake"
    assert f1["p_fake"] >= 0.65

    # Voice fake + Video real -> fused FAKE
    f2 = model_video.fuse_multimodal_verdict(audio_p=0.95, video_p=0.05)
    assert f2["verdict"] == "fake"
    assert f2["p_fake"] >= 0.70

    # Both real -> fused REAL
    f3 = model_video.fuse_multimodal_verdict(audio_p=0.03, video_p=0.06)
    assert f3["verdict"] == "real"


# --- 2. FastAPI scorer service/app.py tests -----------------------------------
def test_service_score_video_and_multimodal_endpoints(monkeypatch):
    monkeypatch.setenv("SONAVE_API_TOKEN", "svc-secret")
    svc_mod = load_module("svc_video_test", "service/app.py")
    import detector
    monkeypatch.setattr(
        detector,
        "score_bytes",
        lambda b: {"p_fake": 0.05, "verdict": "real", "confidence": 0.9, "model_version": "test-v1"},
    )

    c = TestClient(svc_mod.app)
    swap_b64 = base64.b64encode(_encode_rgb_png(_make_faceswap_rgb(96, 96))).decode()

    # Unauthenticated -> 401
    assert c.post("/score_video", json={"frame_b64": swap_b64}).status_code == 401
    assert c.post("/score_multimodal", json={"frame_b64": swap_b64}).status_code == 401

    headers = {"X-Sonave-Token": "svc-secret"}
    r_vid = c.post("/score_video", json={"frame_b64": swap_b64, "speaker_id": "cfo"}, headers=headers)
    assert r_vid.status_code == 200
    j_vid = r_vid.json()
    assert j_vid["video_p"] >= 0.65
    assert j_vid["verdict"] == "fake"

    r_mm = c.post(
        "/score_multimodal",
        json={"frame_b64": swap_b64, "speaker_id": "cfo", "audio_rms": 0.25},
        headers=headers,
    )
    assert r_mm.status_code == 200
    j_mm = r_mm.json()
    assert j_mm["verdict"] == "fake"
    assert j_mm["video_p"] >= 0.65


# --- 3. Attribution, Forensics & Meet Media Ingest tests ----------------------
def test_video_attribution_and_forensic_report():
    import attribution
    import forensics

    attr = attribution.attribute_visual_engine(0.92, speaker_name="CFO", engine_hint="deepfacelive")
    assert attr is not None
    assert "DeepFaceLive" in attr["engine_name"]
    assert attr["modality"] == "video"

    sync = attribution.compute_av_sync_status(0.92, av_sync_score=0.22, lag_ms=190)
    assert sync["is_desynced"] is True
    assert sync["lag_ms"] >= 140

    inc = {
        "id": 77,
        "speaker": "CFO (Face-Swap Clone)",
        "first_ts": 1700000000.0,
        "last_ts": 1700000030.0,
        "peak_p": 0.946,
        "rolling": 0.946,
        "windows": 4,
        "model": "deepfacelive",
        "status": "open",
        "hold": 1,
        "video_p": 0.946,
        "video_attribution": attr,
        "av_sync": sync,
        "visual_metrics": {"spatial_seam_score": 0.91, "temporal_jitter_score": 0.78},
    }
    html = forensics.generate_report_html(inc, domain="usesonave.com", secret="test-sec")
    assert "DeepFaceLive" in html
    assert "94.6%" in html
    assert "DESYNC" in html or "190" in html


def test_meet_media_ingest_video_sdp_and_frame_callback():
    import meet_media_ingest

    received = []
    sess = meet_media_ingest.MeetMediaSession(
        space_id="abc-defg-hij",
        access_token="tok",
        on_video_frame=lambda spk, fb, ts: received.append((spk, fb, ts)),
    )
    sdp = sess.build_sdp_offer()
    assert "m=audio" in sdp
    assert "m=video" in sdp
    assert "VP8/90000" in sdp

    sess.ingest_speaker_video_frame("Participant_1", b"\x89PNG_fake_frame", ts=123.4)
    assert sess.video_frames_received == 1
    assert len(received) == 1
    assert received[0][0] == "Participant_1"
    assert sess.status()["video_frames_received"] == 1


# --- 4. Railway App end-to-end multimodal video tests -------------------------
def test_railway_video_score_simulation_and_multimodal_quality(railway_mod):
    c = TestClient(railway_mod.app)

    # 1. Inject simulated face-swap attack
    r_sim = c.post(
        "/api/generator/inject-video-test",
        json={"speaker_name": "CFO (Face-Swap Clone)", "engine_id": "heygen_liveavatar", "is_real": False, "lag_ms": 195},
    )
    assert r_sim.status_code == 200
    j_sim = r_sim.json()
    assert j_sim["ok"] is True
    assert j_sim["video_verdict"] == "fake"
    assert j_sim["video_p"] >= 0.90
    assert "HeyGen" in j_sim["video_attribution"]["engine_name"]
    assert j_sim["av_sync"]["is_desynced"] is True

    # 2. Verify GET /api/quality fuses video verdict into speaker row
    r_q = c.get("/api/quality")
    assert r_q.status_code == 200
    q_data = r_q.json()
    spk_key = j_sim["speaker"]
    assert spk_key in q_data
    row = q_data[spk_key]
    assert row["video_verdict"] == "fake"
    assert row["auth_verdict"] == "fake"
    assert row["auth_p"] >= 0.90
    assert row["video_attribution"]["engine_id"] == "heygen_liveavatar"
    assert row["av_sync"]["is_desynced"] is True

    # 3. Verify an incident was automatically opened and its forensic report includes video telemetry
    r_inc = c.get("/api/incidents")
    assert r_inc.status_code == 200
    inc_list = r_inc.json()["incidents"]
    assert any(i["speaker"] == spk_key and i["hold"] == 1 for i in inc_list)
    inc_id = next(i["id"] for i in inc_list if i["speaker"] == spk_key)

    r_rep = c.get(f"/report/{inc_id}")
    assert r_rep.status_code == 200
    assert "HeyGen" in r_rep.text

    # 4. Clear simulated threats
    r_clr = c.post("/api/generator/clear-test")
    assert r_clr.status_code == 200
    assert spk_key in r_clr.json()["removed"]


def test_railway_video_score_endpoint_and_websocket(railway_mod):
    c = TestClient(railway_mod.app)
    nat_b64 = base64.b64encode(_encode_rgb_png(_make_natural_face_rgb(96, 96))).decode()
    swap_b64 = base64.b64encode(_encode_rgb_png(_make_faceswap_rgb(96, 96))).decode()

    # POST /api/video/score with natural frame
    r1 = c.post("/api/video/score", json={"speaker": "Host_User", "frame_b64": nat_b64, "audio_rms": 0.02})
    assert r1.status_code == 200
    assert r1.json()["video_verdict"] == "real"

    # WebSocket /api/ws/video-stream with face-swap frames
    with c.websocket_connect("/api/ws/video-stream?speaker=Remote_Caller") as ws:
        ws.send_json({"speaker": "Remote_Caller", "frame_b64": swap_b64, "audio_rms": 0.22, "mouth_delta": 0.001})
        msg1 = ws.receive_json()
        assert msg1["video_p"] >= 0.65
        assert msg1["video_verdict"] == "fake"

        ws.send_json({"speaker": "Remote_Caller", "frame_b64": swap_b64, "audio_rms": 0.24, "mouth_delta": 0.001})
        msg2 = ws.receive_json()
        assert msg2["fake_streak"] >= 2
        assert msg2["video_verdict"] == "fake"


def test_recall_ws_audio_ingests_video_separate_png(railway_mod):
    c = TestClient(railway_mod.app)
    swap_b64 = base64.b64encode(_encode_rgb_png(_make_faceswap_rgb(96, 96))).decode()

    with c.websocket_connect("/api/ws/audio") as ws:
        ws.send_json({
            "event": "video_separate_png.data",
            "data": {
                "data": {
                    "buffer": swap_b64,
                    "participant": {"id": 101, "name": "Recall Video Caller"},
                }
            },
        })
        # Also send an audio frame so ws_audio processes cleanly
        ws.send_json({
            "event": "audio_separate_raw.data",
            "data": {
                "data": {
                    "buffer": base64.b64encode(pcm16([0.1] * 1600)).decode(),
                    "participant": {"id": 101, "name": "Recall Video Caller"},
                }
            },
        })

    q = c.get("/api/quality").json()
    assert "Recall_Video_Caller" in q
    assert q["Recall_Video_Caller"]["video_p"] >= 0.65
    assert q["Recall_Video_Caller"]["video_verdict"] == "fake"


def test_meet_addon_and_console_include_video_shield_controls(railway_mod):
    c = TestClient(railway_mod.app)
    r_addon = c.get("/meet-addon")
    assert r_addon.status_code == 200
    assert "camera=(self" in r_addon.headers.get("permissions-policy", "")
    assert "videoShieldBar" in r_addon.text
    assert "toggleVideoShield" in r_addon.text
    assert "runVideoFaceSwapTest" in r_addon.text
    assert "addonVideoEngineSel" in r_addon.text
    assert "onAddonVideoEngineChange" in r_addon.text
    assert "fallbackSigninBtn" in r_addon.text
    assert "handleFallbackSignin" in r_addon.text
    assert "isLocalHost()" in r_addon.text
    assert "saveTok('__session__')" in r_addon.text

    r_con = c.get("/console")
    assert r_con.status_code == 200
    assert "injectConsoleVideoSim" in r_con.text
    assert "MULTIMODAL DEEPFAKE RISK" in r_con.text


def test_inject_all_five_visual_deepfake_models(railway_mod):
    c = TestClient(railway_mod.app)
    expected_engines = [
        ("deepfacelive", "CFO (DeepFaceLive Clone)", "DeepFaceLive", "mask seam", 184),
        ("heygen_liveavatar", "VP Finance (HeyGen Avatar)", "HeyGen", "diffusion", 216),
        ("facefusion_simswap", "Director (FaceFusion Swap)", "FaceFusion", "DCT", 162),
        ("wav2lip_retalking", "Exec (Wav2Lip Lip-Sync)", "Wav2Lip", "96x96", 242),
        ("hallo_liveportrait", "Partner (LivePortrait Puppet)", "LivePortrait", "jitter", 196),
    ]
    seen_seams = set()
    seen_lags = set()
    injected_speakers = []
    for eng_id, spk_name, expected_name_sub, expected_sig_sub, expected_lag in expected_engines:
        r = c.post(
            "/api/generator/inject-video-test",
            json={"speaker_name": spk_name, "engine_id": eng_id, "is_real": False},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["video_verdict"] == "fake"
        assert body["video_p"] >= 0.85
        attr = body["video_attribution"]
        assert attr is not None
        assert attr["engine_id"] == eng_id
        assert expected_name_sub in attr["engine_name"]
        assert expected_sig_sub.lower() in attr["artifact_signature"].lower()
        assert body["av_sync"]["is_desynced"] is True
        assert body["av_sync"]["lag_ms"] == expected_lag
        seen_seams.add(body["visual_metrics"]["spatial_seam_score"])
        seen_lags.add(body["av_sync"]["lag_ms"])
        injected_speakers.append(body["speaker"])

    # Each visual engine has its own distinct forensic seam and lag profile
    assert len(seen_seams) == 5
    assert len(seen_lags) == 5

    # Clearing removes all 5 speakers (including Exec_Wav2Lip_Lip-Sync and Partner_LivePortrait_Puppet)
    r_clr = c.post("/api/generator/inject-video-test", json={"clear": True})
    assert r_clr.status_code == 200
    removed = r_clr.json()["removed"]
    for spk_key in injected_speakers:
        assert spk_key in removed
    q_after = c.get("/api/quality").json()
    for spk_key in injected_speakers:
        assert spk_key not in q_after


def test_widescreen_16_9_frames_and_malformed_inputs(railway_mod):
    import model_video

    det = model_video.VideoDeepfakeDetector()
    # 16:9 widescreen frames (e.g. 90x160 Google Meet tab capture)
    wide_nat = _encode_rgb_png(_make_natural_face_rgb(90, 160, seed=21))
    wide_swap = _encode_rgb_png(_make_faceswap_rgb(90, 160))

    res_nat = det.score_frame(wide_nat, speaker_id="wide_real", audio_rms=0.01)
    res_swap = det.score_frame(wide_swap, speaker_id="wide_swap", audio_rms=0.01)
    assert res_nat["verdict"] == "real"
    assert res_swap["verdict"] == "fake"
    assert res_swap["video_p"] >= 0.65

    # Malformed base64 / empty frames on Railway endpoint
    c = TestClient(railway_mod.app)
    r_bad = c.post("/api/video/score", json={"speaker": "Test", "frame_b64": "!!!not-base64!!!"})
    assert r_bad.status_code == 400

    # Clear via inject-video-test {clear: True}
    c.post("/api/generator/inject-video-test", json={"speaker_name": "Swap_Test", "is_real": False})
    r_clr = c.post("/api/generator/inject-video-test", json={"clear": True})
    assert r_clr.status_code == 200
    assert r_clr.json()["cleared"] is True
    assert "Swap_Test" in r_clr.json()["removed"]


def test_huggingface_vit_deepfake_backbone_and_ensemble(monkeypatch):
    import model_video

    bb = model_video.HuggingFaceViTDeepfakeBackbone(
        model_id="dima806/deepfake_vs_real_image_detection"
    )
    # 1. Verify label resolution across standard and edge-case HuggingFace deepfake ViT schemas
    bb._resolve_label_mapping({0: "Real", 1: "Fake"})
    assert bb._fake_indices == [1]
    assert bb._real_indices == [0]

    # Ensure substring 'ai' inside 'Portrait' or 'Plain' is not misclassified as Fake
    bb._resolve_label_mapping({0: "Real_Portrait", 1: "AI_Generated"})
    assert bb._fake_indices == [1]
    assert bb._real_indices == [0]

    # Ensure negated labels like 'Non-Fake' resolve to real
    bb._resolve_label_mapping({0: "Non-Fake", 1: "Fake"})
    assert bb._fake_indices == [1]
    assert bb._real_indices == [0]

    bb._resolve_label_mapping({0: "Deepfake", 1: "Realism"})
    assert bb._fake_indices == [0]
    assert bb._real_indices == [1]

    # 2. Wire a lightweight mock ViT processor & model to test neural inference + ensemble
    class _DummyOutput:
        def __init__(self, logits):
            self.logits = logits

    class _DummyViTModel:
        def __init__(self, fake_logit: float, real_logit: float):
            self.fake_logit = fake_logit
            self.real_logit = real_logit

        def __call__(self, **kwargs):
            # index 0 = Deepfake, index 1 = Realism
            return _DummyOutput(np.array([[self.fake_logit, self.real_logit]], dtype=np.float32))

    bb._processor = lambda images, return_tensors="pt": {"pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32)}
    bb._device = "cpu"
    bb.loaded_model_id = "dima806/deepfake_vs_real_image_detection"

    # Case A: ViT backbone detects a smooth-boundary diffusion/GAN deepfake (high neural P(Fake))
    bb._model = _DummyViTModel(fake_logit=3.2, real_logit=-1.5)
    det = model_video.VideoDeepfakeDetector(vit_backbone=bb)
    nat_png = _encode_rgb_png(_make_natural_face_rgb(96, 96, seed=19))
    res_fake = det.score_frame(nat_png, speaker_id="vit_diffusion_fake", audio_rms=0.01)

    assert res_fake["hf_loaded"] is True
    assert res_fake["hf_model_id"] == "dima806/deepfake_vs_real_image_detection"
    assert res_fake["vit_score"] is not None and res_fake["vit_score"] > 0.95
    assert res_fake["visual_metrics"]["vit_score"] == res_fake["vit_score"]
    assert res_fake["visual_metrics"]["hf_loaded"] is True
    assert res_fake["video_p"] >= 0.65
    assert res_fake["verdict"] == "fake"

    # Case B: ViT backbone confirms a real human face (low neural P(Fake))
    bb._model = _DummyViTModel(fake_logit=-2.8, real_logit=2.6)
    res_real = det.score_frame(nat_png, speaker_id="vit_real_user", audio_rms=0.01)
    assert res_real["hf_loaded"] is True
    assert res_real["vit_score"] is not None and res_real["vit_score"] < 0.05
    assert res_real["video_p"] < 0.45
    assert res_real["verdict"] == "real"

    # Case C: Runtime SONAVE_ENABLE_HF_VIDEO=0 disables neural scoring even after model is loaded
    monkeypatch.setenv("SONAVE_ENABLE_HF_VIDEO", "0")
    res_disabled = det.score_frame(nat_png, speaker_id="vit_disabled_user", audio_rms=0.01)
    assert res_disabled["hf_loaded"] is False
    assert res_disabled["vit_score"] is None
    assert res_disabled["visual_metrics"]["vit_score"] is None
    assert res_disabled["vit_ensemble_scores"] == {}
    assert res_disabled["vit_models_loaded"] == []


def test_multi_vit_visual_ensemble_loading_and_aggregation(monkeypatch, railway_mod):
    import model_video
    import transformers

    assert model_video.DEFAULT_HF_VIDEO_ENSEMBLE == (
        "dima806/deepfake_vs_real_image_detection",
        "prithivMLmods/Deep-Fake-Detector-v2-Model",
        "Wvolf/ViT_Deepfake_Detection",
        "umm-maybe/AI-image-detector",
    )

    # Verify env var configuration (SONAVE_VIDEO_HF_MODELS and SONAVE_VIDEO_HF_MODEL)
    monkeypatch.setenv("SONAVE_VIDEO_HF_MODELS", "model/a, model/b")
    bb_env = model_video.HuggingFaceViTDeepfakeBackbone()
    assert bb_env.model_ids == ["model/a", "model/b"]
    monkeypatch.delenv("SONAVE_VIDEO_HF_MODELS", raising=False)

    monkeypatch.setenv("SONAVE_VIDEO_HF_MODEL", "Wvolf/ViT_Deepfake_Detection")
    bb_single_env = model_video.HuggingFaceViTDeepfakeBackbone()
    assert bb_single_env.model_ids[0] == "Wvolf/ViT_Deepfake_Detection"
    assert len(bb_single_env.model_ids) == 4
    monkeypatch.delenv("SONAVE_VIDEO_HF_MODEL", raising=False)

    # Mock all 4 ensemble models with their real-world HuggingFace id2label schemas
    # and distinct specialist logits (e.g., diffusion detector catches an AI avatar)
    model_specs = {
        "dima806/deepfake_vs_real_image_detection": {
            "id2label": {0: "Real", 1: "Fake"},
            "logits": [-0.4, 1.2],  # P(Fake=idx 1) ~ 0.8320
            "image_size": 224,
        },
        "prithivMLmods/Deep-Fake-Detector-v2-Model": {
            "id2label": {0: "Realism", 1: "Deepfake"},
            "logits": [-1.8, 2.4],  # P(Deepfake=idx 1) ~ 0.9852
            "image_size": 224,
        },
        "Wvolf/ViT_Deepfake_Detection": {
            "id2label": {0: "Real", 1: "Fake"},
            "logits": [-0.8, 1.5],  # P(Fake=idx 1) ~ 0.9089
            "image_size": 224,
        },
        "umm-maybe/AI-image-detector": {
            "id2label": {0: "artificial", 1: "human"},
            "logits": [3.0, -2.0],  # P(artificial=idx 0) ~ 0.9933
            "image_size": [224, 224],
        },
    }

    class _MockConfig:
        def __init__(self, id2label, image_size):
            self.id2label = id2label
            self.image_size = image_size

    class _MockOutput:
        def __init__(self, logits):
            self.logits = np.asarray([logits], dtype=np.float32)

    class _MockHFModel:
        def __init__(self, spec):
            self.config = _MockConfig(spec["id2label"], spec["image_size"])
            self._logits = spec["logits"]

        def to(self, dev):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            assert "pixel_values" in kwargs
            return _MockOutput(self._logits)

    def _mock_model_from_pretrained(model_id, local_files_only=False, **kwargs):
        if model_id not in model_specs:
            raise OSError(f"Unknown model {model_id}")
        return _MockHFModel(model_specs[model_id])

    class _MockAutoImageProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            raise ImportError("AutoImageProcessor requires the Torchvision library")

    monkeypatch.setattr(
        transformers.AutoModelForImageClassification,
        "from_pretrained",
        _mock_model_from_pretrained,
    )
    monkeypatch.setattr(
        transformers,
        "AutoImageProcessor",
        _MockAutoImageProcessor,
    )
    monkeypatch.setenv("SONAVE_ENABLE_HF_VIDEO", "1")

    # Verify fallback_model_id is only loaded when configured primary model_ids fail
    bb_primary_ok = model_video.HuggingFaceViTDeepfakeBackbone(
        model_id="dima806/deepfake_vs_real_image_detection",
        fallback_model_id="prithivMLmods/Deep-Fake-Detector-v2-Model",
        device="cpu",
    )
    assert bb_primary_ok.load(allow_download=True) is True
    assert bb_primary_ok.loaded_model_ids == ["dima806/deepfake_vs_real_image_detection"]

    bb_primary_fail = model_video.HuggingFaceViTDeepfakeBackbone(
        model_id="nonexistent/missing-model",
        fallback_model_id="prithivMLmods/Deep-Fake-Detector-v2-Model",
        device="cpu",
    )
    assert bb_primary_fail.load(allow_download=True) is True
    assert bb_primary_fail.loaded_model_ids == ["prithivMLmods/Deep-Fake-Detector-v2-Model"]

    bb_ens = model_video.HuggingFaceViTDeepfakeBackbone(device="cpu")
    assert bb_ens.load(allow_download=True) is True
    assert bb_ens.is_loaded is True
    assert bb_ens.loaded_model_ids == list(model_video.DEFAULT_HF_VIDEO_ENSEMBLE)
    assert len(bb_ens._ensemble) == 4

    # Verify per-model label resolution: first 3 have fake=[1], umm-maybe has fake=[0]
    by_id = {e["model_id"]: e for e in bb_ens._ensemble}
    assert by_id["dima806/deepfake_vs_real_image_detection"]["fake_indices"] == [1]
    assert by_id["prithivMLmods/Deep-Fake-Detector-v2-Model"]["fake_indices"] == [1]
    assert by_id["Wvolf/ViT_Deepfake_Detection"]["fake_indices"] == [1]
    assert by_id["umm-maybe/AI-image-detector"]["fake_indices"] == [0]
    assert by_id["umm-maybe/AI-image-detector"]["real_indices"] == [1]

    det_ens = model_video.VideoDeepfakeDetector(vit_backbone=bb_ens)
    monkeypatch.setattr(model_video, "_VIDEO_DETECTOR", det_ens)

    warm_info = model_video.warm_hf_video_backbone(allow_download=True)
    assert warm_info["ok"] is True
    assert warm_info["hf_loaded"] is True
    assert warm_info["vit_models_loaded"] == list(model_video.DEFAULT_HF_VIDEO_ENSEMBLE)

    nat_png = _encode_rgb_png(_make_natural_face_rgb(96, 96, seed=31))
    res = det_ens.score_frame(nat_png, speaker_id="ens_speaker", audio_rms=0.01)

    assert res["hf_loaded"] is True
    assert res["vit_models_loaded"] == list(model_video.DEFAULT_HF_VIDEO_ENSEMBLE)
    ens_scores = res["vit_ensemble_scores"]
    assert set(ens_scores.keys()) == set(model_video.DEFAULT_HF_VIDEO_ENSEMBLE)
    assert ens_scores["umm-maybe/AI-image-detector"] > 0.99
    assert ens_scores["prithivMLmods/Deep-Fake-Detector-v2-Model"] > 0.98

    vals = list(ens_scores.values())
    expected_vit = round(float(np.clip(0.55 * np.mean(vals) + 0.45 * np.max(vals), 0.001, 0.999)), 4)
    assert res["vit_score"] == expected_vit
    assert res["visual_metrics"]["vit_ensemble_scores"] == ens_scores
    assert res["visual_metrics"]["vit_models_loaded"] == list(model_video.DEFAULT_HF_VIDEO_ENSEMBLE)
    assert res["verdict"] == "fake"

    # Verify Railway /api/video/score propagates vit_ensemble_scores and vit_models_loaded
    c = TestClient(railway_mod.app)
    r_api = c.post(
        "/api/video/score",
        json={"speaker": "Ensemble_Caller", "frame_b64": base64.b64encode(nat_png).decode(), "audio_rms": 0.01},
    )
    assert r_api.status_code == 200
    j_api = r_api.json()
    assert j_api["vit_ensemble_scores"] == ens_scores
    assert j_api["vit_models_loaded"] == list(model_video.DEFAULT_HF_VIDEO_ENSEMBLE)
    assert j_api["visual_metrics"]["vit_ensemble_scores"] == ens_scores


def test_meet_addon_never_hangs_on_ws_disconnect_or_vit_loading(railway_mod, monkeypatch):
    """Regression test: opening, streaming, and disconnecting /api/ws/mic-stream and
    /api/ws/video-stream when ACTIVE_STREAMS drops to 0 must never deadlock _STATE_LOCK
    or block GET /meet-addon and GET /api/quality (<50ms response time)."""
    import time
    import transformers
    import model_video

    c = TestClient(railway_mod.app)
    railway_mod.ACTIVE_STREAMS.clear()

    # 1. Connect and disconnect /api/ws/mic-stream when it is the sole active stream (1 -> 0)
    with c.websocket_connect("/api/ws/mic-stream?speaker=Host_Tester&space_id=live-room") as ws_mic:
        ws_mic.send_bytes(pcm16([0.15] * 1600))

    # Immediately verify GET /meet-addon and GET /api/quality respond in <50ms (no deadlock)
    t0 = time.perf_counter()
    r_addon = c.get("/meet-addon")
    dt_addon_ms = (time.perf_counter() - t0) * 1000.0
    assert r_addon.status_code == 200
    assert dt_addon_ms < 50.0, f"GET /meet-addon took {dt_addon_ms:.1f}ms"

    # Plain-HTTP loopback (http://127.0.0.1:8080) must NOT send upgrade-insecure-requests
    # or HSTS (which causes Chrome to send TLS ClientHello to plain-HTTP uvicorn)
    c_local = TestClient(railway_mod.app, base_url="http://127.0.0.1:8080")
    r_local_addon = c_local.get("/meet-addon")
    assert r_local_addon.status_code == 200
    assert "upgrade-insecure-requests" not in r_local_addon.headers.get("content-security-policy", "")
    assert "strict-transport-security" not in r_local_addon.headers

    t1 = time.perf_counter()
    r_qual = c.get("/api/quality?space_id=live-room")
    dt_qual_ms = (time.perf_counter() - t1) * 1000.0
    assert r_qual.status_code == 200
    assert dt_qual_ms < 50.0, f"GET /api/quality took {dt_qual_ms:.1f}ms"

    # 2. Connect and disconnect /api/ws/video-stream when it is the sole active stream (1 -> 0)
    railway_mod.ACTIVE_STREAMS.clear()
    nat_b64 = base64.b64encode(_encode_rgb_png(_make_natural_face_rgb(64, 64))).decode()
    with c.websocket_connect("/api/ws/video-stream?speaker=Host_Tester&space_id=live-room") as ws_vid:
        ws_vid.send_json({"speaker": "Host_Tester", "frame_b64": nat_b64, "audio_rms": 0.05})
        msg = ws_vid.receive_json()
        assert msg["video_verdict"] in ("real", "suspect", "fake")

    t2 = time.perf_counter()
    r_addon2 = c.get("/meet-addon")
    dt_addon2_ms = (time.perf_counter() - t2) * 1000.0
    assert r_addon2.status_code == 200
    assert dt_addon2_ms < 50.0, f"GET /meet-addon after video disconnect took {dt_addon2_ms:.1f}ms"

    # 3. Verify POST /api/meet/sessions/connect with ad-hoc space 'live-room' never blocks on external HTTP
    t3 = time.perf_counter()
    r_conn = c.post("/api/meet/sessions/connect", json={"space_id": "https://meet.google.com/live-room", "access_token": "ya29.dummy"})
    dt_conn_ms = (time.perf_counter() - t3) * 1000.0
    assert r_conn.status_code == 200
    assert r_conn.json()["ok"] is True
    assert dt_conn_ms < 50.0, f"POST /api/meet/sessions/connect took {dt_conn_ms:.1f}ms"

    # 4. Verify cold HuggingFaceViTDeepfakeBackbone.score() with SONAVE_FORCE_HF_VIDEO=1
    # genuinely spawns the background daemon thread and returns immediately (<50ms) even
    # when AutoModelForImageClassification.from_pretrained takes 200ms
    def _slow_from_pretrained(model_id, **kwargs):
        time.sleep(0.20)
        raise OSError("Simulated slow local cache miss")

    monkeypatch.setenv("SONAVE_FORCE_HF_VIDEO", "1")
    monkeypatch.setattr(
        transformers.AutoModelForImageClassification,
        "from_pretrained",
        _slow_from_pretrained,
    )
    bb_cold = model_video.HuggingFaceViTDeepfakeBackbone(model_id="nonexistent/slow-model")
    img = _make_natural_face_rgb(64, 64).astype(np.float32) / 255.0
    t4 = time.perf_counter()
    s_cold = bb_cold.score(img)
    dt_cold_ms = (time.perf_counter() - t4) * 1000.0
    assert s_cold["hf_loaded"] is False
    assert bb_cold._bg_loading is True
    assert dt_cold_ms < 50.0, f"Cold ViT score() blocked for {dt_conn_ms:.1f}ms while background load was active"


def test_live_webcam_face_swap_stream_and_instant_clear_recovery(railway_mod):
    """Verify that when the user has their webcam active (Cam Self-Check) and clicks
    Test Face-Swap:
      1. sampleAndSendVideoFrame sends simulated_attack=True + engine_id over /api/ws/video-stream
         so every 1-second webcam tick stays FAKE with full engine telemetry (never overwritten by REAL).
      2. The Host speaker row itself reflects the active webcam face-swap without spawning
         a duplicate dummy speaker card alongside a green Host card.
      3. Clicking Clear Face-Swap preserves the Host baseline row and the very next clean
         webcam frame immediately restores VIDEO AUTHENTIC (real) with no lingering fake window."""
    c = TestClient(railway_mod.app)

    # Verify meet-addon.html includes live webcam face-swap rendering, payload flags, and no-store header
    r_html = c.get("/meet-addon")
    assert r_html.status_code == 200
    assert "no-store" in r_html.headers.get("cache-control", "")
    assert "applyLiveFaceSwapEffect" in r_html.text
    assert "detectLiveFaceBox" in r_html.text
    assert "renderVideoPreviewFrame" in r_html.text
    assert "payload.simulated_attack = true" in r_html.text

    # 1. Connect meeting session -> seeds Host baseline
    r_conn = c.post("/api/meet/sessions/connect", json={"space_id": "https://meet.google.com/live-room"})
    assert r_conn.status_code == 200

    # 1b. If user clicked Test Face-Swap BEFORE starting Cam Self-Check, standalone CFO speaker exists
    r_pre = c.post(
        "/api/generator/inject-video-test",
        json={"speaker_name": "CFO", "engine_id": "deepfacelive", "is_real": False, "lag_ms": 184},
    )
    assert r_pre.status_code == 200
    assert "CFO" in c.get("/api/quality?space_id=live-room").json()

    nat_b64 = base64.b64encode(_encode_rgb_png(_make_natural_face_rgb(96, 96, seed=7, mouth_open=0.3))).decode()
    swap_b64 = base64.b64encode(_encode_rgb_png(_make_faceswap_rgb(96, 96))).decode()

    with c.websocket_connect("/api/ws/video-stream?speaker=Host&space_id=live-room") as ws_vid:
        # 2. Normal webcam frame -> REAL
        ws_vid.send_json({
            "speaker": "Host",
            "frame_b64": nat_b64,
            "audio_rms": 0.02,
            "mouth_delta": 0.022,
            "source": "webcam_self_check",
        })
        res_clean1 = ws_vid.receive_json()
        assert res_clean1["video_verdict"] == "real"
        assert res_clean1["video_p"] < 0.45

        q1 = c.get("/api/quality?space_id=live-room").json()
        assert "Host" in q1
        assert q1["Host"]["video_verdict"] == "real"
        assert q1["Host"]["auth_verdict"] == "real"

        # 3. Subsequent 1-second webcam ticks from sampleAndSendVideoFrame with _simVideoActive=True
        #    Must atomically purge standalone CFO_DeepFaceLive_Clone and flip Host to FAKE
        for _ in range(3):
            ws_vid.send_json({
                "speaker": "Host",
                "frame_b64": swap_b64,
                "audio_rms": 0.38,
                "mouth_delta": 0.0005,
                "simulated_attack": True,
                "engine_id": "deepfacelive",
                "target_video_p": 0.952,
                "vit_score": 0.94,
                "spatial_seam_score": 0.93,
                "temporal_jitter_score": 0.76,
                "lag_ms": 184,
                "source": "webcam_self_check",
            })
            res_sim = ws_vid.receive_json()
            assert res_sim["video_verdict"] == "fake"
            assert res_sim["video_p"] == pytest.approx(0.952, abs=0.01)
            assert res_sim["vit_score"] == pytest.approx(0.94, abs=0.01)
            assert res_sim["visual_metrics"]["spatial_seam_score"] == pytest.approx(0.93, abs=0.01)
            assert res_sim["visual_metrics"]["temporal_jitter_score"] == pytest.approx(0.76, abs=0.01)
            assert res_sim["av_sync"]["is_desynced"] is True
            assert res_sim["av_sync"]["lag_ms"] == 184
            assert res_sim["video_attribution"]["engine_id"] == "deepfacelive"

        # Verify /api/quality shows ONLY Host as FAKE (standalone CFO card was purged)
        q_sim = c.get("/api/quality?space_id=live-room").json()
        spk_keys = [k for k in q_sim.keys() if not k.startswith("_")]
        assert spk_keys == ["Host"]
        assert q_sim["Host"]["video_verdict"] == "fake"
        assert q_sim["Host"]["auth_verdict"] == "fake"
        assert q_sim["Host"]["auth_p"] >= 0.90
        assert q_sim["Host"]["video_attribution"]["engine_id"] == "deepfacelive"

        # Verify REST fallback /api/video/score also forwards vit_score and simulated_attack
        r_rest = c.post(
            "/api/video/score",
            json={
                "speaker_name": "Host",
                "space_id": "live-room",
                "frame_b64": swap_b64,
                "audio_rms": 0.38,
                "mouth_delta": 0.0005,
                "simulated_attack": True,
                "engine_id": "heygen_liveavatar",
                "target_video_p": 0.964,
                "vit_score": 0.95,
                "spatial_seam_score": 0.89,
                "temporal_jitter_score": 0.84,
                "lag_ms": 215,
            },
        )
        assert r_rest.status_code == 200
        rest_data = r_rest.json()
        assert rest_data["video_verdict"] == "fake"
        assert rest_data["vit_score"] == pytest.approx(0.95, abs=0.01)
        assert rest_data["video_attribution"]["engine_id"] == "heygen_liveavatar"

        # 5. User clicks Clear Face-Swap -> clears simulated video state but preserves Host row
        r_clr = c.post("/api/generator/inject-video-test", json={"clear": True})
        assert r_clr.status_code == 200
        assert "Host" in r_clr.json()["removed"]

        # Immediately send next clean webcam frame -> must immediately score REAL (no stale fake window)
        ws_vid.send_json({
            "speaker": "Host",
            "frame_b64": nat_b64,
            "audio_rms": 0.02,
            "mouth_delta": 0.024,
            "source": "webcam_self_check",
        })
        res_clean2 = ws_vid.receive_json()
        assert res_clean2["video_verdict"] == "real"
        assert res_clean2["video_p"] < 0.45
        assert res_clean2["av_sync"]["is_desynced"] is False

        q_after = c.get("/api/quality?space_id=live-room").json()
        assert "Host" in q_after
        assert q_after["Host"]["video_verdict"] == "real"
        assert q_after["Host"]["auth_verdict"] == "real"
        assert q_after["Host"]["baseline"] is True

    # 6. After closing /api/ws/video-stream (Stop Cam Check), active meeting session & Host row remain intact
    q_post_close = c.get("/api/quality?space_id=live-room").json()
    assert "Host" in q_post_close
    assert q_post_close["Host"]["baseline"] is True


def test_railway_standalone_root_dir_deploy_has_vendored_model_video(monkeypatch):
    """Regression test for Railway production deployment where Root Directory = 'railway'
    (so ../src does not exist inside the container). Verifies:
      1. railway/model_video.py exists, matches src/model_video.py, and loads standalone when src/ is absent.
      2. HuggingFaceViTDeepfakeBackbone.is_enabled() defaults to False inside a Railway CPU
         container (RAILWAY_ENVIRONMENT or RAILWAY_PUBLIC_DOMAIN set) unless SONAVE_ENABLE_HF_VIDEO=1
         or SONAVE_FORCE_LOCAL_HF_VIDEO=1 is explicitly set.
      3. railway/app.py imports cleanly without src/ on sys.path, sets SONAVE_ENABLE_HF_VIDEO=0 on Railway,
         and scores video frames end-to-end."""
    from pathlib import Path
    import os
    import sys

    root = Path(__file__).resolve().parent.parent
    src_mv_path = root / "src" / "model_video.py"
    rw_mv_path = root / "railway" / "model_video.py"
    assert rw_mv_path.is_file(), "railway/model_video.py must be vendored for Railway Root Directory deploys"
    assert rw_mv_path.read_text(encoding="utf-8").strip() == src_mv_path.read_text(encoding="utf-8").strip(), (
        "railway/model_video.py must stay in sync with src/model_video.py"
    )

    # Load railway/model_video.py directly
    rw_mv = load_module("railway_vendored_model_video", "railway/model_video.py")
    monkeypatch.delenv("SONAVE_ENABLE_HF_VIDEO", raising=False)
    monkeypatch.delenv("SONAVE_FORCE_LOCAL_HF_VIDEO", raising=False)
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    assert rw_mv.HuggingFaceViTDeepfakeBackbone.is_enabled() is False

    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "sonave.up.railway.app")
    assert rw_mv.HuggingFaceViTDeepfakeBackbone.is_enabled() is False

    monkeypatch.setenv("SONAVE_FORCE_LOCAL_HF_VIDEO", "1")
    assert rw_mv.HuggingFaceViTDeepfakeBackbone.is_enabled() is True
    monkeypatch.delenv("SONAVE_FORCE_LOCAL_HF_VIDEO", raising=False)

    monkeypatch.setenv("SONAVE_ENABLE_HF_VIDEO", "1")
    assert rw_mv.HuggingFaceViTDeepfakeBackbone.is_enabled() is True

    monkeypatch.delenv("SONAVE_ENABLE_HF_VIDEO", raising=False)
    monkeypatch.delenv("RAILWAY_PUBLIC_DOMAIN", raising=False)
    assert rw_mv.HuggingFaceViTDeepfakeBackbone.is_enabled() is True

    # Simulate Railway container where only railway/ is on sys.path and model_video is not pre-imported from src/
    saved_mv = sys.modules.pop("model_video", None)
    src_resolved = (root / "src").resolve()
    orig_is_dir = Path.is_dir
    saved_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if Path(p).resolve() != src_resolved]
        monkeypatch.setattr(
            Path,
            "is_dir",
            lambda self: False if self.resolve() == src_resolved else orig_is_dir(self),
        )
        monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "sonave.up.railway.app")
        monkeypatch.delenv("SONAVE_ENABLE_HF_VIDEO", raising=False)
        rw_isolated = load_module("rwapp_isolated", "railway/app.py")
        assert rw_isolated.model_video is not None
        assert Path(rw_isolated.model_video.__file__).resolve() == rw_mv_path.resolve()
        assert os.environ.get("SONAVE_ENABLE_HF_VIDEO") == "0"
        nat_png = _encode_rgb_png(_make_natural_face_rgb(64, 64, seed=7))
        entry = rw_isolated._score_video_and_store("admin", "IsolatedSpk", nat_png, audio_rms=0.02)
        assert entry["video_verdict"] == "real"
        assert entry["hf_loaded"] is False
    finally:
        sys.path = saved_path
        if saved_mv is not None:
            sys.modules["model_video"] = saved_mv


def test_synthesize_visual_deepfake_all_five_engines_genuine_detection():
    """Verify synthesize_visual_deepfake() applies real pixel-level visual deepfake
    transformations for all 5 engines and that VideoDeepfakeDetector.score_frame()
    genuinely detects every transformed frame as FAKE (video_p >= 0.65) without
    relying on simulated_attack=True score overrides."""
    import model_video
    from railway import attribution

    det = model_video.VideoDeepfakeDetector()
    model_video.reset_visual_synthesis_state()

    nat = _make_natural_face_rgb(128, 128, seed=42, mouth_open=0.22)
    res_nat = det.score_frame(nat, speaker_id="real_cam_user", audio_rms=0.015)
    assert res_nat["verdict"] == "real"
    assert res_nat["video_p"] < 0.45

    engines = [
        "deepfacelive",
        "facefusion_simswap",
        "heygen_liveavatar",
        "wav2lip_retalking",
        "hallo_liveportrait",
    ]
    for idx, eng_id in enumerate(engines):
        spk = f"synth_{eng_id}"
        # Run 2 frames so stateful buffers (e.g. heygen_liveavatar mouth history & jitter) populate
        out1 = model_video.synthesize_visual_deepfake(nat, eng_id, speaker_id=spk)
        nat2 = _make_natural_face_rgb(128, 128, seed=42 + idx + 1, mouth_open=0.55)
        out2 = model_video.synthesize_visual_deepfake(nat2, eng_id, speaker_id=spk)

        assert out2.shape == nat2.shape
        assert out2.dtype == np.uint8
        mean_diff = float(np.abs(out2.astype(np.float32) - nat2.astype(np.float32)).mean())
        assert mean_diff > 1.0, f"{eng_id} did not visibly transform frame pixels"

        if eng_id == "wav2lip_retalking":
            # Upper face (forehead/eyes) must remain untouched; only lower-face mouth box is inpainted
            upper_h = int(128 * 0.45)
            assert np.array_equal(out2[:upper_h], nat2[:upper_h])
            lower_diff = float(np.abs(out2[int(128 * 0.56):int(128 * 0.84)].astype(np.float32) - nat2[int(128 * 0.56):int(128 * 0.84)].astype(np.float32)).mean())
            assert lower_diff > 3.5

        # Score genuinely via VideoDeepfakeDetector (no simulated_attack override)
        det.score_frame(out1, speaker_id=spk, audio_rms=0.24)
        res = det.score_frame(out2, speaker_id=spk, audio_rms=0.28)
        assert res["verdict"] == "fake", f"Expected fake verdict for {eng_id}, got {res}"
        assert res["video_p"] >= 0.65, f"Expected video_p >= 0.65 for {eng_id}, got {res['video_p']}"

        # Verify attribution from real extracted visual_metrics (without engine_hint)
        attr_no_hint = attribution.attribute_visual_engine(
            res["video_p"], visual_metrics=res["visual_metrics"]
        )
        assert attr_no_hint is not None
        assert attr_no_hint["engine_id"] == eng_id

    # Verify 320x240 (h=240, w=320) JPEG webcam frames with quiet mic (audio_rms=0.02) on a SINGLE frame
    nat_cam = _make_natural_face_rgb(240, 320, seed=77, mouth_open=0.15)
    jpeg_b64 = model_video.encode_frame_b64(nat_cam, fmt="jpeg")
    jpeg_bytes = base64.b64decode(jpeg_b64.split(",", 1)[1])
    res_cam_nat = det.score_frame(jpeg_bytes, speaker_id="cam_320x240_nat", audio_rms=0.02)
    assert res_cam_nat["verdict"] == "real"
    assert res_cam_nat["video_p"] < 0.45

    blind_mismatches = []
    for eng_id in engines:
        res_1f = det.score_frame(
            jpeg_bytes,
            speaker_id=f"cam_1f_{eng_id}",
            audio_rms=0.02,
            client_metrics={"apply_deepfake_model": eng_id},
        )
        assert res_1f["verdict"] == "fake", f"320x240 1-frame {eng_id} expected fake, got {res_1f}"
        assert res_1f["video_p"] >= 0.65
        vm_1f_blind = {k: v for k, v in res_1f["visual_metrics"].items() if k != "engine_hint"}
        attr_1f = attribution.attribute_visual_engine(
            res_1f["video_p"], visual_metrics=vm_1f_blind
        )
        assert attr_1f is not None
        if attr_1f["engine_id"] != eng_id:
            blind_mismatches.append((eng_id, attr_1f["engine_id"], vm_1f_blind))
    assert not blind_mismatches, f"Blind 320x240 mismatches: {blind_mismatches}"

    # Verify tiny and non-square frames (6x6, 8x8, 14x14, 16x16, 24x64) never raise broadcast errors
    for th, tw in [(6, 6), (8, 8), (14, 14), (16, 16), (24, 64)]:
        tiny = _make_natural_face_rgb(th, tw, seed=19)
        for eng_id in engines:
            t_out = model_video.synthesize_visual_deepfake(tiny, eng_id, speaker_id=f"tiny_{th}x{tw}")
            assert t_out.shape == (th, tw, 3)


def test_live_webcam_apply_deepfake_model_ws_and_rest_without_simulated_override(railway_mod):
    """Verify that sending a clean webcam frame with apply_deepfake_model (and
    simulated_attack NOT set) over /api/ws/video-stream and /api/video/score:
      1. Runs synthesize_visual_deepfake() on the user's frame and returns transformed_frame_b64.
      2. Genuinely scores the transformed frame via VideoDeepfakeDetector as FAKE (video_p >= 0.65)
         with simulated=False and deepfake_filter_active=True.
      3. Immediately recovers to REAL when the user stops the deepfake filter on the next frame."""
    import model_video

    c = TestClient(railway_mod.app)

    # Check UI updates in /meet-addon and /console: no cartoon eye/wireframe drawing, new button labels & larger canvas
    r_html = c.get("/meet-addon")
    assert r_html.status_code == 200
    assert "Apply Deepfake to My Cam" in r_html.text
    assert "Stop Deepfake Filter" in r_html.text
    assert "payload.apply_deepfake_model = eng" in r_html.text
    assert "res.transformed_frame_b64" in r_html.text
    assert "rgba(20, 28, 36, 0.78)" not in r_html.text  # cartoon eye scleras removed

    r_con = c.get("/console")
    assert r_con.status_code == 200
    assert "Run Visual Deepfake" in r_con.text
    assert "Stop Deepfake Filter" in r_con.text
    assert "videoCaptureCanvas" in r_con.text
    assert "payload.apply_deepfake_model = eng" in r_con.text

    c.post("/api/meet/sessions/connect", json={"space_id": "https://meet.google.com/cam-self-filter"})
    nat_rgb = _make_natural_face_rgb(128, 128, seed=31, mouth_open=0.25)
    nat_b64 = base64.b64encode(_encode_rgb_png(nat_rgb)).decode()

    with c.websocket_connect("/api/ws/video-stream?speaker=Host&space_id=cam-self-filter") as ws_vid:
        # 1. Clean webcam frame -> REAL
        ws_vid.send_json({
            "speaker": "Host",
            "frame_b64": nat_b64,
            "audio_rms": 0.02,
            "mouth_delta": 0.022,
            "source": "webcam_self_check",
        })
        res_clean = ws_vid.receive_json()
        assert res_clean["video_verdict"] == "real"
        assert res_clean["video_p"] < 0.45
        assert res_clean["transformed_frame_b64"] is None

        # 2. User activates "Apply Deepfake to My Cam" -> sends raw webcam frame + apply_deepfake_model
        #    Note: simulated_attack is NOT set! Detection is 100% genuine on the synthesized pixels.
        for eng_id in ("deepfacelive", "heygen_liveavatar", "wav2lip_retalking", "facefusion_simswap"):
            ws_vid.send_json({
                "speaker": "Host",
                "frame_b64": nat_b64,
                "audio_rms": 0.26,
                "mouth_delta": 0.001,
                "apply_deepfake_model": eng_id,
                "source": "webcam_self_check",
            })
            res_df = ws_vid.receive_json()
            assert res_df["video_verdict"] == "fake", f"Expected fake for {eng_id}, got {res_df}"
            assert res_df["video_p"] >= 0.65
            assert res_df["simulated"] is False
            assert res_df["deepfake_filter_active"] is True
            assert res_df["applied_deepfake_model"] == eng_id
            assert res_df["transformed_frame_b64"] is not None
            assert res_df["transformed_frame_b64"].startswith(
                ("data:image/jpeg;base64,", "data:image/png;base64,")
            )
            assert res_df["video_attribution"]["engine_id"] == eng_id

            # Decode returned transformed_frame_b64 and confirm it is a valid modified RGB frame
            tf_raw = base64.b64decode(res_df["transformed_frame_b64"].split(",", 1)[1])
            tf_arr = model_video.decode_frame(tf_raw) * 255.0
            assert tf_arr.shape == nat_rgb.shape
            assert float(np.abs(tf_arr - nat_rgb.astype(np.float32)).mean()) > 1.0

        # 3. User clicks "Stop Deepfake Filter" -> next frame has no apply_deepfake_model
        #    Must immediately recover to REAL without needing a separate reset call
        ws_vid.send_json({
            "speaker": "Host",
            "frame_b64": nat_b64,
            "audio_rms": 0.02,
            "mouth_delta": 0.024,
            "source": "webcam_self_check",
        })
        res_rec = ws_vid.receive_json()
        assert res_rec["video_verdict"] == "real"
        assert res_rec["video_p"] < 0.45
        assert res_rec["deepfake_filter_active"] is False
        assert res_rec["transformed_frame_b64"] is None

    # 4. Also verify REST /api/video/score with apply_deepfake_model
    r_rest = c.post(
        "/api/video/score",
        json={
            "speaker": "Host",
            "space_id": "cam-self-filter",
            "frame_b64": nat_b64,
            "audio_rms": 0.25,
            "mouth_delta": 0.001,
            "apply_deepfake_model": "hallo_liveportrait",
        },
    )
    assert r_rest.status_code == 200
    body_rest = r_rest.json()
    assert body_rest["video_verdict"] == "fake"
    assert body_rest["video_p"] >= 0.65
    assert body_rest["simulated"] is False
    assert body_rest["deepfake_filter_active"] is True
    assert body_rest["applied_deepfake_model"] == "hallo_liveportrait"
    assert body_rest["transformed_frame_b64"].startswith(
        ("data:image/jpeg;base64,", "data:image/png;base64,")
    )
    assert body_rest["video_attribution"]["engine_id"] == "hallo_liveportrait"


def test_real_webcam_false_positive_prevention_and_forensic_corroboration(monkeypatch, railway_mod):
    """Regression test for real indoor low-res webcam false-positive prevention:
    1. Specialist face-deepfake ViTs (dima806 & Wvolf < 0.01) prevent out-of-domain AI-art
       models (prithivMLmods & umm-maybe ~ 0.78) from triggering a false positive.
    2. Dark lower-face beard/shadow (mouth_lum_mean < 0.72 * upper_lum_mean) and high
       hf_peak_ratio (ceiling fan / background blades) do not inflate perioral or outer seam scores.
    3. Forensic corroboration gating dampens uncorroborated ViT spikes (e.g. ViT=0.85 with
       Seam=0.13, Jitter=0.02, AV=0.96) to REAL (< 0.25) in both score_frame() and _score_video_and_store().
    4. All 5 visual deepfake synthesis engines applied to a bearded/indoor webcam frame still
       genuinely score FAKE (video_p >= 0.65)."""
    from pathlib import Path
    import model_video
    import transformers

    # 1. Test ViT ensemble specialist weighting when dima806 & Wvolf confirm REAL (<0.01)
    #    while prithivMLmods & umm-maybe false-positive (~0.76-0.78) on low-res webcam JPEG
    webcam_specs = {
        "dima806/deepfake_vs_real_image_detection": {
            "id2label": {0: "Real", 1: "Fake"},
            "logits": [3.01, -3.01],  # P(Fake) ~ 0.0024
            "image_size": 224,
        },
        "prithivMLmods/Deep-Fake-Detector-v2-Model": {
            "id2label": {0: "Realism", 1: "Deepfake"},
            "logits": [-0.566, 0.566],  # P(Deepfake) ~ 0.7562
            "image_size": 224,
        },
        "Wvolf/ViT_Deepfake_Detection": {
            "id2label": {0: "Real", 1: "Fake"},
            "logits": [2.90, -2.90],  # P(Fake) ~ 0.0030
            "image_size": 224,
        },
        "umm-maybe/AI-image-detector": {
            "id2label": {0: "artificial", 1: "human"},
            "logits": [0.636, -0.636],  # P(artificial) ~ 0.7811
            "image_size": [224, 224],
        },
    }

    class _Cfg:
        def __init__(self, id2label, image_size):
            self.id2label = id2label
            self.image_size = image_size

    class _Out:
        def __init__(self, logits):
            self.logits = np.asarray([logits], dtype=np.float32)

    class _Mdl:
        def __init__(self, spec):
            self.config = _Cfg(spec["id2label"], spec["image_size"])
            self._logits = spec["logits"]

        def to(self, dev):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return _Out(self._logits)

    monkeypatch.setattr(
        transformers.AutoModelForImageClassification,
        "from_pretrained",
        lambda mid, **kw: _Mdl(webcam_specs[mid]),
    )
    monkeypatch.setenv("SONAVE_ENABLE_HF_VIDEO", "1")

    bb_cam = model_video.HuggingFaceViTDeepfakeBackbone(device="cpu")
    assert bb_cam.load(allow_download=True) is True

    # Build a realistic indoor webcam frame with a dark beard/shadow on the lower face
    bearded_cam = _make_natural_face_rgb(160, 210, seed=108, mouth_open=0.1)
    # Darken lower-face beard region (so mouth_lum_mean < 0.72 * upper_lum_mean)
    bearded_cam[92:134, 72:138] = (bearded_cam[92:134, 72:138].astype(np.float32) * 0.35).astype(np.uint8)

    det_cam = model_video.VideoDeepfakeDetector(vit_backbone=bb_cam)
    res_bearded = det_cam.score_frame(bearded_cam, speaker_id="bearded_host", audio_rms=0.02)
    assert res_bearded["verdict"] == "real", f"Bearded real webcam frame scored fake: {res_bearded}"
    assert res_bearded["video_p"] < 0.20
    assert res_bearded["vit_score"] is not None and res_bearded["vit_score"] < 0.15

    # Verify all 5 visual deepfake synthesis engines on bearded_cam genuinely score FAKE (>= 0.65)
    det_no_vit = model_video.VideoDeepfakeDetector()
    for eng_id in (
        "deepfacelive",
        "facefusion_simswap",
        "heygen_liveavatar",
        "wav2lip_retalking",
        "hallo_liveportrait",
    ):
        r_syn = det_no_vit.score_frame(
            bearded_cam,
            speaker_id=f"bearded_syn_{eng_id}",
            audio_rms=0.02,
            client_metrics={"apply_deepfake_model": eng_id},
        )
        assert r_syn["verdict"] == "fake", f"Expected fake for {eng_id} on bearded webcam, got {r_syn}"
        assert r_syn["video_p"] >= 0.65

    # If user's cropped webcam thumbnail exists on disk, verify it and all 5 synthesized engines directly
    derek_thumb = Path(r"C:\Users\Derek\.gemini\antigravity\brain\51b1814f-2a59-4d80-9afd-49fd27ed0d73\scratch\derek_thumb.png")
    if derek_thumb.is_file():
        raw_thumb = model_video.decode_frame(derek_thumb.read_bytes())
        crop_thumb = (raw_thumb[8:-8, 8:-8] * 255.0).clip(0, 255).astype(np.uint8)
        r_derek = det_cam.score_frame(crop_thumb, speaker_id="derek_real", audio_rms=0.02)
        assert r_derek["verdict"] == "real", f"derek_thumb scored fake: {r_derek}"
        assert r_derek["video_p"] < 0.20
        assert r_derek["vit_score"] < 0.15
        for eng_id in (
            "deepfacelive",
            "facefusion_simswap",
            "heygen_liveavatar",
            "wav2lip_retalking",
            "hallo_liveportrait",
        ):
            r_dsyn = det_no_vit.score_frame(
                crop_thumb,
                speaker_id=f"derek_syn_{eng_id}",
                audio_rms=0.02,
                client_metrics={"apply_deepfake_model": eng_id},
            )
            assert r_dsyn["verdict"] == "fake", f"derek_thumb {eng_id} expected fake, got {r_dsyn}"
            assert r_dsyn["video_p"] >= 0.65

    # 2. Test forensic corroboration gating when ViT=0.85 but Seam=0.13, Jitter=0.02, AV=0.96
    det_spike = model_video.VideoDeepfakeDetector()
    monkeypatch.setattr(
        det_spike.vit_backbone,
        "score",
        lambda img, fi=None: {
            "vit_score": 0.85,
            "vit_ensemble_scores": {"custom/vit": 0.85},
            "vit_models_loaded": ["custom/vit"],
            "hf_loaded": True,
            "hf_model_id": "custom/vit",
            "device": "cpu",
        },
    )
    monkeypatch.setattr(
        det_spike.spatial_net,
        "analyze",
        lambda img, fi=None: {
            "spatial_score": 0.13,
            "spatial_seam_score": 0.13,
            "frequency_seam_score": 0.11,
            "perioral_seam_score": 0.04,
            "skin_smoothing_score": 0.06,
            "boundary_seam_ratio": 1.12,
            "mouth_seam_ratio": 1.04,
            "hf_energy_ratio": 0.05,
            "chroma_step": 0.015,
        },
    )
    monkeypatch.setattr(
        det_spike.temporal_tracker,
        "update",
        lambda spk, img, fi=None, ts=None: {
            "temporal_jitter_score": 0.02,
            "mouth_delta": 0.025,
            "face_flicker": 0.005,
            "centroid_jitter": 0.002,
            "frames_tracked": 4,
        },
    )
    monkeypatch.setattr(
        det_spike.av_analyzer,
        "analyze",
        lambda speaker_id, mouth_delta, audio_rms=None, client_av_lag_ms=None: {
            "av_sync_score": 0.96,
            "sync_score": 0.96,
            "sync_coherence": 0.96,
            "av_mismatch_score": 0.04,
            "lag_ms": 28.0,
            "is_desynced": False,
            "desync_flag": False,
            "status": "AV Lip-Sync Coherent",
        },
    )
    res_gated = det_spike.score_frame(bearded_cam, speaker_id="gated_user", audio_rms=0.02)
    assert res_gated["verdict"] == "real", f"Uncorroborated ViT=0.85 spike must be gated to real: {res_gated}"
    assert res_gated["video_p"] < 0.25
    assert res_gated["vit_score"] < 0.35

    # 3. Also verify _score_video_and_store in railway/app.py dampens remote SCORER_URL spike
    #    (video_p=0.71, vit_score=0.85, Seam=0.13, Jitter=0.02, AV=0.96 -> real < 0.25)
    monkeypatch.setattr(
        railway_mod.model_video,
        "score_video_bytes",
        lambda *a, **kw: {
            "video_p": 0.71,
            "verdict": "fake",
            "face_detected": True,
            "bbox": [20, 20, 120, 120],
            "vit_score": 0.85,
            "vit_ensemble_scores": {
                "dima806/deepfake_vs_real_image_detection": 0.0024,
                "Wvolf/ViT_Deepfake_Detection": 0.0030,
                "prithivMLmods/Deep-Fake-Detector-v2-Model": 0.7564,
                "umm-maybe/AI-image-detector": 0.7812,
            },
            "vit_models_loaded": list(model_video.DEFAULT_HF_VIDEO_ENSEMBLE),
            "hf_model_id": "dima806/deepfake_vs_real_image_detection",
            "hf_loaded": True,
            "spatial_seam_score": 0.13,
            "temporal_jitter_score": 0.02,
            "av_sync_score": 0.96,
            "visual_metrics": {
                "vit_score": 0.85,
                "spatial_seam_score": 0.13,
                "temporal_jitter_score": 0.02,
                "av_sync_score": 0.96,
            },
            "av_sync": {"sync_score": 0.96, "lag_ms": 28.0, "desync_flag": False},
            "latency_ms": 18.0,
            "model_version": "sonave-vit-freq-v1",
        },
    )
    railway_mod.VIDEO_STATE.clear()
    entry_rw = railway_mod._score_video_and_store(
        "admin", "Host", _encode_rgb_png(bearded_cam), audio_rms=0.02
    )
    assert entry_rw["video_verdict"] == "real", f"Expected real verdict in _score_video_and_store, got {entry_rw}"
    assert entry_rw["video_p"] < 0.25
    assert entry_rw["vit_score"] < 0.15

