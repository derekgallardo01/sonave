"""
model_video.py — Multimodal Video & Audio-Visual Deepfake Detector for Google Meet.

Detects real-time video deepfakes, face-swaps, and neural avatars across four axes:
  0. Pretrained Vision Transformer Neural Classifier (HuggingFaceViTDeepfakeBackbone):
     - HuggingFace `AutoImageProcessor` + `AutoModelForImageClassification`
       (default `dima806/deepfake_vs_real_image_detection`, fallback
       `prithivMLmods/Deep-Fake-Detector-v2-Model`) scored on the detected face crop.
  1. Spatial & Frequency Residual Analysis (SpatialFrequencyResidualNet):
     - 2D FFT/DCT high-frequency spectrum anomalies & GAN/diffusion upsampling grid artifacts
     - Peri-oral and jawline blending mask boundary seams (DeepFaceLive, InsightFace, SimSwap)
     - Unnatural skin micro-texture smoothing vs. background sensor noise (GFPGAN / CodeFormer)
  2. Temporal Consistency & Liveness Tracking (TemporalConsistencyTracker):
     - Inter-frame face centroid / landmark micro-jitter across a sliding per-speaker window
     - High-frequency luminance/chrominance flicker in the inner face vs. outer background
     - Frame-freeze / re-synthesis stutter detection
  3. Audio-Visual Lip-Sync Coherence (AudioVisualSyncAnalyzer):
     - Cross-modal correlation between lower-face/mouth motion energy (mouth_delta) and
       concurrent acoustic RMS energy (audio_rms) to catch voice-over puppets & lagging avatars.

Designed to run both on the Modal GPU service (with PyTorch + HuggingFace ViT acceleration)
and on the CPU-only Railway control plane (with lazy HF ViT loading and pure NumPy + Pillow fallback).
"""
from __future__ import annotations

import io
import logging
import math
import os
import struct
import threading
import time
import zlib
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger("sonave.video")

MODEL_VIDEO_VERSION = "sonave-vit-freq-v1"
DEFAULT_HF_VIDEO_MODEL = "dima806/deepfake_vs_real_image_detection"
FALLBACK_HF_VIDEO_MODEL = "prithivMLmods/Deep-Fake-Detector-v2-Model"
DEFAULT_HF_VIDEO_ENSEMBLE: tuple[str, ...] = (
    "dima806/deepfake_vs_real_image_detection",
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "Wvolf/ViT_Deepfake_Detection",
    "umm-maybe/AI-image-detector",
)
TAU_VIDEO_REAL = 0.45
TAU_VIDEO_FAKE = 0.65


def video_verdict(p_video_fake: float) -> str:
    """Tri-state verdict for video deepfake probability."""
    if p_video_fake >= TAU_VIDEO_FAKE:
        return "fake"
    if p_video_fake >= TAU_VIDEO_REAL:
        return "suspect"
    return "real"


def _decode_minimal_png(data: bytes) -> np.ndarray | None:
    """Pure-stdlib PNG decoder for 8-bit RGB/RGBA/Grayscale non-interlaced PNGs."""
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    pos = 8
    width = height = bit_depth = color_type = 0
    idat_chunks: list[bytes] = []
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        chunk_type = data[pos + 4:pos + 8]
        chunk_data = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if chunk_type == b"IHDR" and len(chunk_data) >= 13:
            width, height, bit_depth, color_type, comp, filt, interlace = struct.unpack(">IIBBBBB", chunk_data[:13])
            if bit_depth != 8 or interlace != 0:
                return None
        elif chunk_type == b"IDAT":
            idat_chunks.append(chunk_data)
        elif chunk_type == b"IEND":
            break
    if width <= 0 or height <= 0 or not idat_chunks:
        return None
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
    if not channels:
        return None
    try:
        raw = zlib.decompress(b"".join(idat_chunks))
    except Exception:
        return None
    stride = width * channels
    expected = height * (1 + stride)
    if len(raw) < expected:
        return None
    out = np.zeros((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)
    idx = 0
    for y in range(height):
        ftype = raw[idx]
        row = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=idx + 1).astype(np.int32)
        idx += 1 + stride
        if ftype == 1:  # Sub
            for x in range(channels, stride):
                row[x] = (row[x] + row[x - channels]) & 0xFF
        elif ftype == 2:  # Up
            row = (row + prev) & 0xFF
        elif ftype == 3:  # Average
            for x in range(stride):
                a = row[x - channels] if x >= channels else 0
                b = int(prev[x])
                row[x] = (row[x] + ((a + b) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for x in range(stride):
                a = int(row[x - channels]) if x >= channels else 0
                b = int(prev[x])
                c = int(prev[x - channels]) if x >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                row[x] = (row[x] + pr) & 0xFF
        out[y] = row.astype(np.uint8)
        prev = row
    img = out.reshape((height, width, channels))
    if channels == 1:
        img = np.repeat(img, 3, axis=2)
    elif channels == 2:
        img = np.repeat(img[:, :, :1], 3, axis=2)
    elif channels == 4:
        img = img[:, :, :3]
    return img.astype(np.float32) / 255.0


def decode_frame(frame: bytes | np.ndarray) -> np.ndarray:
    """Decode a video frame (JPEG/PNG/WebP/BMP bytes or numpy array) to float32 RGB [0, 1] of shape (H, W, 3)."""
    if isinstance(frame, np.ndarray):
        arr = frame.astype(np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.max() > 1.05:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)

    if not isinstance(frame, (bytes, bytearray, memoryview)) or len(frame) == 0:
        raise ValueError("Empty or invalid video frame payload")

    raw_bytes = bytes(frame)

    # 1. Try Pillow (handles JPEG, PNG, WebP, BMP, etc.)
    try:
        from PIL import Image
        with Image.open(io.BytesIO(raw_bytes)) as im:
            im_rgb = im.convert("RGB")
            if im_rgb.width > 320 or im_rgb.height > 320:
                im_rgb = im_rgb.resize((256, 256))
            return np.asarray(im_rgb, dtype=np.float32) / 255.0
    except Exception:
        pass

    # 2. Try pure-stdlib PNG decoder
    png_arr = _decode_minimal_png(raw_bytes)
    if png_arr is not None:
        return png_arr

    # 3. Deterministic byte-array reconstruction for synthetic/raw frame buffers in tests
    n = len(raw_bytes)
    if n < 16:
        raise ValueError("Frame payload too short to decode")
    buf = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) / 255.0
    side = int(math.sqrt(max(48, n // 3)))
    side = max(16, min(128, side))
    needed = side * side * 3
    if len(buf) < needed:
        buf = np.resize(buf, needed)
    else:
        buf = buf[:needed]
    return buf.reshape((side, side, 3))


def detect_face_region(img: np.ndarray) -> dict[str, Any]:
    """Locate the primary participant face box and lower-face/mouth ROI.

    Uses YCbCr skin-tone likelihood combined with central portrait weighting and
    interior facial edge density (eyes/nose/mouth contours).
    """
    h, w, _ = img.shape
    if h < 12 or w < 12:
        return {
            "face_detected": False,
            "bbox": (0, 0, h, w),
            "mouth_roi": (h // 2, w // 4, h, (3 * w) // 4),
            "face_coverage": 0.0,
        }

    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    y_lum = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.564 * (b - y_lum) + 0.5
    cr = 0.713 * (r - y_lum) + 0.5

    # Broad human skin chrominance envelope + portrait center prior
    skin_mask = (
        (cr >= 0.51) & (cr <= 0.68)
        & (cb >= 0.34) & (cb <= 0.50)
        & (y_lum >= 0.12) & (y_lum <= 0.95)
    ).astype(np.float32)

    # Spatial gradients inside the central portrait zone
    gy = np.abs(np.diff(y_lum, axis=0, prepend=y_lum[:1, :]))
    gx = np.abs(np.diff(y_lum, axis=1, prepend=y_lum[:, :1]))
    grad = np.hypot(gx, gy)

    cy0, cy1 = int(h * 0.15), int(h * 0.85)
    cx0, cx1 = int(w * 0.20), int(w * 0.80)
    center_grad = float(np.mean(grad[cy0:cy1, cx0:cx1])) if cy1 > cy0 and cx1 > cx0 else 0.0

    # 5x5 spatial box filter on skin_mask to reject isolated background noise pixels
    pad_s = np.pad(skin_mask, 2, mode="edge")
    smooth_skin = np.zeros_like(skin_mask)
    for dy in range(5):
        for dx in range(5):
            smooth_skin += pad_s[dy:dy + h, dx:dx + w]
    smooth_skin /= 25.0
    dense_skin = smooth_skin >= 0.55

    skin_frac = float(np.mean(dense_skin[cy0:cy1, cx0:cx1])) if cy1 > cy0 and cx1 > cx0 else 0.0
    face_detected = bool(skin_frac >= 0.06 or center_grad >= 0.015)

    if np.any(dense_skin):
        ys, xs = np.where(dense_skin)
        # Expand by 2px to undo the 5x5 box-filter erosion so bbox sits on the actual face-to-background boundary
        y0 = max(0, int(np.percentile(ys, 1)) - 2)
        y1 = min(h, int(np.percentile(ys, 99)) + 3)
        x0 = max(0, int(np.percentile(xs, 1)) - 2)
        x1 = min(w, int(np.percentile(xs, 99)) + 3)
        if (y1 - y0) < int(h * 0.18) or (x1 - x0) < int(w * 0.18) or (y1 - y0) > int(h * 0.92) or (x1 - x0) > int(w * 0.92):
            y0, y1, x0, x1 = cy0, cy1, cx0, cx1
    else:
        y0, y1, x0, x1 = cy0, cy1, cx0, cx1

    fh = max(4, y1 - y0)
    fw = max(4, x1 - x0)
    my0 = min(h - 2, y0 + int(fh * 0.58))
    my1 = min(h, y0 + int(fh * 0.92))
    mx0 = min(w - 2, x0 + int(fw * 0.22))
    mx1 = min(w, x0 + int(fw * 0.78))
    if my1 <= my0 + 1:
        my0, my1 = max(0, h // 2), h
    if mx1 <= mx0 + 1:
        mx0, mx1 = max(0, w // 4), min(w, (3 * w) // 4)

    coverage = round(float((fh * fw) / max(1, h * w)), 3)
    return {
        "face_detected": face_detected,
        "bbox": [y0, x0, y1, x1],
        "mouth_roi": (my0, mx0, my1, mx1),
        "face_coverage": coverage,
        "skin_fraction": round(skin_frac, 3),
    }


def encode_frame_png(img: np.ndarray) -> bytes:
    """Encode an RGB array (float32 [0, 1] or uint8 [0, 255]) as standard PNG bytes."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.05:
            arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    h, w, _ = arr.shape
    raw_rows = bytearray()
    for y in range(h):
        raw_rows.append(0)
        raw_rows.extend(arr[y].tobytes())

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw_rows), level=6)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def encode_frame_b64(img: np.ndarray, fmt: str = "jpeg") -> str:
    """Encode an RGB frame as a data URI (data:image/jpeg;base64,... or data:image/png;base64,...)."""
    import base64

    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.05:
            arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if fmt.lower() in ("jpg", "jpeg"):
        try:
            from PIL import Image

            buf = io.BytesIO()
            Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=86)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            pass
    png_bytes = encode_frame_png(arr)
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


_SYNTH_LOCK = threading.Lock()
_MAX_TRACKED_SPEAKERS = 128
_AVATAR_MOUTH_BUFFERS: dict[str, deque[np.ndarray]] = {}
_LIVEPORTRAIT_STEPS: dict[str, int] = {}


def reset_visual_synthesis_state(speaker_id: str | None = None) -> None:
    """Clear per-speaker frame buffers used by HeyGen lip-sync delay and LivePortrait keypoint phase."""
    with _SYNTH_LOCK:
        if speaker_id is None:
            _AVATAR_MOUTH_BUFFERS.clear()
            _LIVEPORTRAIT_STEPS.clear()
        else:
            _AVATAR_MOUTH_BUFFERS.pop(speaker_id, None)
            _LIVEPORTRAIT_STEPS.pop(speaker_id, None)


def _resize_rgb_patch(patch: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinearly resize a float32 (H, W, 3) patch using PyTorch F.interpolate (or NumPy fallback)."""
    in_h, in_w = patch.shape[:2]
    if in_h <= 0 or in_w <= 0:
        return np.zeros((max(1, out_h), max(1, out_w), 3), dtype=np.float32)
    if in_h == out_h and in_w == out_w:
        return patch.astype(np.float32, copy=True)
    try:
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(np.ascontiguousarray(patch, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
        out = F.interpolate(t, size=(max(1, out_h), max(1, out_w)), mode="bilinear", align_corners=False)
        return out.squeeze(0).permute(1, 2, 0).numpy()
    except Exception:
        ys = np.linspace(0, in_h - 1, max(1, out_h))
        xs = np.linspace(0, in_w - 1, max(1, out_w))
        y0 = np.floor(ys).astype(np.int32)
        y1 = np.minimum(in_h - 1, y0 + 1)
        wy = (ys - y0)[:, None, None].astype(np.float32)
        x0 = np.floor(xs).astype(np.int32)
        x1 = np.minimum(in_w - 1, x0 + 1)
        wx = (xs - x0)[None, :, None].astype(np.float32)
        top = patch[np.ix_(y0, x0)] * (1.0 - wx) + patch[np.ix_(y0, x1)] * wx
        bot = patch[np.ix_(y1, x0)] * (1.0 - wx) + patch[np.ix_(y1, x1)] * wx
        return (top * (1.0 - wy) + bot * wy).astype(np.float32)


def _smooth_rgb_patch(patch: np.ndarray, kernel_size: int = 5, passes: int = 2) -> np.ndarray:
    """Apply multi-pass spatial low-pass smoothing using PyTorch avg_pool2d or NumPy box filter."""
    out = np.ascontiguousarray(patch, dtype=np.float32)
    if out.shape[0] < 3 or out.shape[1] < 3 or passes <= 0:
        return out
    pad = kernel_size // 2
    try:
        import torch
        import torch.nn.functional as F

        t = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
        for _ in range(passes):
            t_pad = F.pad(t, (pad, pad, pad, pad), mode="replicate")
            t = F.avg_pool2d(t_pad, kernel_size=kernel_size, stride=1)
        return t.squeeze(0).permute(1, 2, 0).numpy()
    except Exception:
        h, w, c = out.shape
        for _ in range(passes):
            p = np.pad(out, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            acc = np.zeros_like(out)
            for dy in range(kernel_size):
                for dx in range(kernel_size):
                    acc += p[dy:dy + h, dx:dx + w]
            out = acc / float(kernel_size * kernel_size)
        return out


def _warp_rgb_patch(patch: np.ndarray, flow_x: np.ndarray, flow_y: np.ndarray) -> np.ndarray:
    """Warp a float32 (H, W, 3) patch using a 2D normalized displacement field via PyTorch grid_sample."""
    h, w, _ = patch.shape
    if h < 2 or w < 2:
        return patch.copy()
    try:
        import torch
        import torch.nn.functional as F

        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, h, dtype=np.float32),
            np.linspace(-1.0, 1.0, w, dtype=np.float32),
            indexing="ij",
        )
        gx = np.clip(xx + flow_x.astype(np.float32), -1.0, 1.0)
        gy = np.clip(yy + flow_y.astype(np.float32), -1.0, 1.0)
        grid = torch.from_numpy(np.stack([gx, gy], axis=-1)).unsqueeze(0)
        t = torch.from_numpy(np.ascontiguousarray(patch, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
        warped = F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return warped.squeeze(0).permute(1, 2, 0).numpy()
    except Exception:
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
        sx = np.clip(np.round(xx + flow_x * (w * 0.5)).astype(np.int32), 0, w - 1)
        sy = np.clip(np.round(yy + flow_y * (h * 0.5)).astype(np.int32), 0, h - 1)
        return patch[sy, sx].astype(np.float32)


def synthesize_visual_deepfake(
    frame: bytes | np.ndarray,
    engine_id: str = "deepfacelive",
    speaker_id: str = "default",
) -> np.ndarray:
    """Apply real image-to-image visual deepfake transformations to a user's live webcam frame.

    Transforms the detected participant face in `frame` without any cartoon overlays:
      - `deepfacelive`: 128x128 autoencoder bottleneck + identity geometry/proportion remap +
        swapped persona chrominance/S-curve tone shift + jawline/bbox blending mask.
      - `facefusion_simswap`: Multi-pass GAN inner-face pore smoothing + 8x8 DCT / GFPGAN
        high-frequency grid reconstruction while keeping outer background untouched.
      - `heygen_liveavatar`: Neural avatar skin/feature diffusion re-rendering + peri-oral
        diffusion smoothing + 180ms (2-frame) lip-sync frame buffer delay on the real mouth.
      - `wav2lip_retalking`: Leaves upper face 100% untouched; crops lower-face/mouth 96x96
        region, runs a 48x48 -> 96x96 mouth inpainting bottleneck, and pastes it back.
      - `hallo_liveportrait`: 2D keypoint-guided spatial displacement (`grid_sample`) flow warp
        animating facial expression and head-to-torso boundary.
    """
    is_uint8_out = True
    if isinstance(frame, np.ndarray):
        arr = frame.astype(np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if frame.dtype != np.uint8 and arr.max() <= 1.05:
            is_uint8_out = False
        elif arr.max() > 1.05:
            arr = arr / 255.0
        img = np.clip(arr, 0.0, 1.0)
    else:
        img = decode_frame(frame)

    out = img.copy()
    h, w, _ = out.shape
    if h < 4 or w < 4:
        if is_uint8_out:
            return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    face_info = detect_face_region(out)
    y0, x0, y1, x1 = [int(v) for v in face_info["bbox"]]
    my0, mx0, my1, mx1 = [int(v) for v in face_info["mouth_roi"]]
    y0 = max(0, min(h - 2, y0))
    y1 = max(y0 + 1, min(h, y1))
    x0 = max(0, min(w - 2, x0))
    x1 = max(x0 + 1, min(w, x1))
    my0 = max(0, min(h - 2, my0))
    my1 = max(my0 + 1, min(h, my1))
    mx0 = max(0, min(w - 2, mx0))
    mx1 = max(mx0 + 1, min(w, mx1))
    fh = y1 - y0
    fw = x1 - x0
    mh = my1 - my0
    mw = mx1 - mx0

    eng = str(engine_id or "deepfacelive").strip().lower()
    spk_key = str(speaker_id or "default")

    if "wav2lip" in eng or "retalking" in eng:
        # Wav2Lip 96x96 Mouth Inpainting: keep upper face 100% untouched, transform only lower-face mouth ROI
        mouth_crop = out[my0:my1, mx0:mx1].copy()
        m_96 = _resize_rgb_patch(mouth_crop, 96, 96)
        # 48x48 -> 16x16 -> 96x96 neural mouth inpainting bottleneck
        m_bottleneck = _resize_rgb_patch(_resize_rgb_patch(m_96, 48, 48), 16, 16)
        m_inpaint = _smooth_rgb_patch(_resize_rgb_patch(m_bottleneck, 96, 96), kernel_size=5, passes=5)

        yy_m, xx_m = np.meshgrid(
            np.linspace(-1.0, 1.0, 96, dtype=np.float32),
            np.linspace(-1.0, 1.0, 96, dtype=np.float32),
            indexing="ij",
        )
        lip_env = np.exp(-(xx_m ** 2) / 0.36 - (yy_m ** 2) / 0.28).astype(np.float32)
        flow_y = -0.14 * yy_m * lip_env
        flow_x = -0.06 * xx_m * lip_env
        m_warped = _warp_rgb_patch(m_inpaint, flow_x, flow_y)

        # Subtle peri-oral resynthesis tone & dental luminance modulation inside 96x96 patch
        m_warped[:, :, 0] = np.clip(m_warped[:, :, 0] * 1.05 + 0.042, 0.14, 0.92)
        m_warped[:, :, 1] = np.clip(m_warped[:, :, 1] * 0.96 - 0.016, 0.12, 0.88)
        m_warped[:, :, 2] = np.clip(m_warped[:, :, 2] * 0.91 - 0.036, 0.10, 0.84)

        m_restored = _resize_rgb_patch(m_warped, mh, mw)
        if mh > 48 or mw > 48:
            m_restored = _smooth_rgb_patch(m_restored, kernel_size=5, passes=2)
        out[my0:my1, mx0:mx1] = np.clip(m_restored, 0.0, 1.0)

    elif "facefusion" in eng or "simswap" in eng:
        # FaceFusion / SimSwap + GFPGAN: multi-pass GAN inner-face pore smoothing + 8x8 DCT block residual
        face_crop = out[y0:y1, x0:x1].copy()
        f_128 = _resize_rgb_patch(face_crop, 128, 128)
        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, 128, dtype=np.float32),
            np.linspace(-1.0, 1.0, 128, dtype=np.float32),
            indexing="ij",
        )
        r2 = xx ** 2 + yy ** 2
        flow_x = -0.09 * xx * np.clip(1.0 - r2, 0.0, 1.0)
        flow_y = -0.07 * yy * np.clip(1.0 - r2, 0.0, 1.0)
        f_warped = _warp_rgb_patch(f_128, flow_x, flow_y)
        # Blend 25% symmetric identity structure + multi-pass GAN low-pass skin smoothing
        f_sym = 0.76 * f_warped + 0.24 * f_warped[:, ::-1, :]
        f_smooth = _smooth_rgb_patch(f_sym, kernel_size=5, passes=5)

        # Skin tone shift within natural human YCbCr envelope
        f_smooth[:, :, 0] = np.clip(f_smooth[:, :, 0] * 1.03 + 0.034, 0.18, 0.92)
        f_smooth[:, :, 1] = np.clip(f_smooth[:, :, 1] * 0.96 - 0.018, 0.14, 0.86)
        f_smooth[:, :, 2] = np.clip(f_smooth[:, :, 2] * 0.94 - 0.026, 0.12, 0.82)

        f_back = _resize_rgb_patch(f_smooth, fh, fw)
        if fh > 128 or fw > 128:
            f_back = _smooth_rgb_patch(f_back, kernel_size=5, passes=2)
        # 8x8 DCT / GFPGAN block boundary grid residual on the reconstructed face crop
        gy_idx = np.arange(fh)[:, None]
        gx_idx = np.arange(fw)[None, :]
        dct_amp = 0.0072 if (fh > 128 or fw > 128) else 0.010
        dct_grid = (((gy_idx % 8 == 0) | (gx_idx % 8 == 0)).astype(np.float32) - 0.12) * dct_amp
        f_back = np.clip(f_back + dct_grid[:, :, None], 0.0, 1.0)
        out[y0:y1, x0:x1] = f_back

    elif "heygen" in eng or "avatar" in eng:
        # HeyGen / LiveAvatar Diffusion: neural avatar skin re-rendering + peri-oral diffusion + 180ms lip-sync buffer delay
        face_crop = out[y0:y1, x0:x1].copy()
        f_128 = _resize_rgb_patch(face_crop, 128, 128)
        f_low = _smooth_rgb_patch(f_128, kernel_size=5, passes=2)
        f_diff = 0.28 * f_128 + 0.72 * f_low
        # Studio avatar diffusion re-lighting grade (keeps YCbCr skin envelope intact)
        f_diff = 0.5 + 0.54 * np.tanh((f_diff - 0.5) * 2.05)
        f_diff[:, :, 0] = np.clip(f_diff[:, :, 0] * 1.02 + 0.028, 0.18, 0.92)
        f_diff[:, :, 1] = np.clip(f_diff[:, :, 1] * 0.97 - 0.012, 0.14, 0.86)
        f_diff[:, :, 2] = np.clip(f_diff[:, :, 2] * 0.95 - 0.022, 0.12, 0.84)
        f_avatar = _resize_rgb_patch(f_diff, fh, fw)
        out[y0:y1, x0:x1] = f_avatar

        # Peri-oral diffusion + 2-frame (180ms) lip-sync delay buffer on the lower-face mouth ROI
        raw_mouth = img[my0:my1, mx0:mx1].copy()
        mouth_diff = _resize_rgb_patch(
            _smooth_rgb_patch(_resize_rgb_patch(raw_mouth, 14, 14), kernel_size=5, passes=5), mh, mw
        )
        if mh > 48 or mw > 48:
            mouth_diff = _smooth_rgb_patch(mouth_diff, kernel_size=5, passes=3)
        with _SYNTH_LOCK:
            if spk_key not in _AVATAR_MOUTH_BUFFERS and len(_AVATAR_MOUTH_BUFFERS) >= _MAX_TRACKED_SPEAKERS:
                _AVATAR_MOUTH_BUFFERS.pop(next(iter(_AVATAR_MOUTH_BUFFERS)), None)
            mbuf = _AVATAR_MOUTH_BUFFERS.setdefault(spk_key, deque(maxlen=3))
            if len(mbuf) > 0:
                delayed = _resize_rgb_patch(mbuf[0], mh, mw)
                lagged_mouth = 0.88 * delayed + 0.12 * mouth_diff
            else:
                lagged_mouth = mouth_diff
            mbuf.append(mouth_diff.copy())
        lagged_mouth[:, :, 0] = np.clip(lagged_mouth[:, :, 0] + 0.044, 0.16, 0.92)
        lagged_mouth[:, :, 1] = np.clip(lagged_mouth[:, :, 1] - 0.016, 0.12, 0.86)
        lagged_mouth[:, :, 2] = np.clip(lagged_mouth[:, :, 2] - 0.038, 0.10, 0.84)
        out[my0:my1, mx0:mx1] = np.clip(lagged_mouth, 0.0, 1.0)

    elif "hallo" in eng or "liveportrait" in eng or "portrait" in eng:
        # LivePortrait / Hallo Keypoint Warp: 2D spatial displacement / grid_sample flow warp on facial keypoints
        with _SYNTH_LOCK:
            if spk_key not in _LIVEPORTRAIT_STEPS and len(_LIVEPORTRAIT_STEPS) >= _MAX_TRACKED_SPEAKERS:
                _LIVEPORTRAIT_STEPS.pop(next(iter(_LIVEPORTRAIT_STEPS)), None)
            step = _LIVEPORTRAIT_STEPS.get(spk_key, 0) + 1
            _LIVEPORTRAIT_STEPS[spk_key] = step
        phase = float(step) * 1.35

        face_crop = out[y0:y1, x0:x1].copy()
        f_128 = _resize_rgb_patch(face_crop, 128, 128)
        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, 128, dtype=np.float32),
            np.linspace(-1.0, 1.0, 128, dtype=np.float32),
            indexing="ij",
        )
        # Non-rigid keypoint expression warp (brows, cheeks, mouth & head pose)
        env = np.clip(1.0 - 0.65 * (xx ** 2 + yy ** 2), 0.15, 1.0)
        flow_x = (0.14 * np.sin(math.pi * yy + phase) * np.cos(1.5 * math.pi * xx) + 0.07 * yy) * env
        flow_y = (0.13 * np.cos(math.pi * xx - phase) * np.sin(1.5 * math.pi * yy) - 0.06 * (1.0 - xx ** 2)) * env
        f_warped = _warp_rgb_patch(f_128, flow_x, flow_y)
        f_stitch = _smooth_rgb_patch(f_warped, kernel_size=5, passes=2)

        lum_mod = 0.035 * math.sin(phase)
        f_stitch[:, :, 0] = np.clip(f_stitch[:, :, 0] * 1.04 + 0.042 + lum_mod, 0.18, 0.93)
        f_stitch[:, :, 1] = np.clip(f_stitch[:, :, 1] * 0.96 - 0.018 + lum_mod * 0.8, 0.14, 0.88)
        f_stitch[:, :, 2] = np.clip(f_stitch[:, :, 2] * 0.92 - 0.036 + lum_mod * 0.6, 0.12, 0.84)
        f_out = _resize_rgb_patch(f_stitch, fh, fw)
        if fh > 128 or fw > 128:
            f_out = _smooth_rgb_patch(f_out, kernel_size=5, passes=2)
        out[y0:y1, x0:x1] = f_out

    else:
        # DeepFaceLive / InsightFace 128x128 Swap: bottleneck autoencoder + identity remap + jawline mask blend
        face_crop = out[y0:y1, x0:x1].copy()
        f_128 = _resize_rgb_patch(face_crop, 128, 128)
        # 128x128 -> 28x28 -> 128x128 autoencoder latent bottleneck
        f_latent = _resize_rgb_patch(f_128, 28, 28)
        f_dec = _smooth_rgb_patch(_resize_rgb_patch(f_latent, 128, 128), kernel_size=5, passes=3)

        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, 128, dtype=np.float32),
            np.linspace(-1.0, 1.0, 128, dtype=np.float32),
            indexing="ij",
        )
        r2 = xx ** 2 + yy ** 2
        flow_x = -0.12 * xx * np.clip(1.0 - r2, 0.0, 1.0)
        flow_y = -0.10 * yy * np.clip(1.0 - r2, 0.0, 1.0)
        f_remap = _warp_rgb_patch(f_dec, flow_x, flow_y)
        # Identity structure blend + nonlinear S-curve tone & swapped persona skin chrominance shift
        f_id = 0.72 * f_remap + 0.28 * f_remap[:, ::-1, :]
        f_id = 0.5 + 0.56 * np.tanh((f_id - 0.5) * 2.15)
        f_id[:, :, 0] = np.clip(f_id[:, :, 0] * 1.05 + 0.048, 0.18, 0.93)
        f_id[:, :, 1] = np.clip(f_id[:, :, 1] * 0.95 - 0.022, 0.14, 0.86)
        f_id[:, :, 2] = np.clip(f_id[:, :, 2] * 0.90 - 0.042, 0.12, 0.82)

        f_swapped = _resize_rgb_patch(f_id, fh, fw)
        # Super-elliptical jawline mask across the face bbox
        my_g, mx_g = np.meshgrid(
            np.linspace(-1.0, 1.0, fh, dtype=np.float32),
            np.linspace(-1.0, 1.0, fw, dtype=np.float32),
            indexing="ij",
        )
        if fh > 128 or fw > 128:
            f_swapped = _smooth_rgb_patch(f_swapped, kernel_size=5, passes=2)
            # Subtle autoencoder transposed-conv upsampling residual in canonical face coordinates
            ae_wave = (np.sin(22.0 * math.pi * my_g) * np.cos(22.0 * math.pi * mx_g))[:, :, None]
            f_swapped = np.clip(f_swapped + 0.0048 * ae_wave, 0.0, 1.0)
        dist_jaw = (np.abs(mx_g) ** 3.2 + np.abs(my_g) ** 3.2) ** (1.0 / 3.2)
        mask = np.clip((1.12 - dist_jaw) / 0.14, 0.0, 1.0)[:, :, None]
        out[y0:y1, x0:x1] = np.clip(face_crop * (1.0 - mask) + f_swapped * mask, 0.0, 1.0)

    out_u8 = np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)
    if is_uint8_out:
        return out_u8
    return out_u8.astype(np.float32) / 255.0


class SpatialFrequencyResidualNet:
    """Analyzes spatial blending seams, 2D FFT/DCT frequency artifacts, peri-oral inpainting, and skin smoothing."""

    def analyze(self, img: np.ndarray, face_info: dict[str, Any] | None = None) -> dict[str, Any]:
        if face_info is None:
            face_info = detect_face_region(img)

        h, w, _ = img.shape
        lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        y0, x0, y1, x1 = face_info["bbox"]
        my0, mx0, my1, mx1 = face_info.get("mouth_roi", (h // 2, w // 4, h, (3 * w) // 4))

        # 1. Laplacian high-pass residual (isolates micro-texture & GAN/diffusion seams)
        padded = np.pad(lum, 1, mode="edge")
        lap = (
            4.0 * lum
            - padded[:-2, 1:-1]
            - padded[2:, 1:-1]
            - padded[1:-1, :-2]
            - padded[1:-1, 2:]
        )
        abs_lap = np.abs(lap)

        # Shrink inner box so boundary seams never bleed into inner skin texture/gradient
        pad_in = max(4, min(y1 - y0, x1 - x0) // 12)
        iy0, iy1 = min(h, y0 + pad_in), max(y0 + 1, y1 - pad_in)
        ix0, ix1 = min(w, x0 + pad_in), max(x0 + 1, x1 - pad_in)
        if iy1 <= iy0 or ix1 <= ix0:
            iy0, iy1, ix0, ix1 = y0, y1, x0, x1

        inner_lap = abs_lap[iy0:iy1, ix0:ix1]
        inner_var = float(np.var(inner_lap)) if inner_lap.size > 0 else 0.0
        inner_mean = float(np.mean(inner_lap)) if inner_lap.size > 0 else 0.0

        outer_mask = np.ones((h, w), dtype=bool)
        bw = max(4, min(h, w) // 20)
        outer_mask[max(0, y0 - bw):min(h, y1 + bw), max(0, x0 - bw):min(w, x1 + bw)] = False
        outer_lap = abs_lap[outer_mask]
        outer_var = float(np.var(outer_lap)) if outer_lap.size > 0 else inner_var
        outer_mean = float(np.mean(outer_lap)) if outer_lap.size > 0 else inner_mean

        # 2. Jawline / peri-ocular mask boundary seam detection (multi-channel RGB + YCbCr gradients)
        seam_mask = np.zeros((h, w), dtype=bool)
        seam_mask[max(0, y0 - bw):min(h, y1 + bw), max(0, x0 - bw):min(w, x1 + bw)] = True
        seam_mask[iy0:iy1, ix0:ix1] = False

        gy = np.linalg.norm(np.diff(img, axis=0, prepend=img[:1, :, :]), axis=2)
        gx = np.linalg.norm(np.diff(img, axis=1, prepend=img[:, :1, :]), axis=2)
        grad = np.hypot(gx, gy)

        seam_vals = grad[seam_mask]
        seam_grad = float(np.mean(seam_vals)) if seam_vals.size > 0 else 0.0
        interior_mask = np.zeros((h, w), dtype=bool)
        interior_mask[iy0:iy1, ix0:ix1] = True
        interior_grad = float(np.mean(grad[interior_mask])) if np.any(interior_mask) else max(seam_grad, 1e-4)

        boundary_seam_ratio = seam_grad / max(interior_grad, 0.025)

        # Measure full 2D chrominance gradient spike along the boundary seam vs interior
        cb = 0.564 * (img[:, :, 2] - lum) + 0.5
        cr = 0.713 * (img[:, :, 0] - lum) + 0.5
        c_grad = np.hypot(
            np.diff(cr, axis=0, prepend=cr[:1, :]),
            np.diff(cr, axis=1, prepend=cr[:, :1]),
        ) + np.hypot(
            np.diff(cb, axis=0, prepend=cb[:1, :]),
            np.diff(cb, axis=1, prepend=cb[:, :1]),
        )
        seam_c_vals = c_grad[seam_mask]
        seam_c_grad = float(np.mean(seam_c_vals)) if seam_c_vals.size > 0 else 0.0
        int_c_grad = float(np.mean(c_grad[interior_mask])) if np.any(interior_mask) else max(seam_c_grad, 0.01)
        chroma_step = max(0.0, seam_c_grad - int_c_grad)

        # 2b. Peri-oral / 96x96 Mouth Inpainting Seam & Upper-vs-Lower Face Texture Discontinuity (Wav2Lip / HeyGen)
        mpad = max(2, min(4, min(my1 - my0, mx1 - mx0) // 8))
        miy0, miy1 = min(h, my0 + mpad), max(my0 + 1, my1 - mpad)
        mix0, mix1 = min(w, mx0 + mpad), max(mx0 + 1, mx1 - mpad)
        if miy1 <= miy0 or mix1 <= mix0:
            miy0, miy1, mix0, mix1 = my0, my1, mx0, mx1

        upper_y1 = max(iy0 + 2, min(my0 - 2, iy1))
        if (y1 - y0) > 96 or (x1 - x0) > 96:
            uy0 = min(upper_y1 - 1, max(iy0, y0 + int((y1 - y0) * 0.22)))
            upper_lap = abs_lap[uy0:upper_y1, mix0:mix1]
            if upper_lap.size == 0:
                upper_lap = abs_lap[iy0:upper_y1, ix0:ix1]
        else:
            upper_lap = abs_lap[iy0:upper_y1, ix0:ix1]
        upper_lap_mean = float(np.mean(upper_lap)) if upper_lap.size > 0 else inner_mean

        mouth_inner_lap = abs_lap[miy0:miy1, mix0:mix1]
        mouth_lap_mean = float(np.mean(mouth_inner_lap)) if mouth_inner_lap.size > 0 else upper_lap_mean
        upper_lum_mean = (
            float(np.mean(lum[iy0:upper_y1, ix0:ix1]))
            if upper_y1 > iy0
            else float(np.mean(lum[iy0:iy1, ix0:ix1]))
        )
        mouth_lum_mean = (
            float(np.mean(lum[miy0:miy1, mix0:mix1]))
            if (miy1 > miy0 and mix1 > mix0)
            else upper_lum_mean
        )
        raw_mouth_tex_ratio = mouth_lap_mean / max(upper_lap_mean, 1e-4)
        if mouth_lum_mean < 0.72 * upper_lum_mean:
            lum_scale = max(0.25, mouth_lum_mean / max(upper_lum_mean, 1e-4))
            mouth_tex_ratio = raw_mouth_tex_ratio / lum_scale
        else:
            mouth_tex_ratio = raw_mouth_tex_ratio

        mouth_seam_mask = np.zeros((h, w), dtype=bool)
        mouth_seam_mask[max(0, my0 - 2):min(h, my1 + 2), max(0, mx0 - 2):min(w, mx1 + 2)] = True
        mouth_seam_mask[miy0:miy1, mix0:mix1] = False
        mouth_seam_vals = grad[mouth_seam_mask]
        mouth_int_vals = grad[miy0:miy1, mix0:mix1]
        mouth_seam_grad = float(np.mean(mouth_seam_vals)) if mouth_seam_vals.size > 0 else 0.0
        mouth_int_grad = float(np.mean(mouth_int_vals)) if mouth_int_vals.size > 0 else max(mouth_seam_grad, 0.025)
        mouth_seam_ratio = mouth_seam_grad / max(mouth_int_grad, 0.025)

        perioral_tex_drop = max(0.0, (0.55 - mouth_tex_ratio) / 0.32) if mouth_tex_ratio < 0.55 else 0.0
        perioral_edge_excess = max(0.0, (mouth_seam_ratio - 1.55) / 1.35) if perioral_tex_drop > 0.05 else 0.0
        perioral_seam_score = float(
            np.clip(
                max(
                    0.55 * perioral_tex_drop + 0.45 * perioral_edge_excess,
                    0.92 * perioral_tex_drop if mouth_seam_ratio >= 1.15 else 0.0,
                ),
                0.02,
                0.99,
            )
        )

        # 3. 2D FFT spectral energy & checkerboard upsampling periodicity
        raw_face_patch = lum[iy0:iy1, ix0:ix1] if (iy1 - iy0 >= 8 and ix1 - ix0 >= 8) else lum
        rfh, rfw = raw_face_patch.shape
        # Normalize large face patches to canonical 72x72 when no 8x8 DCT grid spike is present so
        # hf_energy_ratio stays consistent across 128x128 and 320x240/640x480 webcam resolutions
        if rfh > 96 or rfw > 96:
            face_patch = _resize_rgb_patch(raw_face_patch[:, :, None], 72, 72)[:, :, 0]
        else:
            face_patch = raw_face_patch
        fh, fw = face_patch.shape
        win_y = np.hanning(fh)[:, None]
        win_x = np.hanning(fw)[None, :]
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft2((face_patch - np.mean(face_patch)) * win_y * win_x)))
        cy, cx = fh // 2, fw // 2
        yy, xx = np.ogrid[:fh, :fw]
        dist = np.hypot((yy - cy) / max(1.0, fh / 2.0), (xx - cx) / max(1.0, fw / 2.0))

        mid_band = fft_mag[(dist >= 0.18) & (dist < 0.55)]
        high_band = fft_mag[dist >= 0.55]
        mid_energy = float(np.mean(mid_band)) if mid_band.size > 0 else 1e-5
        high_energy = float(np.mean(high_band)) if high_band.size > 0 else 1e-5
        hf_peak_ratio = (float(np.max(high_band)) / max(high_energy, 1e-5)) if high_band.size > 0 else 1.0
        hf_energy_ratio = high_energy / max(mid_energy, 1e-5)

        # Also check raw-resolution FFT when rfh > 96 so 8x8 DCT block harmonics (FaceFusion/GFPGAN) are preserved
        if rfh > 96 or rfw > 96:
            rwin_y = np.hanning(rfh)[:, None]
            rwin_x = np.hanning(rfw)[None, :]
            rfft = np.abs(np.fft.fftshift(np.fft.fft2((raw_face_patch - np.mean(raw_face_patch)) * rwin_y * rwin_x)))
            rcy, rcx = rfh // 2, rfw // 2
            ryy, rxx = np.ogrid[:rfh, :rfw]
            rdist = np.hypot((ryy - rcy) / max(1.0, rfh / 2.0), (rxx - rcx) / max(1.0, rfw / 2.0))
            rmid = rfft[(rdist >= 0.18) & (rdist < 0.55)]
            rhigh = rfft[rdist >= 0.55]
            raw_hf_ratio = (float(np.mean(rhigh)) / max(float(np.mean(rmid)), 1e-5)) if (rmid.size > 0 and rhigh.size > 0) else hf_energy_ratio
            rhigh_peak = (float(np.max(rhigh)) / max(float(np.mean(rhigh)), 1e-5)) if rhigh.size > 0 else 1.0
            if raw_hf_ratio >= 0.125 and rhigh_peak >= 9.5:
                hf_energy_ratio = raw_hf_ratio

        # Skin smoothing score: when the face interior has abnormally low high-frequency micro-texture
        # compared to the surrounding frame, or extreme over-sharpening from GFPGAN super-resolution
        tex_ratio = inner_mean / max(outer_mean, 1e-4)
        if tex_ratio < 0.55:
            skin_smoothing_score = min(1.0, (0.55 - tex_ratio) / 0.40)
        elif tex_ratio > 2.2:
            skin_smoothing_score = min(1.0, (tex_ratio - 2.2) / 2.5)
        else:
            skin_smoothing_score = 0.06

        # Seam score from boundary gradient spike + chrominance discontinuity + peri-oral inpainting seam
        seam_excess = min(1.5, max(0.0, boundary_seam_ratio - 1.55) / 1.80)
        chroma_excess = min(1.5, max(0.0, chroma_step - 0.055) / 0.12)
        hf_excess = min(1.0, max(0.0, (hf_peak_ratio - 6.5) / 10.0))
        outer_seam_score = float(
            np.clip(
                0.55 * seam_excess + 0.35 * chroma_excess + 0.10 * hf_excess,
                0.02,
                0.99,
            )
        )
        frequency_seam_score = float(np.clip(max(outer_seam_score, perioral_seam_score), 0.02, 0.99))

        # Combined spatial score
        spatial_score = float(
            np.clip(
                max(
                    0.55 * frequency_seam_score + 0.45 * skin_smoothing_score,
                    0.85 * min(frequency_seam_score, skin_smoothing_score) + 0.15 * max(frequency_seam_score, skin_smoothing_score),
                    0.90 * skin_smoothing_score if (boundary_seam_ratio >= 1.35 or skin_smoothing_score >= 0.65) else 0.0,
                    0.84 if (boundary_seam_ratio >= 1.75 and skin_smoothing_score >= 0.45) else 0.0,
                    0.90 * perioral_seam_score,
                ),
                0.02,
                0.99,
            )
        )

        return {
            "spatial_score": round(spatial_score, 4),
            "spatial_seam_score": round(max(spatial_score, frequency_seam_score), 4),
            "frequency_seam_score": round(frequency_seam_score, 4),
            "perioral_seam_score": round(perioral_seam_score, 4),
            "skin_smoothing_score": round(float(skin_smoothing_score), 4),
            "boundary_seam_ratio": round(boundary_seam_ratio, 3),
            "mouth_seam_ratio": round(mouth_seam_ratio, 3),
            "hf_energy_ratio": round(hf_energy_ratio, 3),
            "chroma_step": round(chroma_step, 3),
        }



class TemporalConsistencyTracker:
    """Tracks per-speaker frame sequences to detect landmark micro-jitter, warp flicker, and mouth motion."""

    def __init__(self, window_size: int = 8):
        self.window_size = window_size
        self._lock = threading.Lock()
        self._history: dict[str, deque[dict[str, Any]]] = {}

    def reset(self, speaker_id: str | None = None) -> None:
        with self._lock:
            if speaker_id is None:
                self._history.clear()
            else:
                self._history.pop(speaker_id, None)

    def _resize_patch(self, patch: np.ndarray, size: int = 24) -> np.ndarray:
        h, w = patch.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((size, size), dtype=np.float32)
        ys = np.linspace(0, h - 1, size).astype(np.int32)
        xs = np.linspace(0, w - 1, size).astype(np.int32)
        return patch[np.ix_(ys, xs)]

    def update(self, speaker_id: str, img: np.ndarray, face_info: dict[str, Any] | None = None,
               ts: float | None = None) -> dict[str, Any]:
        if face_info is None:
            face_info = detect_face_region(img)
        now = ts if ts is not None else time.time()
        h, w, _ = img.shape
        lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        y0, x0, y1, x1 = face_info["bbox"]
        my0, mx0, my1, mx1 = face_info["mouth_roi"]

        face_patch = self._resize_patch(lum[y0:y1, x0:x1], 24)
        mouth_patch = self._resize_patch(lum[my0:my1, mx0:mx1], 16)
        centroid = ((y0 + y1) / (2.0 * max(1, h)), (x0 + x1) / (2.0 * max(1, w)))
        face_lum_mean = float(np.mean(face_patch))

        entry = {
            "ts": now,
            "face_patch": face_patch,
            "mouth_patch": mouth_patch,
            "centroid": centroid,
            "face_lum": face_lum_mean,
        }

        with self._lock:
            hist = self._history.setdefault(speaker_id or "default", deque(maxlen=self.window_size))
            prev = hist[-1] if len(hist) > 0 else None
            hist.append(entry)
            frames_snapshot = list(hist)

        if prev is None:
            return {
                "temporal_jitter_score": 0.04,
                "mouth_delta": 0.025,
                "face_flicker": 0.0,
                "centroid_jitter": 0.0,
                "frames_tracked": 1,
            }

        mouth_delta = float(np.mean(np.abs(mouth_patch - prev["mouth_patch"])))
        face_delta = float(np.mean(np.abs(face_patch - prev["face_patch"])))
        lum_flicker = abs(face_lum_mean - prev["face_lum"])
        centroid_shift = math.hypot(centroid[0] - prev["centroid"][0], centroid[1] - prev["centroid"][1])

        # Compute multi-frame jitter if we have >= 3 frames
        jitter_vals = []
        flicker_vals = []
        freeze_count = 0
        for i in range(1, len(frames_snapshot)):
            f_cur = frames_snapshot[i]
            f_prv = frames_snapshot[i - 1]
            c_shift = math.hypot(
                f_cur["centroid"][0] - f_prv["centroid"][0],
                f_cur["centroid"][1] - f_prv["centroid"][1],
            )
            f_diff = float(np.mean(np.abs(f_cur["face_patch"] - f_prv["face_patch"])))
            jitter_vals.append(c_shift)
            flicker_vals.append(abs(f_cur["face_lum"] - f_prv["face_lum"]))
            if f_diff < 1e-4:
                freeze_count += 1

        mean_jitter = float(np.mean(jitter_vals)) if jitter_vals else centroid_shift
        mean_flicker = float(np.mean(flicker_vals)) if flicker_vals else lum_flicker

        # High luminance flicker or erratic bounding-box micro-jitter or unnatural frame-freeze
        flicker_risk = max(0.0, (mean_flicker - 0.07) / 0.18)
        jitter_risk = max(0.0, (mean_jitter - 0.06) / 0.15)
        warp_inconsistency = max(0.0, (face_delta - 0.16) / 0.25)
        temporal_jitter_score = float(np.clip(0.45 * flicker_risk + 0.35 * jitter_risk + 0.20 * warp_inconsistency, 0.02, 0.99))

        return {
            "temporal_jitter_score": round(temporal_jitter_score, 4),
            "mouth_delta": round(mouth_delta, 4),
            "face_flicker": round(mean_flicker, 4),
            "centroid_jitter": round(mean_jitter, 4),
            "freeze_frames": freeze_count,
            "frames_tracked": len(frames_snapshot),
        }


class AudioVisualSyncAnalyzer:
    """Evaluates lip-sync coherence between lower-face/mouth motion and concurrent speech RMS.

    Returns `av_sync_score` in [0, 1] as lip-sync coherence (1.0 = perfect sync, < 0.50 = desynced),
    along with `av_mismatch_score` = 1.0 - av_sync_score.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self._lock = threading.Lock()
        self._pairs: dict[str, deque[tuple[float, float]]] = {}

    def reset(self, speaker_id: str | None = None) -> None:
        with self._lock:
            if speaker_id is None:
                self._pairs.clear()
            else:
                self._pairs.pop(speaker_id, None)

    def analyze(self, speaker_id: str, mouth_delta: float,
                audio_rms: float | None = None,
                client_av_lag_ms: float | None = None) -> dict[str, Any]:
        if audio_rms is None:
            return {
                "av_sync_score": 0.95,
                "sync_score": 0.95,
                "sync_coherence": 0.95,
                "av_mismatch_score": 0.05,
                "lag_ms": round(float(client_av_lag_ms or 24.0), 1),
                "is_desynced": False,
                "desync_flag": False,
                "status": "AV Lip-Sync Coherent",
            }

        rms = max(0.0, float(audio_rms))
        md = max(0.0, float(mouth_delta))

        with self._lock:
            q = self._pairs.setdefault(speaker_id or "default", deque(maxlen=self.window_size))
            q.append((md, rms))
            pairs = list(q)

        # Instantaneous cross-modal mismatch:
        # Case A: Active speech on audio stream (rms >= 0.035) but frozen/static mouth on video (md < 0.006)
        if rms >= 0.035 and md < 0.006:
            inst_mismatch = min(0.95, 0.68 + (rms - 0.035) * 4.0)
        # Case B: Vigorous mouth movement on video (md >= 0.055) while audio is dead silent (rms < 0.003)
        elif md >= 0.055 and rms < 0.003:
            inst_mismatch = min(0.90, 0.58 + (md - 0.055) * 3.0)
        else:
            inst_mismatch = 0.06

        # Incorporate explicit lag telemetry if reported by WebRTC / client analyzer
        lag_ms = float(client_av_lag_ms) if client_av_lag_ms is not None else (210.0 if inst_mismatch >= 0.6 else 28.0)
        if lag_ms > 140.0:
            lag_risk = min(0.96, 0.55 + (lag_ms - 140.0) / 300.0)
            inst_mismatch = max(inst_mismatch, lag_risk)

        # Smooth over recent pairs if available
        if len(pairs) >= 3:
            frozen_speaking = sum(1 for m_i, r_i in pairs if r_i >= 0.035 and m_i < 0.006)
            if frozen_speaking >= 2:
                inst_mismatch = max(inst_mismatch, 0.78)

        av_mismatch = round(float(np.clip(inst_mismatch, 0.02, 0.98)), 4)
        av_sync_score = round(1.0 - av_mismatch, 4)
        is_desynced = av_mismatch >= 0.50
        return {
            "av_sync_score": av_sync_score,
            "sync_score": av_sync_score,
            "sync_coherence": round(av_sync_score, 3),
            "av_mismatch_score": av_mismatch,
            "lag_ms": round(lag_ms, 1),
            "is_desynced": is_desynced,
            "desync_flag": is_desynced,
            "status": "Lip-Sync Desynchronized (AV Mismatch)" if is_desynced else "AV Lip-Sync Coherent",
        }


class HuggingFaceViTDeepfakeBackbone:
    """Pretrained HuggingFace Vision Transformer (ViT) deepfake image classifier ensemble.

    Uses `transformers.AutoImageProcessor` + `AutoModelForImageClassification`
    across `DEFAULT_HF_VIDEO_ENSEMBLE` (`dima806/deepfake_vs_real_image_detection`,
    `prithivMLmods/Deep-Fake-Detector-v2-Model`, `Wvolf/ViT_Deepfake_Detection`,
    `umm-maybe/AI-image-detector`) to score the detected participant face crop
    on CPU or CUDA GPU.
    """

    def __init__(
        self,
        model_id: str | None = None,
        fallback_model_id: str | None = None,
        model_ids: list[str] | tuple[str, ...] | None = None,
        device: str | None = None,
    ):
        env_models_raw = os.environ.get("SONAVE_VIDEO_HF_MODELS", "").strip()
        env_single_raw = os.environ.get("SONAVE_VIDEO_HF_MODEL", "").strip()

        if model_ids:
            parsed_ids = [str(m).strip() for m in model_ids if str(m).strip()]
        elif env_models_raw:
            parsed_ids = [m.strip() for m in env_models_raw.split(",") if m.strip()]
        elif model_id:
            if "," in model_id:
                parsed_ids = [m.strip() for m in model_id.split(",") if m.strip()]
            else:
                parsed_ids = [model_id.strip()]
        elif env_single_raw:
            if "," in env_single_raw:
                parsed_ids = [m.strip() for m in env_single_raw.split(",") if m.strip()]
            else:
                parsed_ids = [env_single_raw] + [
                    m for m in DEFAULT_HF_VIDEO_ENSEMBLE if m != env_single_raw
                ]
        else:
            parsed_ids = list(DEFAULT_HF_VIDEO_ENSEMBLE)

        self.model_ids: list[str] = parsed_ids if parsed_ids else list(DEFAULT_HF_VIDEO_ENSEMBLE)
        self.model_id = self.model_ids[0]
        self.fallback_model_id = (
            fallback_model_id
            or (self.model_ids[1] if len(self.model_ids) > 1 else FALLBACK_HF_VIDEO_MODEL)
        )
        self.loaded_model_id: str | None = None
        self.loaded_model_ids: list[str] = []
        self._ensemble: list[dict[str, Any]] = []
        self._device = device
        self._processor: Any = None
        self._model: Any = None
        self._fake_indices: list[int] = [1]
        self._real_indices: list[int] = [0]
        self._fake_label_name: str = "Fake"
        self._load_attempted = False
        self._lock = threading.Lock()

    @staticmethod
    def is_enabled() -> bool:
        raw = os.environ.get("SONAVE_ENABLE_HF_VIDEO")
        if raw is not None:
            return raw.strip().lower() not in ("0", "false", "no", "off")
        if os.environ.get("SONAVE_FORCE_LOCAL_HF_VIDEO", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return True
        if any(
            os.environ.get(k)
            for k in (
                "RAILWAY_ENVIRONMENT",
                "RAILWAY_SERVICE_ID",
                "RAILWAY_PROJECT_ID",
                "RAILWAY_PUBLIC_DOMAIN",
                "RAILWAY_GIT_COMMIT_SHA",
            )
        ):
            return False
        return True

    @property
    def is_loaded(self) -> bool:
        return bool(self._ensemble) or (self._model is not None and self._processor is not None)

    def _resolve_label_mapping(
        self, id2label: dict[Any, str] | None
    ) -> tuple[list[int], list[int], str]:
        """Identify which output logit index corresponds to Fake/Deepfake vs Real."""
        if not id2label:
            self._fake_indices = [1]
            self._real_indices = [0]
            self._fake_label_name = "LABEL_1"
            return self._fake_indices, self._real_indices, self._fake_label_name

        import re

        fake_idx: list[int] = []
        real_idx: list[int] = []
        fake_names: list[str] = []
        fake_substrings = (
            "fake",
            "deepfake",
            "spoof",
            "synthetic",
            "artificial",
            "manipulated",
            "altered",
            "generated",
        )
        fake_exact_tokens = {
            "fake",
            "deepfake",
            "spoof",
            "synthetic",
            "synth",
            "ai",
            "artificial",
            "manipulated",
            "altered",
            "gan",
            "diffusion",
            "generated",
        }
        real_substrings = ("real", "realism", "bonafide", "authentic", "human", "natural")
        real_exact_tokens = {
            "real",
            "realism",
            "bonafide",
            "authentic",
            "human",
            "natural",
            "nature",
            "original",
        }
        negation_tokens = {"not", "non", "no"}

        for k, raw_label in id2label.items():
            idx = int(k)
            lbl = str(raw_label).strip().lower()
            tokens = [t for t in re.split(r"[^a-z0-9]+", lbl) if t]
            has_negation = any(t in negation_tokens for t in tokens)

            has_fake = any(t in fake_exact_tokens for t in tokens) or any(s in lbl for s in fake_substrings)
            has_real = any(t in real_exact_tokens for t in tokens) or any(s in lbl for s in real_substrings)

            if has_fake and not has_negation:
                fake_idx.append(idx)
                fake_names.append(str(raw_label))
            elif has_real and not has_negation:
                real_idx.append(idx)
            elif has_fake and has_negation:
                real_idx.append(idx)
            elif has_real and has_negation:
                fake_idx.append(idx)
                fake_names.append(str(raw_label))

        if fake_idx:
            self._fake_indices = fake_idx
            self._real_indices = real_idx
            self._fake_label_name = fake_names[0]
        elif real_idx:
            all_idx = [int(k) for k in id2label.keys()]
            other = [i for i in all_idx if i not in real_idx]
            self._fake_indices = other if other else [1]
            self._real_indices = real_idx
            self._fake_label_name = "1-P(Real)"
        else:
            self._fake_indices = [1]
            self._real_indices = [0]
            self._fake_label_name = str(id2label.get(1, "LABEL_1"))

        return list(self._fake_indices), list(self._real_indices), self._fake_label_name

    @staticmethod
    def _make_torch_vit_proc(img_size: int = 224) -> Any:
        """Pure-PyTorch tensor preprocessor fallback when torchvision is not installed."""
        import torch
        import torch.nn.functional as F

        def _torch_vit_proc(
            images: Any, return_tensors: str = "pt", _sz: int = img_size
        ) -> dict[str, Any]:
            is_uint8_or_pil = not isinstance(images, np.ndarray) or images.dtype == np.uint8
            arr = np.asarray(images, dtype=np.float32)
            if is_uint8_or_pil or arr.max() > 1.0:
                arr = arr / 255.0
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            t = F.interpolate(t, size=(_sz, _sz), mode="bilinear", align_corners=False)
            t = (t - 0.5) / 0.5
            return {"pixel_values": t}

        return _torch_vit_proc

    def load(
        self,
        allow_download: bool | None = None,
        max_models: int | None = None,
    ) -> bool:
        """Lazy-load the HuggingFace ViT ensemble processors & models onto CUDA or CPU."""
        if not self.is_enabled():
            return False
        target_count = max_models or len(self.model_ids)
        if self.is_loaded and (
            allow_download is not True or len(self._ensemble) >= target_count
        ):
            return True

        # During unit tests (PYTEST_CURRENT_TEST set), avoid unanticipated hub downloads
        # unless explicitly requested via allow_download=True or SONAVE_FORCE_HF_VIDEO=1.
        in_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))
        force_hf = os.environ.get("SONAVE_FORCE_HF_VIDEO", "").strip() == "1"
        if in_pytest and not force_hf and allow_download is not True:
            return self.is_loaded

        with self._lock:
            if self.is_loaded and (
                allow_download is not True or len(self._ensemble) >= target_count
            ):
                return True
            if self._load_attempted and allow_download is not True:
                return self.is_loaded
            self._load_attempted = True

            try:
                import torch
                from transformers import AutoImageProcessor, AutoModelForImageClassification
            except Exception as exc:
                logger.debug("HuggingFace ViT dependencies unavailable: %s", exc)
                return self.is_loaded

            dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            if allow_download is None:
                can_download = os.environ.get("SONAVE_HF_VIDEO_DOWNLOAD", "0").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                )
            else:
                can_download = bool(allow_download)
            local_modes = [True, False] if can_download else [True]

            existing_by_id = {entry["model_id"]: entry for entry in self._ensemble}
            loaded_entries: list[dict[str, Any]] = list(self._ensemble)

            def _try_load_candidate(candidate: str) -> None:
                if not candidate or candidate in existing_by_id:
                    return
                if max_models is not None and len(loaded_entries) >= max_models:
                    return
                for local_only in local_modes:
                    try:
                        mdl = AutoModelForImageClassification.from_pretrained(
                            candidate, local_files_only=local_only
                        )
                        raw_sz = getattr(getattr(mdl, "config", None), "image_size", 224)
                        if isinstance(raw_sz, (list, tuple)):
                            img_size = int(raw_sz[0]) if raw_sz else 224
                        else:
                            img_size = int(raw_sz or 224)
                        try:
                            try:
                                proc = AutoImageProcessor.from_pretrained(
                                    candidate, local_files_only=local_only, use_fast=False
                                )
                            except TypeError:
                                proc = AutoImageProcessor.from_pretrained(
                                    candidate, local_files_only=local_only
                                )
                        except (ImportError, Exception):
                            proc = self._make_torch_vit_proc(img_size)
                        mdl.to(dev)
                        mdl.eval()
                        id2label = getattr(getattr(mdl, "config", None), "id2label", None)
                        fake_idx, real_idx, fake_lbl = self._resolve_label_mapping(id2label)
                        entry = {
                            "model_id": candidate,
                            "model": mdl,
                            "processor": proc,
                            "fake_indices": fake_idx,
                            "real_indices": real_idx,
                            "fake_label": fake_lbl,
                        }
                        loaded_entries.append(entry)
                        existing_by_id[candidate] = entry
                        logger.info(
                            "Loaded HuggingFace video deepfake ViT model=%s device=%s fake_idx=%s",
                            candidate,
                            dev,
                            fake_idx,
                        )
                        break
                    except Exception as exc:
                        logger.debug(
                            "HF ViT load attempt candidate=%s local_only=%s failed: %s",
                            candidate,
                            local_only,
                            exc,
                        )

            for candidate in self.model_ids:
                if max_models is not None and len(loaded_entries) >= max_models:
                    break
                _try_load_candidate(candidate)

            if not loaded_entries and self.fallback_model_id and self.fallback_model_id not in self.model_ids:
                _try_load_candidate(self.fallback_model_id)

            if loaded_entries:
                self._ensemble = loaded_entries
                self.loaded_model_ids = [e["model_id"] for e in loaded_entries]
                primary = loaded_entries[0]
                self._device = dev
                self.loaded_model_id = primary["model_id"]
                self._fake_indices = list(primary["fake_indices"])
                self._real_indices = list(primary["real_indices"])
                self._fake_label_name = primary["fake_label"]
                self._processor = primary["processor"]
                self._model = primary["model"]
                return True
            return False

    def _trigger_background_load(self) -> None:
        """Start a non-blocking daemon thread to load cached ViT weights without stalling live requests."""
        if self._load_attempted or self.is_loaded or not self.is_enabled():
            return
        in_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))
        force_hf = os.environ.get("SONAVE_FORCE_HF_VIDEO", "").strip() == "1"
        if in_pytest and not force_hf:
            return
        if getattr(self, "_bg_loading", False):
            return
        self._bg_loading = True

        def _bg_worker() -> None:
            try:
                time.sleep(0.005)
                # On live request paths, only load local cached weights and default to 1 primary
                # model on CPU unless SONAVE_VIDEO_HF_MODELS was explicitly configured.
                explicit_multi = bool(os.environ.get("SONAVE_VIDEO_HF_MODELS", "").strip())
                limit = None if explicit_multi else 1
                self.load(allow_download=False, max_models=limit)
            except Exception as exc:
                logger.debug("Background HF ViT load skipped: %s", exc)
            finally:
                self._bg_loading = False

        threading.Thread(target=_bg_worker, daemon=True, name="SonaveViTLoader").start()

    def _crop_face_pil(self, img: np.ndarray, face_info: dict[str, Any] | None = None) -> Any:
        """Extract a padded RGB PIL image (or uint8 RGB array fallback) around the participant face bbox."""
        h, w, _ = img.shape
        if face_info and face_info.get("face_detected") and "bbox" in face_info:
            y0, x0, y1, x1 = face_info["bbox"]
            fh, fw = max(4, y1 - y0), max(4, x1 - x0)
            pad_y, pad_x = int(fh * 0.15), int(fw * 0.15)
            cy0, cy1 = max(0, y0 - pad_y), min(h, y1 + pad_y)
            cx0, cx1 = max(0, x0 - pad_x), min(w, x1 + pad_x)
            crop = img[cy0:cy1, cx0:cx1] if (cy1 - cy0 >= 16 and cx1 - cx0 >= 16) else img
        else:
            crop = img

        uint8_rgb = np.clip(crop * 255.0, 0.0, 255.0).astype(np.uint8)
        try:
            from PIL import Image
            return Image.fromarray(uint8_rgb, mode="RGB")
        except Exception:
            return uint8_rgb

    @staticmethod
    def _infer_probs(model: Any, inputs: dict[str, Any]) -> np.ndarray:
        """Run forward pass and return 1D class probability array."""
        try:
            import torch
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                if hasattr(logits, "detach"):
                    return torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
                arr_l = np.asarray(logits, dtype=np.float64)[0]
                e_x = np.exp(arr_l - np.max(arr_l))
                return (e_x / np.sum(e_x)).astype(np.float32)
        except ImportError:
            outputs = model(**inputs)
            arr_l = np.asarray(outputs.logits, dtype=np.float64)[0]
            e_x = np.exp(arr_l - np.max(arr_l))
            return (e_x / np.sum(e_x)).astype(np.float32)

    def score(self, img: np.ndarray, face_info: dict[str, Any] | None = None) -> dict[str, Any]:
        """Score a decoded float32 RGB image [0, 1] across all loaded pretrained ViT ensemble models."""
        if not self.is_enabled():
            return {
                "vit_score": None,
                "vit_ensemble_scores": {},
                "vit_models_loaded": [],
                "hf_loaded": False,
                "hf_model_id": self.loaded_model_id or self.model_id,
                "device": self._device or "cpu",
            }
        if not self.is_loaded:
            if os.environ.get("SONAVE_SYNC_HF_VIDEO_LOAD", "").strip() == "1":
                self.load(allow_download=False, max_models=1)
            else:
                self._trigger_background_load()
            if not self.is_loaded:
                return {
                    "vit_score": None,
                    "vit_ensemble_scores": {},
                    "vit_models_loaded": [],
                    "hf_loaded": False,
                    "hf_model_id": self.loaded_model_id or self.model_id,
                    "device": self._device or "cpu",
                }

        try:
            pil_face = self._crop_face_pil(img, face_info)
            dev = self._device or "cpu"

            if self._ensemble and not (
                len(self._ensemble) == 1
                and self._model is not None
                and self._ensemble[0]["model"] is not self._model
            ):
                members = list(self._ensemble)
            else:
                members = [
                    {
                        "model_id": self.loaded_model_id or self.model_id,
                        "model": self._model,
                        "processor": self._processor,
                        "fake_indices": list(self._fake_indices),
                        "real_indices": list(self._real_indices),
                        "fake_label": self._fake_label_name,
                    }
                ]

            ensemble_scores: dict[str, float] = {}
            for member in members:
                m_id = member["model_id"]
                m_proc = member["processor"]
                m_mdl = member["model"]
                if m_proc is None or m_mdl is None:
                    continue
                try:
                    inputs = m_proc(images=pil_face, return_tensors="pt")
                    inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
                    probs = self._infer_probs(m_mdl, inputs)
                    n_classes = len(probs)
                    valid_fake = [i for i in member["fake_indices"] if 0 <= i < n_classes]
                    valid_real = [i for i in member["real_indices"] if 0 <= i < n_classes]

                    if valid_fake:
                        m_p = float(np.sum([probs[i] for i in valid_fake]))
                    elif valid_real:
                        m_p = float(1.0 - np.sum([probs[i] for i in valid_real]))
                    else:
                        m_p = float(probs[-1]) if n_classes > 0 else 0.0

                    ensemble_scores[m_id] = round(float(np.clip(m_p, 0.001, 0.999)), 4)
                except Exception as m_exc:
                    logger.warning("HuggingFace ViT member=%s inference failed: %s", m_id, m_exc)

            if not ensemble_scores:
                return {
                    "vit_score": None,
                    "vit_ensemble_scores": {},
                    "vit_models_loaded": [],
                    "hf_loaded": False,
                    "hf_model_id": self.loaded_model_id or self.model_id,
                    "device": str(dev),
                }

            scores_list = list(ensemble_scores.values())
            mean_p = float(np.mean(scores_list))
            max_p = float(np.max(scores_list))
            face_specialists = [
                v for k, v in ensemble_scores.items() if "dima806" in k or "wvolf" in k.lower()
            ]
            if face_specialists and max(face_specialists) < 0.22:
                p_fake = round(
                    float(
                        np.clip(
                            0.85 * float(np.mean(face_specialists)) + 0.15 * mean_p,
                            0.001,
                            0.999,
                        )
                    ),
                    4,
                )
            else:
                p_fake = round(float(np.clip(0.55 * mean_p + 0.45 * max_p, 0.001, 0.999)), 4)
            loaded_list = list(ensemble_scores.keys())

            return {
                "vit_score": p_fake,
                "vit_ensemble_scores": ensemble_scores,
                "vit_models_loaded": loaded_list,
                "hf_loaded": True,
                "hf_model_id": self.loaded_model_id or loaded_list[0],
                "device": str(dev),
                "fake_label": members[0].get("fake_label", self._fake_label_name),
            }
        except Exception as exc:
            logger.warning("HuggingFace ViT inference failed: %s", exc)
            return {
                "vit_score": None,
                "vit_ensemble_scores": {},
                "vit_models_loaded": [],
                "hf_loaded": False,
                "hf_model_id": self.loaded_model_id or self.model_id,
                "device": str(self._device or "cpu"),
            }


class VideoDeepfakeDetector:
    """End-to-end Google Meet video & multimodal deepfake detector."""

    def __init__(self, vit_backbone: HuggingFaceViTDeepfakeBackbone | None = None):
        self.vit_backbone = vit_backbone or HuggingFaceViTDeepfakeBackbone()
        self.spatial_net = SpatialFrequencyResidualNet()
        self.temporal_tracker = TemporalConsistencyTracker()
        self.av_analyzer = AudioVisualSyncAnalyzer()

    def reset_speaker(self, speaker_id: str | None = None) -> None:
        self.temporal_tracker.reset(speaker_id)
        self.av_analyzer.reset(speaker_id)
        reset_visual_synthesis_state(speaker_id)

    def score_frame(
        self,
        frame: bytes | np.ndarray,
        speaker_id: str = "default",
        audio_rms: float | None = None,
        client_metrics: dict[str, Any] | None = None,
        ts: float | None = None,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        """Decode and score a participant video frame, returning video_p / p_video_fake and forensic breakdown."""
        eff_ts = ts if ts is not None else timestamp
        cm = dict(client_metrics or {})
        apply_model = str(cm.get("apply_deepfake_model") or "").strip().lower()
        transformed_frame_b64: str | None = cm.get("transformed_frame_b64")

        img = decode_frame(frame)
        if apply_model and not cm.get("_already_synthesized"):
            syn_arr = synthesize_visual_deepfake(img, engine_id=apply_model, speaker_id=speaker_id)
            transformed_frame_b64 = encode_frame_b64(syn_arr, fmt="jpeg")
            img = decode_frame(syn_arr)
            cm.pop("mouth_delta", None)

        face_info = detect_face_region(img)
        vit_res = self.vit_backbone.score(img, face_info)
        spatial = self.spatial_net.analyze(img, face_info)
        temporal = self.temporal_tracker.update(speaker_id, img, face_info, ts=eff_ts)

        eff_mouth_delta = float(cm.get("mouth_delta", temporal["mouth_delta"]))
        eff_rms = float(cm["audio_rms"]) if "audio_rms" in cm and cm["audio_rms"] is not None else audio_rms
        eff_lag = float(cm["lag_ms"]) if "lag_ms" in cm and cm["lag_ms"] is not None else None

        av = self.av_analyzer.analyze(
            speaker_id=speaker_id,
            mouth_delta=eff_mouth_delta,
            audio_rms=eff_rms,
            client_av_lag_ms=eff_lag,
        )

        s_score = float(spatial["spatial_score"])
        f_score = float(spatial["frequency_seam_score"])
        seam_score = float(spatial["spatial_seam_score"])
        t_score = float(temporal["temporal_jitter_score"])
        av_sync_score = float(av["av_sync_score"])
        av_mismatch = float(av["av_mismatch_score"])
        vit_score: float | None = (
            float(vit_res["vit_score"]) if vit_res.get("vit_score") is not None else None
        )

        # Weighted ensemble combining pretrained ViT neural classifier (when active)
        # with spatial-frequency seam analysis, temporal consistency, and AV lip-sync.
        if vit_score is not None:
            forensic_max = max(s_score, f_score, seam_score, t_score, av_mismatch)
            uncorroborated_vit = (
                forensic_max < 0.28
                and vit_score < 0.90
                and not apply_model
                and not cm.get("simulated_attack")
            )
            if uncorroborated_vit:
                eff_vit = vit_score * min(1.0, (forensic_max / 0.28) ** 1.25)
                vit_score = round(eff_vit, 4)
            visual_core = 0.45 * vit_score + 0.30 * s_score + 0.25 * f_score
            weighted = (
                0.38 * vit_score
                + 0.26 * s_score
                + 0.20 * f_score
                + 0.08 * t_score
                + 0.08 * av_mismatch
            )
            peak_signal = max(
                vit_score * (0.65 if uncorroborated_vit else 0.95),
                s_score,
                f_score,
                seam_score,
                visual_core,
                t_score * 0.92,
                av_mismatch * 0.90,
            )
            p_video = max(0.45 * weighted + 0.55 * peak_signal, 0.88 * visual_core, 0.86 * peak_signal)
        else:
            visual_core = 0.55 * s_score + 0.45 * f_score
            weighted = 0.46 * s_score + 0.34 * f_score + 0.10 * t_score + 0.10 * av_mismatch
            peak_signal = max(s_score, f_score, seam_score, visual_core, t_score * 0.92, av_mismatch * 0.90)
            p_video = max(0.45 * weighted + 0.55 * peak_signal, 0.88 * visual_core, 0.86 * peak_signal)

        # Allow explicit red-team / client anomaly override when testing synthetic face-swaps
        if cm.get("simulated_attack") or cm.get("force_p_fake") is not None:
            p_video = float(cm.get("force_p_fake", cm.get("target_video_p", 0.94)))
            f_score = max(f_score, 0.91)
            s_score = max(s_score, 0.89)
            seam_score = max(seam_score, 0.91)
            if vit_score is not None:
                vit_score = max(vit_score, round(min(0.99, p_video * 1.01), 4))

        p_video = round(float(np.clip(p_video, 0.01, 0.995)), 4)
        v = video_verdict(p_video)
        conf = round(min(1.0, abs(p_video - TAU_VIDEO_REAL) / max(TAU_VIDEO_REAL, 1.0 - TAU_VIDEO_REAL)), 3)

        vit_ensemble_scores = dict(vit_res.get("vit_ensemble_scores") or {})
        vit_models_loaded = list(vit_res.get("vit_models_loaded") or [])

        visual_metrics = {
            "vit_score": round(vit_score, 4) if vit_score is not None else None,
            "vit_ensemble_scores": vit_ensemble_scores,
            "vit_models_loaded": vit_models_loaded,
            "hf_model_id": vit_res["hf_model_id"],
            "hf_loaded": bool(vit_res["hf_loaded"]),
            "spatial_score": round(s_score, 4),
            "spatial_seam_score": round(seam_score, 4),
            "frequency_seam_score": round(f_score, 4),
            "perioral_seam_score": spatial.get("perioral_seam_score", 0.02),
            "skin_smoothing_score": spatial["skin_smoothing_score"],
            "boundary_seam_ratio": spatial["boundary_seam_ratio"],
            "mouth_seam_ratio": spatial.get("mouth_seam_ratio", 1.0),
            "hf_energy_ratio": spatial["hf_energy_ratio"],
            "chroma_step": spatial.get("chroma_step", 0.0),
            "temporal_jitter_score": round(t_score, 4),
            "mouth_delta": round(eff_mouth_delta, 4),
            "av_sync_score": round(av_sync_score, 4),
            "lag_ms": av["lag_ms"],
            "face_detected": face_info["face_detected"],
            "face_coverage": face_info["face_coverage"],
            "engine_hint": cm.get("engine_id") or apply_model or "",
        }

        out_res = {
            "video_p": p_video,
            "p_video_fake": p_video,
            "verdict": v,
            "confidence": conf,
            "face_detected": face_info["face_detected"],
            "bbox": list(face_info["bbox"]),
            "vit_score": round(vit_score, 4) if vit_score is not None else None,
            "vit_ensemble_scores": vit_ensemble_scores,
            "vit_models_loaded": vit_models_loaded,
            "hf_model_id": vit_res["hf_model_id"],
            "hf_loaded": bool(vit_res["hf_loaded"]),
            "spatial_score": round(s_score, 4),
            "spatial_seam_score": round(seam_score, 4),
            "frequency_seam_score": round(f_score, 4),
            "perioral_seam_score": spatial.get("perioral_seam_score", 0.02),
            "temporal_jitter_score": round(t_score, 4),
            "av_sync_score": round(av_sync_score, 4),
            "mouth_delta": round(eff_mouth_delta, 4),
            "lag_ms": av["lag_ms"],
            "av_sync": av,
            "visual_metrics": visual_metrics,
            "model_version": MODEL_VIDEO_VERSION,
        }
        if transformed_frame_b64:
            out_res["transformed_frame_b64"] = transformed_frame_b64
        if apply_model:
            out_res["applied_deepfake_model"] = apply_model
        return out_res


_VIDEO_DETECTOR: VideoDeepfakeDetector | None = None
_DETECTOR_LOCK = threading.Lock()


def get_video_detector() -> VideoDeepfakeDetector:
    """Return process-wide singleton VideoDeepfakeDetector."""
    global _VIDEO_DETECTOR
    if _VIDEO_DETECTOR is None:
        with _DETECTOR_LOCK:
            if _VIDEO_DETECTOR is None:
                _VIDEO_DETECTOR = VideoDeepfakeDetector()
    return _VIDEO_DETECTOR


def warm_hf_video_backbone(allow_download: bool = True) -> dict[str, Any]:
    """Download (if needed) and warm the HuggingFace ViT video deepfake ensemble."""
    det = get_video_detector()
    ok = det.vit_backbone.load(allow_download=allow_download)
    return {
        "ok": ok,
        "hf_loaded": det.vit_backbone.is_loaded,
        "hf_model_id": det.vit_backbone.loaded_model_id or det.vit_backbone.model_id,
        "vit_models_loaded": list(det.vit_backbone.loaded_model_ids),
        "vit_models_configured": list(det.vit_backbone.model_ids),
        "device": det.vit_backbone._device or "cpu",
        "fake_label": det.vit_backbone._fake_label_name,
    }


def score_video_bytes(
    frame_bytes: bytes | np.ndarray,
    speaker_id: str = "default",
    audio_rms: float | None = None,
    client_metrics: dict[str, Any] | None = None,
    ts: float | None = None,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """Convenience function to score a single video frame."""
    return get_video_detector().score_frame(
        frame_bytes,
        speaker_id=speaker_id,
        audio_rms=audio_rms,
        client_metrics=client_metrics,
        ts=ts if ts is not None else timestamp,
    )


def fuse_multimodal_verdict(
    p_audio: float | None = None,
    p_video: float | None = None,
    av_sync_score: float | None = None,
    *,
    audio_p: float | None = None,
    video_p: float | None = None,
) -> dict[str, Any]:
    """Fuse voice deepfake probability and video deepfake probability into a unified authenticity verdict.

    Policy (approved): auth_p = max(audio_p, video_p), boosted slightly when both modalities
    or AV lip-sync show concurrent anomalies.
    """
    eff_audio = p_audio if p_audio is not None else audio_p
    eff_video = p_video if p_video is not None else video_p
    pa = float(eff_audio) if eff_audio is not None else 0.0
    pv = float(eff_video) if eff_video is not None else 0.0

    # av_sync_score is coherence in [0, 1] (lower = desynced)
    av_s = float(av_sync_score) if av_sync_score is not None else 0.95
    av_mismatch = max(0.0, 1.0 - av_s) if av_sync_score is not None else 0.0

    fused = max(pa, pv)
    if pa >= 0.45 and pv >= 0.45:
        fused = min(0.995, fused + 0.06)
    if av_mismatch >= 0.50:
        fused = max(fused, min(0.95, 0.55 * fused + 0.45 * av_mismatch))

    fused = round(float(np.clip(fused, 0.0, 1.0)), 4)
    if fused >= 0.70 or pv >= TAU_VIDEO_FAKE:
        v = "fake"
    elif fused >= 0.45:
        v = "suspect"
    else:
        v = "real"

    modality_trigger = "none"
    if v in ("fake", "suspect"):
        if pa >= 0.65 and pv >= TAU_VIDEO_FAKE:
            modality_trigger = "multimodal_voice_and_video"
        elif pv >= TAU_VIDEO_FAKE or (pv > pa and pv >= TAU_VIDEO_REAL):
            modality_trigger = "video_deepfake"
        else:
            modality_trigger = "voice_deepfake"

    return {
        "p_fake": fused,
        "audio_p": round(pa, 4) if eff_audio is not None else None,
        "video_p": round(pv, 4) if eff_video is not None else None,
        "p_audio_fake": round(pa, 4) if eff_audio is not None else None,
        "p_video_fake": round(pv, 4) if eff_video is not None else None,
        "av_sync_score": round(av_s, 4) if av_sync_score is not None else None,
        "verdict": v,
        "modality_trigger": modality_trigger,
    }


if __name__ == "__main__":
    import json
    import sys

    allow_dl = "--no-download" not in sys.argv
    info = warm_hf_video_backbone(allow_download=allow_dl)
    print(json.dumps(info, indent=2))
