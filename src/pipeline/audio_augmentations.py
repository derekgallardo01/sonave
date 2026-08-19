"""src/pipeline/audio_augmentations.py — Meeting Audio Simulator & DSP Augmentation Engine.

Simulates the real-world acoustic degradations of meeting platforms:
  - Lossy Opus/WebRTC low-bitrate compression (6–24 kbps)
  - Packet Loss Concealment (PLC) burst dropouts
  - Room Impulse Response (RIR) acoustic reverberation
  - Background office/ambient noise mixing with randomized SNR
"""
from __future__ import annotations

import math
import random
import numpy as np
import torch
from typing import Optional


class MeetingAudioAugmentor:
    """Applies realistic meeting room and VoIP network degradations to audio tensors."""

    def __init__(self, sample_rate: int = 16000,
                 p_opus: float = 0.60,
                 p_plc: float = 0.40,
                 p_reverb: float = 0.50,
                 p_noise: float = 0.50):
        self.sample_rate = sample_rate
        self.p_opus = p_opus
        self.p_plc = p_plc
        self.p_reverb = p_reverb
        self.p_noise = p_noise

    def apply_packet_loss(self, audio: np.ndarray) -> np.ndarray:
        """Simulate WebRTC Packet Loss Concealment (PLC) with 20-60ms burst dropouts."""
        out = audio.copy()
        num_drops = random.randint(1, 3)
        for _ in range(num_drops):
            drop_len = int(self.sample_rate * random.uniform(0.02, 0.06)) # 20-60ms
            if drop_len < len(out):
                start = random.randint(0, len(out) - drop_len)
                # Attenuate and smooth edges to simulate PLC interpolation
                fade = np.linspace(1.0, 0.05, drop_len // 2)
                fade_out = np.concatenate([fade, fade[::-1]])
                out[start:start + len(fade_out)] *= fade_out
        return out

    def apply_reverb(self, audio: np.ndarray) -> np.ndarray:
        """Simulate room acoustics via synthetic exponential decay impulse response."""
        decay_time = random.uniform(0.08, 0.25) # 80ms - 250ms RT60
        decay_samples = int(self.sample_rate * decay_time)
        t = np.linspace(0, 1, decay_samples)
        rir = np.random.randn(decay_samples) * np.exp(-6.0 * t)
        rir = rir / (np.max(np.abs(rir)) + 1e-6)
        
        # Convolve and wet/dry mix
        convolved = np.convolve(audio, rir, mode="full")[:len(audio)]
        wet_dry = random.uniform(0.15, 0.35)
        return (1.0 - wet_dry) * audio + wet_dry * convolved

    def apply_opus_simulation(self, audio: np.ndarray) -> np.ndarray:
        """Simulate low-bitrate Opus MDCT quantization artifacts and band-limiting."""
        # Lowpass filter above 7.5 kHz (Opus wideband cutoff)
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sample_rate)
        
        # Bandpass filter and high-frequency attenuation
        mask = np.ones_like(fft, dtype=np.float32)
        mask[freqs > 7500] *= 0.1
        
        # Quantization noise simulation
        quant_level = random.uniform(0.002, 0.01)
        quantized_fft = np.round(fft.real / quant_level) * quant_level + 1j * (np.round(fft.imag / quant_level) * quant_level)
        return np.fft.irfft(quantized_fft * mask, n=len(audio)).astype(np.float32)

    def apply_background_noise(self, audio: np.ndarray) -> np.ndarray:
        """Mix subtle background HVAC/office pink noise with realistic SNR (15–30 dB)."""
        snr_db = random.uniform(18.0, 32.0)
        signal_power = np.mean(audio ** 2) + 1e-8
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        
        # Pink noise approximation (1/f)
        white = np.random.randn(len(audio))
        pink = np.cumsum(white)
        pink = pink - np.mean(pink)
        pink = pink / (np.std(pink) + 1e-6) * np.sqrt(noise_power)
        
        return audio + pink.astype(np.float32)

    def augment(self, audio: np.ndarray | torch.Tensor) -> np.ndarray:
        """Apply a full randomized chain of meeting audio augmentations."""
        is_torch = isinstance(audio, torch.Tensor)
        arr = audio.cpu().numpy() if is_torch else np.array(audio, dtype=np.float32)
        
        if random.random() < self.p_plc:
            arr = self.apply_packet_loss(arr)
        if random.random() < self.p_opus:
            arr = self.apply_opus_simulation(arr)
        if random.random() < self.p_reverb:
            arr = self.apply_reverb(arr)
        if random.random() < self.p_noise:
            arr = self.apply_background_noise(arr)
            
        # Peak normalization
        max_val = np.max(np.abs(arr)) + 1e-6
        if max_val > 1.0:
            arr = arr / max_val
            
        return arr
