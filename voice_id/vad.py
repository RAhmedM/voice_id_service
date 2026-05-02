"""Silero VAD wrapper.

Silero VAD keeps internal LSTM state between calls, so each call needs its
own instance. Loading is cheap after the first time (cached on disk).
"""
import numpy as np
import torch
from silero_vad import load_silero_vad

from . import config


class SpeechDetector:
    def __init__(self, threshold: float = None):
        self.model = load_silero_vad()
        self.threshold = threshold if threshold is not None else config.VAD_THRESHOLD
        self.sample_rate = config.INPUT_SAMPLE_RATE
        self.frame_size = config.VAD_FRAME_SAMPLES

    def is_speech(self, frame_f32: np.ndarray) -> bool:
        """`frame_f32` must be exactly VAD_FRAME_SAMPLES long (256 samples at 8 kHz)."""
        if len(frame_f32) != self.frame_size:
            raise ValueError(
                f"VAD expects exactly {self.frame_size} samples, got {len(frame_f32)}"
            )
        tensor = torch.from_numpy(frame_f32).float()
        with torch.no_grad():
            prob = float(self.model(tensor, self.sample_rate).item())
        return prob >= self.threshold
