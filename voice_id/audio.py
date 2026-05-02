"""Audio helpers shared by the enrollment and identification paths.

Keeping these in one place is deliberate: any divergence between how enrollment
audio and inference audio are processed silently degrades accuracy. Resample,
high-pass, and level-normalize the same way on both sides.
"""
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

from . import config


# 4th-order Butterworth HPF designed once at import time.
_HPF_SOS = butter(4, config.HPF_CUTOFF_HZ, btype="highpass",
                  fs=config.TARGET_SAMPLE_RATE, output="sos")


def resample_to_target(audio: np.ndarray, sr: int) -> np.ndarray:
    """Polyphase resample to TARGET_SAMPLE_RATE with proper anti-aliasing."""
    if sr == config.TARGET_SAMPLE_RATE:
        return audio.astype(np.float32, copy=False)
    g = gcd(sr, config.TARGET_SAMPLE_RATE)
    up = config.TARGET_SAMPLE_RATE // g
    down = sr // g
    return resample_poly(audio, up, down).astype(np.float32)


def hpf(audio_16k: np.ndarray) -> np.ndarray:
    """Zero-phase 80 Hz high-pass to strip DC offset and low-frequency rumble."""
    if len(audio_16k) < 33:
        # sosfiltfilt needs more samples than the filter's padlen.
        return audio_16k.astype(np.float32, copy=False)
    return sosfiltfilt(_HPF_SOS, audio_16k).astype(np.float32)


def rms_normalize(audio: np.ndarray, target: float = None) -> np.ndarray:
    """Scale audio so its RMS equals `target`, clamped below 0.99 peak."""
    if target is None:
        target = config.TARGET_RMS
    audio = audio.astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(audio * audio)))
    if rms < 1e-6:
        return audio
    out = audio * (target / rms)
    peak = float(np.max(np.abs(out)))
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def preprocess_for_embed(audio_16k: np.ndarray) -> np.ndarray:
    """Pipeline applied to every clip going into the speaker encoder."""
    return rms_normalize(hpf(audio_16k))


def load_file_to_target(path: Path) -> np.ndarray:
    """Read an audio file and return 1-D float32 at TARGET_SAMPLE_RATE mono."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return resample_to_target(audio.astype(np.float32), sr)
