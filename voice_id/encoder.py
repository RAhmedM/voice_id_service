"""ECAPA-TDNN speaker encoder wrapper.

One instance is shared across all sessions. Inference is stateless and safe
to call concurrently. Preprocessing (HPF + RMS-normalize) is applied here so
the enrollment and inference paths cannot diverge.
"""
import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier

from . import config
from .audio import preprocess_for_embed


class SpeakerEncoder:
    def __init__(self):
        self.device = config.DEVICE
        self.model = EncoderClassifier.from_hparams(
            source=config.MODEL_NAME,
            savedir=config.MODEL_SAVEDIR,
            run_opts={"device": self.device},
        )

    def embed(self, audio_16k_f32: np.ndarray) -> np.ndarray:
        """Convert 16 kHz float32 mono audio into a 192-dim L2-normalized embedding."""
        if audio_16k_f32.ndim != 1:
            raise ValueError(f"expected 1-D audio, got shape {audio_16k_f32.shape}")

        audio_16k_f32 = preprocess_for_embed(audio_16k_f32)

        signal = torch.from_numpy(audio_16k_f32).float().unsqueeze(0)
        if self.device != "cpu":
            signal = signal.to(self.device)

        with torch.no_grad():
            emb = self.model.encode_batch(signal).squeeze().cpu().numpy()

        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb
