"""Per-call identification state machine.

Life of a session:

  Connect
    -> receive 8 kHz int16 PCM chunks
    -> slice into 256-sample VAD frames
    -> each frame: VAD classifies as speech or silence
    -> speech frames (plus small trailing silence) accumulate in speech_buffer
    -> when we have >= MIN_SPEECH_SECONDS AND an utterance ends (or hit MAX),
       upsample to 16 kHz, embed, match against DB, emit one identification.
    -> further audio is ignored until the client reconnects.

Protocol messages emitted (JSON over the same WebSocket):

  {"type": "ready", "call_id": "..."}              - sent on connect
  {"type": "identification", "speaker": "...",
   "matched": true|false, "confidence": 0.73,
   "scores": {...}, "speech_seconds": 3.4}         - one per call, when ready
  {"type": "error", "message": "..."}              - on fatal failure
"""
import asyncio
import logging
from typing import Awaitable, Callable

import numpy as np

from . import config
from .encoder import SpeakerEncoder
from .storage import VoiceprintStore
from .vad import SpeechDetector


log = logging.getLogger(__name__)


SendJson = Callable[[dict], Awaitable[None]]


class IdentificationSession:
    def __init__(
        self,
        call_id: str,
        encoder: SpeakerEncoder,
        store: VoiceprintStore,
        send_json: SendJson,
    ):
        self.call_id = call_id
        self.encoder = encoder
        self.store = store
        self.send_json = send_json
        self.vad = SpeechDetector()

        self._raw_buffer = np.array([], dtype=np.float32)
        self._speech_buffer = np.array([], dtype=np.float32)
        self._consecutive_silence_frames = 0
        self._in_speech = False
        self._identified = False
        self._identify_lock = asyncio.Lock()

    async def on_audio_chunk(self, pcm_bytes: bytes) -> None:
        """Accept a chunk of raw 8 kHz int16 little-endian PCM."""
        if self._identified:
            return

        # int16 LE -> float32 [-1, 1]
        audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_f32 = audio_i16.astype(np.float32) / 32768.0
        self._raw_buffer = np.concatenate([self._raw_buffer, audio_f32])

        # Silero VAD needs exact-size frames; slice them out.
        while len(self._raw_buffer) >= config.VAD_FRAME_SAMPLES:
            frame = self._raw_buffer[: config.VAD_FRAME_SAMPLES]
            self._raw_buffer = self._raw_buffer[config.VAD_FRAME_SAMPLES:]
            await self._process_frame(frame)
            if self._identified:
                return

    async def _process_frame(self, frame: np.ndarray) -> None:
        is_speech = self.vad.is_speech(frame)

        if is_speech:
            if not self._in_speech:
                log.info("[%s] speech started", self.call_id)
                self._in_speech = True
            self._speech_buffer = np.concatenate([self._speech_buffer, frame])
            self._consecutive_silence_frames = 0
        else:
            self._consecutive_silence_frames += 1
            # Keep a few frames of trailing silence so the utterance isn't chopped mid-word.
            if self._in_speech and self._consecutive_silence_frames <= config.TRAILING_SILENCE_FRAMES:
                self._speech_buffer = np.concatenate([self._speech_buffer, frame])

        speech_samples = len(self._speech_buffer)
        enough_speech = speech_samples >= config.MIN_SPEECH_SAMPLES
        utterance_ended = (
            self._in_speech
            and self._consecutive_silence_frames >= config.END_OF_UTTERANCE_SILENCE_FRAMES
        )
        hit_max = speech_samples >= config.MAX_SPEECH_SAMPLES

        if enough_speech and (utterance_ended or hit_max):
            await self._run_identification()

    async def _run_identification(self) -> None:
        # Guard against re-entry if multiple triggers fire back-to-back.
        async with self._identify_lock:
            if self._identified:
                return
            self._identified = True

        speech_seconds = len(self._speech_buffer) / config.INPUT_SAMPLE_RATE
        log.info(
            "[%s] identifying on %.2fs of speech",
            self.call_id, speech_seconds,
        )

        if self.store.is_empty():
            await self.send_json({
                "type": "error",
                "message": "no speakers enrolled",
            })
            return

        # Upsample the accumulated speech buffer all at once (not per-chunk) so
        # we don't get boundary discontinuities. Linear interp to match your
        # Asterisk-side upsampler exactly.
        speech_16k = self._upsample_linear(self._speech_buffer)

        try:
            # Model inference can be slow; run it off the event loop.
            embedding = await asyncio.to_thread(self.encoder.embed, speech_16k)
            result = self.store.identify(embedding, config.IDENTIFICATION_THRESHOLD)
        except Exception as e:
            log.exception("[%s] identification failed", self.call_id)
            await self.send_json({"type": "error", "message": str(e)})
            return

        matched = result["speaker"] is not None
        await self.send_json({
            "type": "identification",
            "speaker": result["speaker"] or "unknown",
            "matched": matched,
            "confidence": result["confidence"],
            "scores": result["scores"],
            "speech_seconds": speech_seconds,
        })
        log.info(
            "[%s] verdict=%s confidence=%.3f",
            self.call_id,
            result["speaker"] or "unknown",
            result["confidence"],
        )

    @staticmethod
    def _upsample_linear(audio_8k: np.ndarray) -> np.ndarray:
        """Linear-interp upsample 8 kHz -> 16 kHz. Matches the Asterisk-side upsampler."""
        n = len(audio_8k)
        x_old = np.arange(n)
        x_new = np.linspace(0, n - 1, n * 2)
        return np.interp(x_new, x_old, audio_8k).astype(np.float32)
