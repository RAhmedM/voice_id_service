"""Per-call identification state machine with re-identification.

Life of a session:

  Connect
    -> receive 8 kHz int16 PCM chunks
    -> slice into 256-sample VAD frames
    -> each frame: VAD classifies as speech or silence
    -> speech frames (plus a small trailing silence pad) accumulate in a
       sliding window of the most recent IDENTIFICATION_WINDOW_SECONDS
    -> first identification fires once we have >= MIN_SPEECH_SECONDS AND
       an utterance ends (or the window fills); inference runs as a
       background task so audio capture is never blocked
    -> if ENABLE_REIDENTIFICATION, every additional REID_INTERVAL_SECONDS
       of speech triggers another identification on the current window.
       Each identification carries an incrementing `sequence` field.
    -> session ends on disconnect, idle timeout, or max-session timeout.

Protocol messages emitted (JSON over the same WebSocket):

  {"type": "ready", "call_id": "..."}
  {"type": "identification", "sequence": 0, "speaker": "...",
   "matched": true|false, "confidence": 0.73,
   "scores": {...}, "speech_seconds": 3.4}
  {"type": "error", "message": "..."}
"""
import asyncio
import logging
from typing import Awaitable, Callable, List, Set

import numpy as np

from . import config
from .audio import resample_to_target
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
        # Speech accumulates as a list of small arrays so the sliding window
        # can be trimmed without copying the entire buffer on every frame.
        self._speech_chunks: List[np.ndarray] = []
        self._speech_samples = 0  # samples currently in the sliding window
        # Cumulative speech samples since the last identification fired.
        # Drives the re-identification cadence; silence does not count.
        self._speech_since_last_id = 0
        self._consecutive_silence_frames = 0
        self._in_speech = False

        self._sequence = 0
        self._first_id_done = False
        self._identifying = False  # encoder currently running on a snapshot
        self._abort = False        # store empty / fatal error: stop trying
        self._tasks: Set[asyncio.Task] = set()

    # ----------------------------------------------------------------- API

    async def on_audio_chunk(self, pcm_bytes: bytes) -> None:
        """Accept a chunk of raw 8 kHz int16 little-endian PCM."""
        if self._abort:
            return

        audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_f32 = audio_i16.astype(np.float32) / 32768.0
        self._raw_buffer = np.concatenate([self._raw_buffer, audio_f32])

        # Silero VAD needs exact-size frames; slice them out.
        while len(self._raw_buffer) >= config.VAD_FRAME_SAMPLES:
            frame = self._raw_buffer[: config.VAD_FRAME_SAMPLES]
            self._raw_buffer = self._raw_buffer[config.VAD_FRAME_SAMPLES:]
            await self._process_frame(frame)
            if self._abort:
                return

    async def aclose(self) -> None:
        """Cancel any in-flight identification tasks."""
        for t in list(self._tasks):
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    # ----------------------------------------------------------------- core

    async def _process_frame(self, frame: np.ndarray) -> None:
        is_speech = self.vad.is_speech(frame)

        if is_speech:
            if not self._in_speech:
                log.info("[%s] speech started", self.call_id)
                self._in_speech = True
            self._speech_chunks.append(frame)
            self._speech_samples += len(frame)
            self._speech_since_last_id += len(frame)
            self._consecutive_silence_frames = 0
        else:
            self._consecutive_silence_frames += 1
            if self._in_speech and self._consecutive_silence_frames <= config.TRAILING_SILENCE_FRAMES:
                self._speech_chunks.append(frame)
                self._speech_samples += len(frame)
                # Trailing silence frames go into the window for a smoother
                # tail but they do NOT advance the re-id cadence.

        self._trim_window()

        if not self._first_id_done:
            self._maybe_fire_first_identification()
        elif config.ENABLE_REIDENTIFICATION:
            self._maybe_fire_reidentification()

    def _trim_window(self) -> None:
        """Drop oldest speech so the window holds at most IDENTIFICATION_WINDOW_SAMPLES."""
        limit = config.IDENTIFICATION_WINDOW_SAMPLES
        while self._speech_samples > limit and self._speech_chunks:
            front = self._speech_chunks[0]
            if self._speech_samples - len(front) >= limit:
                self._speech_chunks.pop(0)
                self._speech_samples -= len(front)
            else:
                excess = self._speech_samples - limit
                self._speech_chunks[0] = front[excess:]
                self._speech_samples -= excess
                break

    def _maybe_fire_first_identification(self) -> None:
        if self._identifying:
            return
        enough_speech = self._speech_samples >= config.MIN_SPEECH_SAMPLES
        if not enough_speech:
            return
        utterance_ended = (
            self._in_speech
            and self._consecutive_silence_frames >= config.END_OF_UTTERANCE_SILENCE_FRAMES
        )
        window_full = self._speech_samples >= config.IDENTIFICATION_WINDOW_SAMPLES
        if utterance_ended or window_full:
            self._launch_identification()

    def _maybe_fire_reidentification(self) -> None:
        if self._identifying:
            return
        if self._speech_since_last_id < config.REID_INTERVAL_SAMPLES:
            return
        if self._speech_samples < config.MIN_SPEECH_SAMPLES:
            return
        self._launch_identification()

    def _launch_identification(self) -> None:
        # Snapshot the current window. np.concatenate copies, so the encoder
        # can run safely while _process_frame keeps mutating the buffer.
        if not self._speech_chunks:
            return
        snapshot = np.concatenate(self._speech_chunks)
        seq = self._sequence
        self._sequence += 1
        self._speech_since_last_id = 0
        self._first_id_done = True
        self._identifying = True

        task = asyncio.create_task(self._do_identify(snapshot, seq))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _do_identify(self, speech_8k: np.ndarray, sequence: int) -> None:
        speech_seconds = len(speech_8k) / config.INPUT_SAMPLE_RATE
        log.info(
            "[%s] identification #%d on %.2fs of speech",
            self.call_id, sequence, speech_seconds,
        )
        try:
            if self.store.is_empty():
                # Re-id would spam this on every cadence tick; abort the session.
                self._abort = True
                await self.send_json({"type": "error", "message": "no speakers enrolled"})
                return

            speech_16k = resample_to_target(speech_8k, config.INPUT_SAMPLE_RATE)
            embedding = await asyncio.to_thread(self.encoder.embed, speech_16k)
            result = self.store.identify(embedding, config.IDENTIFICATION_THRESHOLD)

            matched = result["speaker"] is not None
            await self.send_json({
                "type": "identification",
                "sequence": sequence,
                "speaker": result["speaker"] or "unknown",
                "matched": matched,
                "confidence": result["confidence"],
                "scores": result["scores"],
                "speech_seconds": speech_seconds,
            })
            log.info(
                "[%s] verdict #%d=%s confidence=%.3f",
                self.call_id, sequence,
                result["speaker"] or "unknown",
                result["confidence"],
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("[%s] identification #%d failed", self.call_id, sequence)
            try:
                await self.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            self._identifying = False
