"""FastAPI service.

Routes:
  POST   /enroll                          multipart upload, add a speaker
  GET    /speakers                        list enrolled names
  DELETE /speakers/{name}                 remove an enrolled speaker
  GET    /health                          liveness + speaker count
  WS     /ws/identify/{call_id}           per-call identification

Run:
  uvicorn voice_id.main:app --host 0.0.0.0 --port 8000
"""
import asyncio
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
import torchaudio
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    UploadFile, File, Form, HTTPException,
)

from . import config
from .encoder import SpeakerEncoder
from .session import IdentificationSession
from .storage import VoiceprintStore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("voice_id")


# Globals populated on startup. Using app.state would be more idiomatic but
# complicates imports; globals are fine for a single-process service.
encoder: SpeakerEncoder = None
store: VoiceprintStore = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder, store
    log.info("loading speaker encoder (this takes a few seconds on first run)...")
    encoder = SpeakerEncoder()
    store = VoiceprintStore(config.VOICEPRINT_DB_PATH)
    log.info(
        "ready. device=%s, speakers=%s",
        config.DEVICE, store.list_names() or "[]",
    )
    yield


app = FastAPI(title="Voice ID Service", lifespan=lifespan)


def _load_any_audio_to_16k_mono(path: Path) -> np.ndarray:
    """Read an audio file and return 1-D float32 at 16 kHz mono."""
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != config.TARGET_SAMPLE_RATE:
        g = gcd(sr, config.TARGET_SAMPLE_RATE)
        up = config.TARGET_SAMPLE_RATE // g
        down = sr // g
        audio = resample_poly(audio, up, down).astype(np.float32)
    return audio

# ---------------------------------------------------------------- HTTP routes

@app.post("/enroll")
async def enroll(
    name: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Enroll a speaker from one or more audio clips.

    Each clip is decoded, resampled to 16 kHz mono, embedded, and the mean of
    all embeddings (L2-normalized) becomes the stored centroid.
    """
    if not files:
        raise HTTPException(400, "at least one audio file required")

    embeddings = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, f in enumerate(files):
            dest = tmp_dir / (f.filename or f"clip_{i}.wav")
            dest.write_bytes(await f.read())
            try:
                audio_16k = _load_any_audio_to_16k_mono(dest)
            except Exception as e:
                raise HTTPException(400, f"failed to read {f.filename}: {e}")
            emb = await asyncio.to_thread(encoder.embed, audio_16k)
            embeddings.append(emb)

    centroid = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    store.add(name, centroid)

    log.info("enrolled '%s' from %d clips", name, len(files))
    return {
        "name": name,
        "clips_used": len(files),
        "total_speakers": len(store.list_names()),
    }


@app.get("/speakers")
def list_speakers():
    return {"speakers": store.list_names()}


@app.delete("/speakers/{name}")
def delete_speaker(name: str):
    if not store.remove(name):
        raise HTTPException(404, f"no speaker '{name}'")
    return {"removed": name}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "speakers_enrolled": len(store.list_names()),
        "device": config.DEVICE,
    }


# ---------------------------------------------------------------- WebSocket

@app.websocket("/ws/identify/{call_id}")
async def ws_identify(websocket: WebSocket, call_id: str):
    """One connection = one call.

    Client sends raw 8 kHz int16 LE PCM as binary frames.
    Server sends JSON text frames (see session.py for protocol).
    """
    await websocket.accept()
    log.info("[%s] websocket connected", call_id)

    async def send_json(msg: dict):
        try:
            await websocket.send_json(msg)
        except Exception:
            log.exception("[%s] failed to send json", call_id)

    session = IdentificationSession(call_id, encoder, store, send_json)
    await send_json({"type": "ready", "call_id": call_id})

    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            # Binary frame = audio chunk
            if msg.get("bytes") is not None:
                await session.on_audio_chunk(msg["bytes"])
            # Text frames reserved for future control messages (ignored for now)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("[%s] websocket error", call_id)
    finally:
        log.info("[%s] websocket closed", call_id)
