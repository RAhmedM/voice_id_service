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
from typing import List, Optional

import numpy as np
import torch
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    UploadFile, File, Form, HTTPException,
)

from . import config
from .audio import load_file_to_target
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
encoder: Optional[SpeakerEncoder] = None
store: Optional[VoiceprintStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder, store
    # PyTorch on CPU spawns OMP threads internally; with asyncio.to_thread
    # several inferences can run in parallel and trample each other's cores.
    # Pin to one OMP thread per inference and let asyncio's pool provide
    # parallelism instead.
    torch.set_num_threads(1)
    log.info("loading speaker encoder (this takes a few seconds on first run)...")
    encoder = SpeakerEncoder()
    store = VoiceprintStore(config.VOICEPRINT_DB_PATH)
    log.info(
        "ready. device=%s, speakers=%s",
        config.DEVICE, store.list_names() or "[]",
    )
    yield


app = FastAPI(title="Voice ID Service", lifespan=lifespan)


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
                audio_16k = load_file_to_target(dest)
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
    Server sends JSON text frames (see session.py for protocol). With
    re-identification enabled the server keeps emitting `identification`
    events for the duration of the call. The connection closes on
    disconnect, idle timeout, or the absolute session ceiling.
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

    loop = asyncio.get_event_loop()
    deadline = loop.time() + config.WS_MAX_SESSION_S

    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                log.info("[%s] max session length reached, closing", call_id)
                break
            timeout = min(config.WS_IDLE_TIMEOUT_S, remaining)
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=timeout)
            except asyncio.TimeoutError:
                log.info("[%s] idle timeout, closing", call_id)
                break

            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                await session.on_audio_chunk(msg["bytes"])
            # Text frames reserved for future control messages (ignored for now)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("[%s] websocket error", call_id)
    finally:
        await session.aclose()
        try:
            await websocket.close()
        except Exception:
            pass
        log.info("[%s] websocket closed", call_id)
