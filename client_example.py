"""Quick test client.

Streams an 8 kHz mono WAV to the service in real-time 20 ms chunks, exactly
the way Asterisk would, and prints whatever the server sends back.

Usage:
    python client_example.py path/to/sample_8k.wav
    python client_example.py path/to/sample_8k.wav ws://localhost:8000/ws/identify/my-call
"""
import asyncio
import json
import sys

import numpy as np
import soundfile as sf
import websockets


CHUNK_MS = 20
SAMPLE_RATE = 8000
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)


async def run(wav_path: str, uri: str):
    audio, sr = sf.read(wav_path, dtype="int16")
    if sr != SAMPLE_RATE:
        print(f"warning: file is {sr} Hz, service expects {SAMPLE_RATE} Hz. "
              "Resample with `sox input.wav -r 8000 out.wav` for a realistic test.")
    if audio.ndim > 1:
        audio = audio[:, 0]

    async with websockets.connect(uri) as ws:

        async def receiver():
            try:
                async for raw in ws:
                    print("<-", json.dumps(json.loads(raw), indent=2))
            except websockets.ConnectionClosed:
                pass

        recv_task = asyncio.create_task(receiver())

        for i in range(0, len(audio), CHUNK_SAMPLES):
            chunk = audio[i:i + CHUNK_SAMPLES].tobytes()
            await ws.send(chunk)
            await asyncio.sleep(CHUNK_MS / 1000)  # pace like a real call

        # Let the server finish processing + emit result
        try:
            await asyncio.wait_for(recv_task, timeout=5.0)
        except asyncio.TimeoutError:
            pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python client_example.py <wav_path> [ws_uri]")
        sys.exit(1)
    wav = sys.argv[1]
    uri = sys.argv[2] if len(sys.argv) > 2 else "ws://localhost:8000/ws/identify/test-call-1"
    asyncio.run(run(wav, uri))
