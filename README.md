# Voice ID Service

FastAPI service that identifies the speaker on an Asterisk phone call in real time using ECAPA-TDNN embeddings.

## What it does

- **HTTP** endpoints for enrolling speakers from audio files.
- **WebSocket** endpoint that accepts live 8 kHz PCM audio from Asterisk (one connection per call) and pushes back a single identification result as soon as it has enough speech.

## Architecture

```
    Asterisk EAGI                         This service
  ─────────────────          ───────────────────────────────────────
  8 kHz int16 PCM   ─WS─>    buffer ─> Silero VAD ─> speech accumulator
                                                              │
                                      (≥3s speech + silence)  ▼
                                                       linear upsample → 16 kHz
                                                              ▼
                                                       ECAPA-TDNN encoder
                                                              ▼
                                                    cosine vs. voiceprint DB
                                                              ▼
                             <─WS─   {"type":"identification", ...}
```

## Setup

```bash
cd voice_id_service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads ~80 MB of ECAPA weights and a small Silero VAD model.

## Run the service

```bash
uvicorn voice_id.main:app --host 0.0.0.0 --port 8000
```

Env vars (optional):
- `DEVICE=cuda` to use GPU
- `VOICEPRINT_DB_PATH=/var/lib/voiceid/db.pkl` to relocate the DB

## Enroll speakers

```bash
# 3 clips of Alice
curl -X POST http://localhost:8000/enroll \
  -F "name=alice" \
  -F "files=@alice_1.wav" \
  -F "files=@alice_2.wav" \
  -F "files=@alice_3.wav"

# list / remove
curl http://localhost:8000/speakers
curl -X DELETE http://localhost:8000/speakers/alice
```

**Enrollment audio tips:**
- 3–5 clips, 5–15 seconds of clean speech each.
- **Record through the same phone pipeline you'll serve through**. Mismatched channels (studio mic for enrollment, telephone for inference) is the single biggest accuracy killer.
- Any format torchaudio can decode works (wav, flac, mp3 with ffmpeg installed).

## WebSocket protocol

Connect: `ws://host:8000/ws/identify/{call_id}`

Client (Asterisk-side) sends:
- **Binary frames**: raw 8 kHz little-endian int16 PCM. Any chunk size — the server will re-frame for VAD. Typical Asterisk chunk is 160 samples (20 ms) = 320 bytes.
- Text frames ignored for V1 (reserved for future control messages).

Server sends JSON text frames:

```json
// on connect
{"type": "ready", "call_id": "..."}

// once we have enough speech — exactly one of these per call
{
  "type": "identification",
  "speaker": "alice",            // name if matched, "unknown" otherwise
  "matched": true,
  "confidence": 0.724,           // cosine similarity to the best match
  "scores": {                    // full ranking for debugging / logging
    "alice": 0.724,
    "bob":   0.418,
    "carol": 0.305
  },
  "speech_seconds": 3.42
}

// on any error
{"type": "error", "message": "no speakers enrolled"}
```

After emitting the identification, the server ignores further audio on that connection. Close the WebSocket when the call ends.

## Test it quickly

Terminal 1:
```bash
uvicorn voice_id.main:app --reload
```

Terminal 2:
```bash
# Enroll at least one speaker first (see above)
python client_example.py path/to/8khz_sample.wav
```

The client streams the WAV in 20 ms chunks in real time, so you'll see nothing for ~3 seconds, then the identification JSON.

## Asterisk integration sketch

On your side, in the EAGI script that already produces 8 kHz int16 chunks:

```python
import asyncio, websockets

async def stream_to_id_service(call_id, audio_queue):
    uri = f"ws://voice-id-host:8000/ws/identify/{call_id}"
    async with websockets.connect(uri) as ws:
        # Fire-and-listen: send audio, print results
        async def recv():
            async for msg in ws:
                # Pass the parsed JSON to your agent logic
                handle_identification(json.loads(msg))

        asyncio.create_task(recv())

        while True:
            chunk_8k_int16 = await audio_queue.get()  # from EAGI
            if chunk_8k_int16 is None:
                break
            await ws.send(chunk_8k_int16)  # send raw bytes, don't upsample
```

Note: the service does its own upsampling server-side using the same linear-interp formula you had in your EAGI code. **Don't upsample client-side** — send raw 8 kHz bytes and save the bandwidth.

## Tuning knobs

All in `voice_id/config.py`:

| Setting | Default | What it does |
|---|---|---|
| `MIN_SPEECH_SECONDS` | 3.0 | Minimum speech before attempting ID |
| `MAX_SPEECH_SECONDS` | 6.0 | Force ID once buffered this much |
| `END_OF_UTTERANCE_SILENCE_MS` | 500 | Silence that triggers end-of-utterance |
| `VAD_THRESHOLD` | 0.5 | Silero speech probability cutoff |
| `IDENTIFICATION_THRESHOLD` | 0.50 | Cosine similarity cutoff for non-"unknown" |

**The identification threshold is the one that matters most.** Calibrate it properly once you have real phone-call data:

1. Collect ~20 genuine pairs (same person, different recordings) → genuine score distribution.
2. Collect ~100 imposter pairs (different people) → imposter score distribution.
3. Plot both. Pick the threshold where they separate cleanly.
4. Phone audio tends to need a *lower* threshold than clean wideband (expect 0.35–0.45 instead of 0.45–0.55).

## What's NOT in V1, on purpose

Things to add later as you see real behavior:

- **Re-identification mid-call** (currently identifies once then stops). Useful for handoff detection or spoofing protection.
- **Per-call timeout** if the caller never speaks.
- **Better upsampling** (current linear interp aliases — `scipy.signal.resample_poly` would be a drop-in improvement).
- **Metrics / tracing** (Prometheus endpoint, OpenTelemetry).
- **Auth** — this service has none; put it behind an internal firewall or add an API key middleware.
- **Vector DB** — at a handful of speakers, a dict is faster than any database. Swap `VoiceprintStore` when you scale past ~10k.
- **Fine-tuning** on your real calls — not needed yet, but if accuracy plateaus, ECAPA fine-tunes well on a few hundred utterances.
