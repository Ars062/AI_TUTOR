# Realtime layer — Pipecat voice agent

## Architecture

```text
User mic (LiveKit WebRTC)
  → faster-whisper STT (CPU, int8)
  → KG-RAG tutor (Neo4j + FAISS + Groq)
  → Windows System.Speech TTS (offline)
  → LiveKit WebRTC → User speakers
```

## Files

| File | Purpose |
|------|---------|
| `agent.py` | Pipecat voice agent (LiveKit transport + full pipeline) |
| `stt.py` | faster-whisper STT wrapper |
| `tts.py` | TTS provider abstraction (SAPI / Piper) |
| `vision.py` | Webcam frame capture + context |
| `pipeline.py` | HTTP-based realtime pipeline (fallback) |

## Running the agent

```bash
# 1. Start LiveKit server (dev mode)
tools/livekit/livekit-server.exe --dev

# 2. Start the FastAPI backend
python -m uvicorn backend.main:app --port 8000

# 3. Start the Pipecat agent
python -m realtime.agent --room tutor-room

# 4. Open frontend, go to Live Room, click Join
```

## What happens

1. Frontend joins LiveKit room → mic + camera active
2. Pipecat agent joins same room as "AI-Tutor" bot
3. User speaks → LiveKit streams audio → agent receives it
4. Agent: faster-whisper transcribes → tutor generates answer → TTS speaks
5. User hears the tutor's voice reply through LiveKit

## GPU laptop (next phase)

When you move to the GPU machine, replace:
- `WindowsTTSProcessor` → **Piper** or **Coqui TTS** (neural voice)
- Add **MuseTalk** avatar processor (video output)
- Optionally swap `WhisperSTTService` for **Whisper large-v3** (GPU)
