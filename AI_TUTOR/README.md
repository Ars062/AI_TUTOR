# AI Tutor — Multimodal Voice Chatbot

Knowledge-grounded AI tutoring with real-time voice conversation.

## What's Built (all working)

| Feature | Status | How it works |
|---------|--------|-------------|
| **KG-RAG tutor** | ✅ | Neo4j + FAISS + Groq (gpt-oss-120b) |
| **Chain-of-Thought** | ✅ | Steps-first, inline, §4.2 compliant |
| **STT (speech→text)** | ✅ | faster-whisper (base, CPU, int8) |
| **TTS (text→speech)** | ✅ | Windows System.Speech (offline) |
| **LiveKit WebRTC** | ✅ | mic/camera room, server-issued tokens |
| **Vision capture** | ✅ | webcam frames → visual context |
| **RAG upload** | ✅ | PDF/TXT → chunk → embed → tutor learns |
| **PostgreSQL memory** | ✅ | session persistence (optional, degrades gracefully) |
| **Tools/function calling** | ✅ | get_current_time, calculate, search_kb |
| **Pipecat voice agent** | ✅ | LiveKit + STT + KG-RAG + TTS pipeline |
| **Streamlit UI** | ✅ | main branch, full KG-RAG interface |
| **React frontend** | ✅ | chat + live room + upload + vision toggle |

## Quick Start (CPU laptop)

```bash
# 1. Clone and setup
git checkout multimodal
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Start services
tools/livekit/livekit-server.exe --dev          # LiveKit on :7880
python -m uvicorn backend.main:app --port 8000  # Backend on :8000
cd frontend && npm install && npm run dev       # Frontend on :5173

# 3. (Optional) Start the voice agent
python -m realtime.agent --room tutor-room

# 4. Open http://localhost:5173
```

## Quick Start (GPU laptop — avatar phase)

Same as above, plus:
```bash
# Install MuseTalk / avatar pipeline (GPU-only)
pip install musetalk   # or whatever avatar framework

# Start with avatar enabled
python -m realtime.agent --room tutor-room --avatar musetalk
```

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│  React Frontend (:5173)                             │
│  ├─ Chat view (text + mic + TTS + upload)           │
│  └─ Live Room (LiveKit WebRTC mic/camera)           │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP / WebSocket
┌──────────────────────▼──────────────────────────────┐
│  FastAPI Backend (:8000)                            │
│  ├─ /api/chat     — KG-RAG Q&A                     │
│  ├─ /api/stt      — speech → text                   │
│  ├─ /api/tts      — text → speech                   │
│  ├─ /api/upload   — document → FAISS                │
│  ├─ /api/vision   — webcam frame context            │
│  ├─ /api/tools    — function calling                │
│  └─ /api/session/token — LiveKit JWT                │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  KG-RAG Engine (unchanged from main branch)         │
│  Neo4j (KG) + FAISS (vectors) + Groq (LLM)         │
└─────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  Pipecat Voice Agent (realtime/agent.py)            │
│  LiveKit ←→ faster-whisper STT ←→ tutor ←→ TTS     │
└─────────────────────────────────────────────────────┘
```

## Config (.env)

```bash
cp .env.example .env
# Fill in:
GROQ_API_KEY=your_key
NEO4J_PASSWORD=your_password
# LiveKit defaults work for local dev (key=devkey, secret=secret)
```

## File Map

```
AI_TUTOR/
├── src/                        # KG-RAG engine (unchanged)
│   ├── tutor/tutor_engine.py   # ask_tutor() — core
│   ├── rag/                    # hybrid retrieval + FAISS
│   ├── prompts/                # prompt templates + learner levels
│   └── evaluation/             # CoT validation
├── backend/
│   ├── main.py                 # FastAPI app (all endpoints)
│   ├── upload.py               # document upload + chunking
│   ├── memory.py               # PostgreSQL session memory
│   ├── tools.py                # function calling registry
│   └── realtime.py             # LiveKit token generation
├── realtime/
│   ├── agent.py                # Pipecat voice agent ★
│   ├── stt.py                  # faster-whisper STT
│   ├── tts.py                  # Windows System.Speech TTS
│   ├── vision.py               # webcam frame capture
│   └── pipeline.py             # HTTP-based realtime (fallback)
├── frontend/
│   └── src/App.jsx             # React UI (chat + live room)
├── app/streamlit_app.py        # Streamlit UI (main branch)
├── tools/livekit/              # LiveKit server binary
├── requirements.txt
├── docker-compose.yml          # PostgreSQL + LiveKit containers
└── .env.example
```

## What to do on the GPU laptop

1. **Pull this branch** — all code is here
2. **Install deps** — `pip install -r requirements.txt`
3. **Start LiveKit + backend + frontend** — same as CPU
4. **Add avatar** — plug in MuseTalk or LiveTalking
   - Create `realtime/avatar.py` with a `AvatarProcessor(FrameProcessor)`
   - Takes `OutputAudioRawFrame` + camera → generates lip-synced video
   - Inserts into pipeline: `... → tts → avatar → transport.output(video)`
5. **Upgrade TTS** — swap Windows SAPI for Piper/Coqui (neural voice)
6. **Upgrade STT** — swap faster-whisper base for large-v3 (GPU)

## Only remaining

- 🎭 **Avatar (MuseTalk/LiveTalking)** — GPU only, plug in when ready
- 🗣️ **Neural TTS** — Piper or Coqui on GPU (optional, SAPI works fine for demo)
