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

## Quick Start (CPU laptop — everything works)

```bash
# 1. Clone and setup
git checkout multimodal
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env         # Edit .env -> set GROQ_API_KEY and NEO4J_PASSWORD

# 2. Start Neo4j (REQUIRED — the Knowledge Graph)
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/your_password neo4j:5
python -c "from src.kg.kg_import import import_csv; import_csv()"  # Import KG

# 3. Start LiveKit
tools/livekit/livekit-server.exe --dev

# 4. Start Backend
python -m uvicorn backend.main:app --port 8000

# 5. Start Frontend
cd frontend && npm install && npm run dev

# 6. Start Voice Agent
python -m realtime.agent --room tutor-room

# 7. Open http://localhost:5173
```

## Quick Start (GPU laptop — adds visual avatar)

Same as above, plus:
```bash
# Install MuseTalk / avatar pipeline (GPU-only, REQUIRES CUDA)
git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk && pip install -r requirements.txt
# Download models from MuseTalk README

# The avatar takes TTS audio + face image -> lip-synced video
# This is the ONLY part that needs a GPU
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
3. **Start Neo4j** — `docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5`
4. **Import KG** — `python -c "from src.kg.kg_import import import_csv; import_csv()"`
5. **Start LiveKit + backend + frontend** — same as CPU
6. **Add avatar** — plug in MuseTalk or LiveTalking (THIS IS THE GPU PART)
   - Create `realtime/avatar.py` with a `AvatarProcessor(FrameProcessor)`
   - Takes `OutputAudioRawFrame` + camera → generates lip-synced video
   - Inserts into pipeline: `... → tts → avatar → transport.output(video)`
7. **Optional upgrades:**
   - TTS: swap Windows SAPI for Piper/Coqui (neural voice)
   - STT: swap faster-whisper base for large-v3 (GPU)

## What needs GPU vs what works on CPU

```
CPU LAPTOP (works now, no GPU needed):
  ✅ KG-RAG tutor (Neo4j + FAISS + Groq)
  ✅ Chain-of-Thought reasoning
  ✅ Text chat in browser
  ✅ Voice: mic → STT → tutor → TTS → hear reply
  ✅ LiveKit room with mic + camera
  ✅ Document upload
  ✅ Vision capture

GPU LAPTOP ONLY:
  ❌ Visual avatar (lip-synced face) — requires MuseTalk + CUDA
  ⬆️ Neural TTS (Piper/Coqui) — optional upgrade
  ⬆️ Large Whisper model — optional upgrade
```

## Only remaining

- 🎭 **Avatar (MuseTalk/LiveTalking)** — GPU only, plug in when ready
- 🗣️ **Neural TTS** — Piper or Coqui on GPU (optional, SAPI works fine for demo)
