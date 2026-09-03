# REQUIREMENTS — Step-by-Step Setup Guide

Everything needed to run the AI Tutor on a new machine (CPU or GPU laptop).

---

## 1. Prerequisites

### Python 3.10+
```bash
# Check version
python --version   # must be 3.10.x or 3.11.x

# If not installed, download from:
# https://www.python.org/downloads/
# On install, CHECK "Add Python to PATH"
```

### Node.js 18+ (for React frontend)
```bash
# Check version
node --version     # must be 18+
npm --version      # must be 9+

# If not installed, download from:
# https://nodejs.org/
```

### Git
```bash
git --version
# If not installed: https://git-scm.com/download/win
```

### Docker Desktop (optional — for PostgreSQL and LiveKit containers)
```bash
# Only needed if you want database persistence
# Download from: https://www.docker.com/products/docker-desktop/
```

---

## 2. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI_TUTOR.git
cd AI_TUTOR
git checkout multimodal
```

---

## 3. Python Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

If `pip install` is slow, use `uv` instead:
```bash
pip install uv
uv pip install -r requirements.txt
```

---

## 4. Environment Variables

```bash
# Copy the example env file
copy .env.example .env      # Windows
cp .env.example .env        # Linux/Mac
```

Edit `.env` and fill in:

```bash
# REQUIRED — get from https://console.groq.com/keys
GROQ_API_KEY=gsk_your_key_here

# Neo4j (optional — system works without it, just no knowledge graph)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LiveKit (defaults work for local dev)
LIVEKIT_URL=ws://127.0.0.1:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret

# TTS provider (sapi = Windows offline voice, works on any Windows machine)
TTS_PROVIDER=sapi
```

---

## 5. Neo4j (Optional but Recommended)

The tutor works WITHOUT Neo4j (FAISS vector search only), but KG grounding
gives better answers. Two options:

### Option A: Docker (recommended)
```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5
```

Then open http://localhost:7474 in browser, login, and import your KG:
```bash
# Import knowledge triples (run once)
python -c "from src.kg.kg_import import import_csv; import_csv()"
```

### Option B: Skip it
Just leave `NEO4J_PASSWORD=` empty in `.env`. The tutor will use FAISS only.

---

## 6. LiveKit Server (Required for Voice)

### Option A: Download binary (Windows)
```bash
# Download from: https://github.com/livekit/livekit/releases
# Get livekit-server-windows-amd64.zip
# Extract to tools/livekit/livekit-server.exe

# Or use the provided binary if it exists:
tools/livekit/livekit-server.exe --dev
```

### Option B: Docker
```bash
docker run -d --name livekit \
  -p 7880:7880 -p 7881:7881 \
  -p 50000-60000:50000-60000/udp \
  livekit/livekit-server:latest --dev --bind 0.0.0.0
```

### Option C: Install via Go
```bash
go install github.com/livekit/livekit-server@latest
livekit-server --dev
```

Verify it's running:
```bash
# Should return "livekit" or similar
curl http://127.0.0.1:7880
```

---

## 7. Start the Services

Open **3 separate terminals** (all with `.venv` activated):

### Terminal 1 — Backend API
```bash
cd AI_TUTOR
python -m uvicorn backend.main:app --port 8000
```
Wait for: `[memory] PostgreSQL unavailable ...` or `Uvicorn running on http://0.0.0.0:8000`

### Terminal 2 — Frontend
```bash
cd AI_TUTOR/frontend
npm install      # first time only
npm run dev
```
Opens at: http://localhost:5173

### Terminal 3 — Voice Agent (Pipecat)
```bash
cd AI_TUTOR
python -m realtime.agent --room tutor-room
```
Wait for: `[agent] Connecting to ws://127.0.0.1:7880 room=tutor-room`

---

## 8. Test It

1. Open http://localhost:5173 in Chrome/Edge
2. **Chat view**: Type a question → get answer (KG-RAG tutor)
3. **Chat view**: Click 🎤 → speak → text appears → send → hear reply
4. **Live Room**: Click "Join" → mic/camera activates → speak → AI tutor responds via voice

---

## 9. GPU Laptop — Additional Steps

When you move to the GPU machine, add these on top of steps 1-8:

### 9a. Neural TTS (Piper — much better voice)
```bash
pip install piper-tts

# Download a voice model (e.g., English)
# https://github.com/rhasspy/piper/blob/master/VOICES.md
# Place in: models/piper/en_US-lessac-medium.onnx
```

Then update `.env`:
```bash
TTS_PROVIDER=piper
TTS_VOICE=en_US-lessac-medium
```

### 9b. Avatar (MuseTalk or LiveTalking)
```bash
# MuseTalk (lip-sync from audio)
git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk
pip install -r requirements.txt

# OR LiveTalking (simpler)
git clone https://github.com/OpenTalker/LiveTalking.git
cd LiveTalking
pip install -r requirements.txt
```

Then create `realtime/avatar.py`:
```python
"""Avatar processor — plugs into Pipecat pipeline."""
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

class AvatarProcessor(FrameProcessor):
    """Takes OutputAudioRawFrame → generates lip-synced video."""
    async def process_frame(self, frame, direction):
        # TODO: integrate MuseTalk/LiveTalking
        await self.push_frame(frame, direction)
```

Update `realtime/agent.py` to include avatar in pipeline.

### 9c. Upgrade STT to GPU Whisper
```bash
# The current faster-whisper base model works on CPU.
# For better accuracy on GPU:
pip install faster-gpu-whisper  # if available, or just use faster-whisper with cuda
```

Update `realtime/agent.py`:
```python
stt = WhisperSTTService(model="large-v3", device="cuda", compute_type="float16")
```

### 9d. GPU Dependencies
```bash
# Make sure CUDA is installed
nvidia-smi   # should show your GPU

# PyTorch with CUDA (if not already)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 10. Quick Reference — All Commands

```bash
# === Setup (one time) ===
git clone https://github.com/YOUR_USERNAME/AI_TUTOR.git
cd AI_TUTOR
git checkout multimodal
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env → set GROQ_API_KEY

# === Start LiveKit ===
tools/livekit/livekit-server.exe --dev
# OR: docker run -d --name livekit -p 7880:7880 livekit/livekit-server:latest --dev --bind 0.0.0.0

# === Start Backend (Terminal 1) ===
python -m uvicorn backend.main:app --port 8000

# === Start Frontend (Terminal 2) ===
cd frontend && npm install && npm run dev

# === Start Voice Agent (Terminal 3) ===
python -m realtime.agent --room tutor-room

# === Open browser ===
# http://localhost:5173
```

---

## 11. Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'backend'` | Run from `AI_TUTOR/` directory, not a subdirectory |
| `Connection refused` on port 7880 | LiveKit not running — start it first |
| `Connection refused` on port 8000 | Backend not running — start it first |
| `GROQ_API_KEY not set` | Edit `.env` file, add your Groq API key |
| `neo4j.AuthError` | Wrong Neo4j password in `.env`, or Neo4j not running |
| NLTK import error on Python 3.10 | Run from a different directory, or set `PYTHONSAFEPATH=1` |
| `ffmpeg not found` | Install ffmpeg: `pip install ffmpeg` or download from ffmpeg.org |
| TTS silent / no audio | Check Windows audio is not muted; try `TTS_PROVIDER=sapi` |
| Frontend shows "disconnected" | Backend must be running on :8000 before opening frontend |

---

## 12. File Map

```
AI_TUTOR/
├── src/                        # KG-RAG engine (core, unchanged)
│   ├── config.py               # all settings from .env
│   ├── tutor/tutor_engine.py   # ask_tutor() — the brain
│   ├── rag/                    # hybrid retrieval (FAISS + Neo4j)
│   ├── kg/                     # knowledge graph queries
│   ├── prompts/                # prompt templates + learner levels
│   └── evaluation/             # CoT validation
│
├── backend/                    # FastAPI server
│   ├── main.py                 # all API endpoints
│   ├── upload.py               # document upload + chunking
│   ├── memory.py               # PostgreSQL session memory
│   ├── tools.py                # function calling
│   └── realtime.py             # LiveKit token generation
│
├── realtime/                   # Voice agent
│   ├── agent.py                # Pipecat pipeline (main) ★
│   ├── stt.py                  # faster-whisper STT
│   ├── tts.py                  # Windows System.Speech TTS
│   ├── vision.py               # webcam frame capture
│   └── pipeline.py             # HTTP-based fallback
│
├── frontend/                   # React UI
│   ├── src/App.jsx             # chat + live room + upload
│   └── src/styles.css          # styling
│
├── data/                       # Knowledge base
│   ├── documents/              # CS text files (24 files)
│   └── knowledge_graph/        # KG triples CSV
│
├── tools/livekit/              # LiveKit server binary (gitignored)
├── .env.example                # environment template
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # PostgreSQL + LiveKit containers
└── README.md                   # project overview
```

---

## What Depends on What

```
GROQ_API_KEY ──────────→ backend (LLM calls)
Neo4j ─────────────────→ backend (knowledge graph queries)
                           └── optional, degrades to FAISS-only
LiveKit server ────────→ voice agent + frontend (WebRTC)
                           └── required for voice mode
Python venv ───────────→ backend + voice agent
Node.js ───────────────→ frontend (React build)
Windows audio ─────────→ TTS (System.Speech)
Webcam + mic ──────────→ STT + vision (browser)
```
