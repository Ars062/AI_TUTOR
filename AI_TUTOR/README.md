# AI Tutor — Multimodal Voice Tutor (branch: `musetalk`)

Knowledge-grounded AI tutoring with a real-time voice conversation and a
**neural lip-synced teacher avatar** (MuseTalk) inside a Google-Meet-style
live room.

> **Branch status:** `musetalk` — live room + Piper TTS + MuseTalk GPU
> lip-sync + recording are built and verified end-to-end.
> Base branch `multimodal` has the same tutor without the neural avatar.

---

## What's built (working)

| Feature | Status | How it works |
|---------|--------|-------------|
| **KG-RAG tutor** | ✅ | Neo4j + FAISS + Groq (`openai/gpt-oss-120b`) |
| **Voice Q&A** | ✅ | mic → LiveKit → VAD → Whisper STT → tutor → TTS → avatar |
| **STT** | ✅ | faster-whisper `base`, int8, CPU (`Systran/faster-whisper-base`) |
| **TTS** | ✅ | Piper neural voice, offline (`TTS_PROVIDER=piper`) |
| **MuseTalk avatar** | ✅ | GPU lip-sync on a static teacher portrait, streamed at 25 fps |
| **Puppet avatar (fallback)** | ✅ | 2D amplitude mouth on `teacher.png` — no GPU needed |
| **Meet-style live room** | ✅ | big teacher video + small self PiP + control bar |
| **Screen recording** | ✅ | in-app Record button → `lesson_*.webm` download |
| **Text chat** | ✅ | React chat → `POST /api/chat` → same tutor engine |
| **RAG upload** | ✅ | PDF/TXT → chunk → embed → tutor learns |
| **Chain-of-Thought** | ✅ | steps-first, inline, §4.2 compliant |
| **LiveKit WebRTC** | ✅ | mic/camera room `tutor-room`, server-issued tokens |
| **Streamlit UI** | ✅ | full KG-RAG interface on `:8501` |
| **PostgreSQL memory** | ✅ | optional — degrades gracefully if absent |

---

## Architecture

```text
┌───────────────────────────────────────────────────────────┐
│  React frontend (:5173)                                   │
│  ├─ Chat view   — text + mic + TTS + upload               │
│  └─ Live Room   — Meet layout (teacher big / self PiP,    │
│                   Record button, Mic/Camera/Leave)        │
└───────────────┬───────────────────────────────────────────┘
                │ HTTP + LiveKit WebRTC (:7880)
┌───────────────▼───────────────────────────────────────────┐
│  FastAPI backend (:8000)                                  │
│  /api/chat /api/stt /api/tts /api/upload /api/vision      │
│  /api/tools /api/session/token (LiveKit JWT)              │
└───────────────┬───────────────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────────────┐
│  KG-RAG engine: Neo4j (KG) + FAISS (vectors) + Groq LLM   │
└───────────────┬───────────────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────────────┐
│  Pipecat voice agent (realtime/agent.py) — room tutor-room│
│                                                           │
│  mic → LiveKitIn → VADProcessor(Silero) → WhisperSTT      │
│      → TutorProcessor (RAG+KG → Groq answer)              │
│      → TTSProcessor (Piper, offline)                      │
│      → AvatarProcessor (MuseTalk GPU lip-sync, 1024×768@25)│
│      → LiveKitOut → browser (video + audio)               │
│                                                           │
│  Greeting: first participant joins → +1s →               │
│            "Hello! I'm your AI tutor. ..."               │
└───────────────────────────────────────────────────────────┘
```

**Text path:** browser chat → `:8000/api/chat` → `ask_tutor()` → same
KG-RAG engine (independent of the voice pipeline).

**Voice path details:**

1. Browser publishes mic audio to LiveKit room `tutor-room`.
2. `VADProcessor` (Silero, `confidence=0.6`, `min_volume=0.1`) gates
   speech vs. silence and prints `[vad] vol=… conf=… state=…` for debugging.
3. `WhisperSTTService` transcribes speech segments
   (`audio_passthrough=False` — see Troubleshooting #4).
4. `TutorProcessor` retrieves from FAISS (528 docs) + KG, answers via
   Groq (`ask_tutor`), pushes `TTSSpeakFrame(answer)`.
5. `TTSProcessor` synthesizes PCM with Piper.
6. `AvatarProcessor` renders lip-synced video frames (MuseTalk) paced
   against the TTS audio and forwards both to LiveKit output.

---

## Quick start (Linux)

```bash
git clone https://github.com/Ars062/AI_TUTOR.git
git checkout musetalk
cd AI_TUTOR
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install piper-tts diffusers opencv-python-headless einops omegaconf   # realtime extras
cp .env.example .env            # set GROQ_API_KEY, NEO4J_PASSWORD, AVATAR_* (see Config)

# 1. Neo4j (Knowledge Graph)
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password neo4j:5
python -c "from src.kg.kg_loader import load_kg; load_kg()"

# 2. LiveKit  ← the two env vars are REQUIRED (see Troubleshooting #1)
cd AI_TUTOR
LIVEKIT_ROOM_EMPTY_TIMEOUT=999999999 \
LIVEKIT_ROOM_DEPARTURE_TIMEOUT=999999999 \
./tools/livekit/livekit-server --dev --bind 0.0.0.0

# 3. Backend
python -m uvicorn backend.main:app --port 8000

# 4. Frontend
cd frontend && npm install && npm run dev          # :5173

# 5. Streamlit UI (headless flag is REQUIRED — see Troubleshooting #5)
streamlit run app/streamlit_app.py --server.headless true --server.fileWatcherType none

# 6. STT model (once) + voice agent
python realtime/ensure_stt_model.py                # downloads faster-whisper-base
python -m realtime.agent --room tutor-room

# 7. Open the app
xdg-open http://localhost:5173
```

Windows equivalents: `livekit-server.exe --dev`, `.venv\Scripts\activate`,
`copy .env.example .env`. The agent/TTS/avatar code is cross-platform.

### Ports

| Port | Service |
|------|---------|
| 5173 | React frontend (Vite) |
| 8000 | FastAPI backend |
| 7880 | LiveKit server |
| 8501 | Streamlit UI |
| 7474 | Neo4j browser |

---

## MuseTalk (neural lip-sync)

The avatar takes the TTS audio + one portrait image and generates a
lip-synced talking head on the GPU. This is the only GPU-dependent part.

### Weights (≈3.7 GB) — place under `models/musetalk/`

| File | Exact size (bytes) |
|------|--------------------|
| `musetalkV15/unet.pth` | 3,400,074,924 |
| `musetalkV15/musetalk.json` | 748 |
| `sd-vae-ft-mse/config.json` | 547 |
| `sd-vae-ft-mse/diffusion_pytorch_model.bin` | 334,707,217 |
| `whisper/config.json` | 1,983 |
| `whisper/preprocessor_config.json` | 184,990 |
| `whisper/pytorch_model.bin` | 151,095,027 |

Get them from the [MuseTalk repo](https://github.com/TMElyralab/MuseTalk)
README (Hugging Face release). `models/` is git-ignored.

### Code

- `realtime/musetalk_core.py` — vendored inference core (no mmpose /
  librosa / mmlab needed): VAE + UNet + Whisper audio features, portrait
  latents, feathered face blending.
- `realtime/avatar.py` — `LipsyncEngine` interface with two engines:
  - **`MuseTalkEngine`** — GPU (fp16, CUDA), lazy warmup, streaming
    `render_stream()` (batches of 8, ~15 fps generation, paced to 25 fps).
  - **`PuppetEngine`** — CPU fallback, RMS-driven mouth overlay.
  - `build_engine("auto")` prefers MuseTalk when all 6 weight files exist,
    otherwise falls back to puppet.
- Portrait: `avatar/teacher.png` (1024×768) generated by
  `avatar/make_teacher_avatar.py`. Face bounding box env `MUSETALK_BBOX`
  (default `317,80,707,585`).

### Config

```bash
AVATAR_PROVIDER=musetalk     # provider flag
AVATAR_ENGINE=musetalk       # musetalk | puppet | auto
AVATAR_WIDTH=1024  AVATAR_HEIGHT=768  AVATAR_FPS=25
```

If MuseTalk init fails (no GPU / missing weights) the agent logs a warning
and keeps running with the puppet engine.

---

## Frontend guide

1. `http://localhost:5173` → **Live Room** tab → **Join (mic + camera)**.
2. Teacher appears big-screen (MuseTalk video), you in a small PiP
   bottom-right, Mic / Camera / Leave bar at the bottom.
3. Speak after the greeting → 2–4 s later the teacher answers with
   lip-synced speech.
4. **Record** (top-right) → pick *Entire Screen* → Share → talks are
   captured → **Stop** → downloads `lesson_<timestamp>.webm`.
   - Do not refresh the page while recording.
   - Button is grey until the teacher (agent) is connected.

---

## Config (`.env`)

Required:

```bash
GROQ_API_KEY=...          # Groq console key (LLM)
NEO4J_PASSWORD=...        # matches docker -e NEO4J_AUTH
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey    # livekit-server --dev defaults
LIVEKIT_API_SECRET=secret
AVATAR_PROVIDER=musetalk
AVATAR_ENGINE=musetalk
```

Useful optionals: `LLM_MODEL` (default `openai/gpt-oss-120b`),
`STT_MODEL` (default `Systran/faster-whisper-base`), `TTS_PROVIDER=piper`,
`TTS_VOICE`, `MUSETALK_BBOX`, `DATABASE_URL` (PostgreSQL, optional),
`DOCUMENTS_DIR` / `FAISS_INDEX_PATH` / `KG_CSV_PATH`.
See `.env.example` for the full annotated list.

---

## File map

```
AI_TUTOR/
├── src/                        # KG-RAG engine
│   ├── tutor/tutor_engine.py   #   ask_tutor() — core
│   ├── rag/                    #   hybrid retrieval + FAISS
│   ├── kg/                     #   Neo4j knowledge graph
│   ├── prompts/                #   prompt templates + learner levels
│   └── evaluation/             #   CoT validation
├── backend/main.py             # FastAPI (chat/stt/tts/upload/vision/tools/token)
├── realtime/
│   ├── agent.py                # Pipecat voice agent ★ (pipeline + greeting)
│   ├── avatar.py               # LipsyncEngine, PuppetEngine, MuseTalkEngine ★
│   ├── musetalk_core.py        # MuseTalk inference (VAE/UNet/Whisper) ★
│   ├── tts.py                  # Piper TTS (offline)
│   ├── stt.py / ensure_stt_model.py  # faster-whisper helpers
│   ├── vision.py               # webcam frame capture
│   └── pipeline.py             # HTTP-based realtime (fallback)
├── avatar/
│   ├── make_teacher_avatar.py  # generates teacher.png (portrait + face boxes)
│   └── teacher.png             # avatar portrait (1024×768)
├── frontend/src/App.jsx        # React UI: chat + MeetLayout + Record ★
├── frontend/src/styles.css     # Meet layout / PiP / Record styles
├── models/musetalk/            # weights (git-ignored, ~3.7 GB)
├── tools/livekit/              # livekit-server binary
├── app/streamlit_app.py        # Streamlit UI
├── PROJECT_PLAN.md / SETUP_AND_PLAN.md / START_HERE.md / STATUS.md
└── requirements.txt
```

---

## Verification & debugging

Healthy agent log markers (in order):

```text
Connected to tutor-room as AI-Tutor
[avatar] musetalk ready (weights=... bbox=[317, 80, 707, 585])
Voice tutor RAG index loaded: 528 docs
[agent] Participant joined: student-xxxx
[agent] Greeting queued
[tts] pushing audio frame ...B sr=22050
[avatar] emit: audio=N samples, video_frames=n engine=musetalk
[vad] vol=0.000 conf=0.00 state=VADState.QUIET     ← while you speak:
[vad] vol=0.519 conf=0.87 state=VADState.SPEAKING     vol/conf must rise
[tutor] Q: <your question> -> A: <n> chars
```

`fatal`/`Exception` lines should not appear. One-off `FfiHandle.__del__`
assert noise at shutdown is harmless.

---

## Troubleshooting (real issues we hit — don't rediscover them)

1. **Room closes after exactly 5 minutes / agent becomes a zombie.**
   LiveKit defaults `emptyTimeout=300` and does not count the agent as an
   occupant. Start the server with **`LIVEKIT_ROOM_EMPTY_TIMEOUT=999999999`**
   *and* `LIVEKIT_ROOM_DEPARTURE_TIMEOUT=999999999` (note `ROOM_` in the
   first one — `LIVEKIT_EMPTY_TIMEOUT` is silently ignored).
   A "zombie" agent (alive but disconnected, keep-alive won't restart it):
   `pkill -f "realtime.ag[e]nt"` — the bracket avoids killing your own
   shell (pkill matches its own command line otherwise).

2. **Agent crashes at startup: `free(): invalid pointer` / triton.**
   `realtime/agent.py` imports `triton` on line 1 on purpose — loading it
   before torch/transformers avoids a glibc abort inside
   `torch._dynamo → import triton` when the index preloads in a thread.
   Do not remove that import.

3. **STT never transcribes although audio arrives.**
   - `python realtime/ensure_stt_model.py` then restart the agent.
   - A stale `blobs/*.incomplete` marker in the HF cache makes readiness
     checks fail even when `model.bin` is complete — delete the marker.
   - Watch `[vad]` lines: you need `conf ≥ 0.6` and `vol ≥ 0.1` for speech.
     Thresholds live in `VADParams(confidence=0.6, min_volume=0.1)`
     in `realtime/agent.py` (pipecat's stock `min_volume=0.6` is too high
     for normal mic speech).

4. **Black screen / avatar restarts ~14×/sec when someone joins.**
   `WhisperSTTService(audio_passthrough=False)` is required. With
   passthrough (pipecat's default) your mic frames travel all the way to
   the avatar, which cancels its render task for every frame → nothing can
   render. Keep it False.

5. **Streamlit dies right after the welcome banner.**
   Always pass `--server.headless true` (plus `--server.fileWatcherType
   none` to avoid watcher noise).

6. **Vite proxy / backend must run for chat.** Frontend proxies `/api` to
   `127.0.0.1:8000`; without the backend the text chat and token endpoint
   fail (voice join also needs `/api/session/token`).

---

## Known issues

- **Recording has no audio.** The Record button captures screen *video*
  correctly (`vp8` webm), but the mixed teacher/mic audio track comes out
  silent — the AudioContext source wiring needs a fix. Video-only until
  then.
- MuseTalk lower face can look slightly hazy (neural inpainting) — normal
  for this model.
- First video frame after join starts at low resolution and ramps to
  1024×768 within ~10 s (LiveKit bandwidth estimation).

---

## Git

```bash
git checkout musetalk
git push git@github.com:Ars062/AI_TUTOR.git musetalk   # SSH (HTTPS has no creds)
```

Key commits: `cfc1fe1` Phase 2 Meet room + Piper + pipeline fixes ·
`072e91e` MuseTalk engine + Record button · `7cb95d5` voice-pipeline fixes
(passthrough/VAD). The `multimodal` branch is unchanged at `cfc1fe1`.
