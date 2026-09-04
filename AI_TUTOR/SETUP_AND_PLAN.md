================================================================================
  AI TUTOR — FULL PROJECT PLAN + GPU LAPTOP SETUP
  (multimodal branch)
================================================================================

This document explains:
  1. WHAT we are building (the big picture)
  2. WHAT has been built so far (everything works)
  3. HOW to set it up on the GPU laptop (just clone and run)
  4. WHAT to add on the GPU laptop (avatar, neural TTS)
  5. THE PROPOSAL summary (what the project is about)

Read this top to bottom. Follow the steps in order.


================================================================================
PART 1 — WHAT WE ARE BUILDING
================================================================================

We are building a Tavus-like AI tutoring chatbot. The user:
  - Opens a web browser
  - Sees an AI tutor face/avatar on screen
  - Speaks into their microphone
  - The AI listens (STT), thinks (LLM + knowledge graph), and speaks back (TTS)
  - The avatar lip-syncs to the voice

This is a voice-first, visually-present AI tutor. Not just a text chatbot.

The CORE brain of the tutor is a Knowledge Graph + RAG system:
  - Neo4j stores a knowledge graph of CS concepts (recursion, sorting, trees...)
  - FAISS stores vector embeddings of 24 CS text files
  - When the student asks a question, we search BOTH the graph AND the vectors
  - The combined context is sent to Groq LLM (gpt-oss-120b)
  - The answer is grounded in real knowledge, not hallucinated

The REALTIME layer wraps the brain:
  - Student speaks -> faster-whisper transcribes -> brain generates answer -> TTS speaks it
  - All happening in real-time via LiveKit WebRTC

The AVATAR layer (GPU laptop only):
  - MuseTalk or LiveTalking takes the TTS audio
  - Generates lip-synced video of the tutor face
  - Streams back to the browser


================================================================================
PART 2 — WHAT HAS BEEN BUILT (everything below is DONE and WORKING)
================================================================================

MILESTONE 1: LiveKit WebRTC room
  - Browser can join a room with mic + camera
  - Server generates JWT tokens
  - File: backend/realtime.py, frontend LiveSession component

MILESTONE 2: Speech-to-Text (STT)
  - Click mic button -> record 5 seconds -> faster-whisper transcribes
  - Text appears in chat input, user sends it
  - File: realtime/stt.py

MILESTONE 3: KG-RAG Tutor (inherited from main branch)
  - The brain: Neo4j + FAISS + Groq
  - 24 CS text files, 112 knowledge graph triples
  - Chain-of-Thought reasoning (steps first, conclusion last)
  - Adaptive learner levels (Beginner/Intermediate/Advanced)
  - File: src/tutor/tutor_engine.py

MILESTONE 4: Text-to-Speech (TTS)
  - Click speaker button -> Windows System.Speech generates WAV
  - Auto-speak toggle available
  - File: realtime/tts.py

MILESTONE 5: Realtime Pipeline
  - STT -> LLM -> TTS in one loop
  - Turn manager with barge-in support
  - File: realtime/pipeline.py

MILESTONE 6: Turn-Taking
  - Detects when user starts/stops speaking
  - Supports interruption (user speaks while AI is talking)
  - File: realtime/pipeline.py (TurnManager class)

MILESTONE 7: Vision Capture
  - Toggle webcam on -> captures frames every 3 seconds
  - Sends to backend as visual context for tutor
  - File: realtime/vision.py

MILESTONE 8: PostgreSQL Memory (optional)
  - Conversations stored in database
  - Gracefully degrades if PostgreSQL not running
  - File: backend/memory.py

MILESTONE 9: RAG Document Upload
  - Upload PDF/TXT -> chunk -> embed -> tutor learns from it
  - File: backend/upload.py

MILESTONE 10: Tools / Function Calling
  - get_current_time, calculate, search_knowledge_base
  - File: backend/tools.py

MILESTONE 11: Pipecat Voice Agent
  - LiveKit transport + STT + KG-RAG tutor + TTS
  - Joins room as "AI-Tutor" bot participant
  - Full voice loop: speak -> hear reply in real-time
  - File: realtime/agent.py

FRONTEND:
  - React + Vite app
  - Chat view: text chat + mic + TTS + upload + vision toggle
  - Live Room: LiveKit WebRTC mic/camera
  - CoT Visualizer, learner level selector, debug panel
  - File: frontend/src/App.jsx


================================================================================
PART 3 — HOW TO SET UP ON THE GPU LAPTOP
================================================================================

You need:
  - Python 3.10 or 3.11
  - Node.js 18+
  - Git
  - Docker Desktop (for Neo4j and LiveKit)
  - A Groq API key (free from console.groq.com)

STEP 1: Clone the repo
-----------------------
  git clone https://github.com/YOUR_USERNAME/AI_TUTOR.git
  cd AI_TUTOR
  git checkout multimodal

STEP 2: Python environment
---------------------------
  python -m venv .venv

  # Windows:
  .venv\Scripts\activate

  # Linux/Mac:
  source .venv/bin/activate

  pip install -r requirements.txt

  (If pip is slow, install uv first: pip install uv, then: uv pip install -r requirements.txt)

STEP 3: Environment variables
------------------------------
  copy .env.example .env      (Windows)
  cp .env.example .env        (Linux/Mac)

  Edit .env and set:
    GROQ_API_KEY=gsk_your_key_here    <-- REQUIRED, get from console.groq.com
    NEO4J_PASSWORD=your_password       <-- REQUIRED, set a password for Neo4j

  LiveKit defaults: ws://127.0.0.1:7880, key=devkey, secret=secret

STEP 4: Start Neo4j (REQUIRED — the Knowledge Graph)
------------------------------------------------------
  The KG-RAG system REQUIRES Neo4j. Without it, the tutor loses its
  knowledge graph grounding and becomes a regular RAG chatbot.

  Option A — Docker (recommended):
    docker run -d --name neo4j \
      -p 7474:7474 -p 7687:7687 \
      -e NEO4J_AUTH=neo4j/your_password \
      -e NEO4J_PLUGINS='["apoc"]' \
      neo4j:5

  Option B — Neo4j Desktop:
    Download from: https://neo4j.com/download/
    Create a database, set password, start it.

  Verify: open http://localhost:7474 in browser
    Login with: neo4j / your_password

STEP 5: Import Knowledge Graph into Neo4j
------------------------------------------
  After Neo4j is running, import the 112 knowledge triples:

  cd AI_TUTOR
  python -c "from src.kg.kg_import import import_csv; import_csv()"

  This loads all CS concept relationships (recursion, trees, sorting, etc.)
  into Neo4j. Takes ~5 seconds.

  Verify: open http://localhost:7474, run: MATCH (n) RETURN count(n)
  Should show ~150+ nodes.

STEP 6: Start LiveKit server
-----------------------------
  Option A — binary (if tools/livekit/livekit-server.exe exists):
    tools/livekit/livekit-server.exe --dev

  Option B — Docker:
    docker run -d --name livekit -p 7880:7880 livekit/livekit-server:latest --dev --bind 0.0.0.0

  Option C — download from GitHub:
    https://github.com/livekit/livekit/releases
    Extract livekit-server.exe and run: livekit-server.exe --dev

  Verify: open http://127.0.0.1:7880 in browser (should show something or refuse = OK)

STEP 7: Start the backend (Terminal 1)
---------------------------------------
  cd AI_TUTOR
  python -m uvicorn backend.main:app --port 8000

  Wait for: "Uvicorn running on http://0.0.0.0:8000"
  First run takes ~30 seconds (builds FAISS index from documents).

STEP 8: Start the frontend (Terminal 2)
----------------------------------------
  cd AI_TUTOR/frontend
  npm install        (first time only)
  npm run dev

  Opens at: http://localhost:5173

STEP 9: Start the voice agent (Terminal 3)
-------------------------------------------
  cd AI_TUTOR
  python -m realtime.agent --room tutor-room

  Wait for: "[agent] Connecting to ws://127.0.0.1:7880 room=tutor-room"

STEP 10: Test it
----------------
  1. Open http://localhost:5173 in Chrome or Edge
  2. CHAT VIEW: Type "What is recursion?" -> get answer
  3. CHAT VIEW: Click mic button -> speak -> text appears -> send -> hear reply
  4. LIVE ROOM: Click "Join" -> mic/camera on -> speak -> AI tutor replies with voice

  That's it. Everything works.


================================================================================
PART 4 — WHAT WORKS NOW vs WHAT NEEDS THE GPU LAPTOP
================================================================================

WHAT WORKS NOW (CPU laptop, no GPU needed):
  - Full KG-RAG tutor (Neo4j + FAISS + Groq)
  - Chain-of-Thought reasoning with KG validation
  - Text chat in browser
  - Voice: mic -> STT -> tutor -> TTS -> hear reply (Windows SAPI voice)
  - LiveKit room with mic + camera
  - Document upload (PDF/TXT -> tutor learns from it)
  - Vision capture (webcam frames as context)
  - All of this is FULLY WORKING NOW

WHAT DOES NOT WORK WITHOUT GPU:
  - Visual avatar (lip-synced face) — REQUIRES MuseTalk/LiveTalking + CUDA
  - Neural TTS (Piper/Coqui) — optional upgrade, SAPI works fine
  - Large Whisper model — optional upgrade, base model works on CPU

The visual avatar is the ONLY thing that needs the GPU. Everything else
works on a regular CPU laptop. The GPU laptop setup below adds the avatar
and optional upgrades.


================================================================================
PART 5 — GPU LAPTOP ADDITIONS (avatar + upgrades)
================================================================================

A) NEURAL TTS (better voice than Windows SAPI)
----------------------------------------------
  pip install piper-tts

  Download a voice model from: https://github.com/rhasspy/piper/blob/master/VOICES.md
  Good English voices: en_US-lessac-medium, en_US-amy-medium
  Place model files in: models/piper/

  Then in .env, change:
    TTS_PROVIDER=piper
    TTS_VOICE=en_US-lessac-medium

  Update realtime/tts.py to add a PiperProvider class that calls piper-tts.

B) AVATAR (MuseTalk or LiveTalking)
------------------------------------
  This is the big addition. Two options:

  Option 1 — MuseTalk (better quality):
    git clone https://github.com/TMElyralab/MuseTalk.git
    cd MuseTalk
    pip install -r requirements.txt
    # Download models from MuseTalk README

  Option 2 — LiveTalking (simpler):
    git clone https://github.com/OpenTalker/LiveTalking.git
    cd LiveTalking
    pip install -r requirements.txt

  Then create realtime/avatar.py:
    - A Pipecat FrameProcessor that takes audio + video -> generates lip-synced video
    - Inserts into pipeline: ... -> tts -> avatar -> transport.output(video)

  The avatar processor receives:
    - The TTS audio (to drive lip sync)
    - A source image/video of the tutor face
    - Outputs video frames back to LiveKit

C) UPGRADE STT (optional — larger model for better accuracy)
------------------------------------------------------------
  Current: faster-whisper base model (CPU, works great)
  Upgrade: faster-whisper large-v3 (GPU, more accurate)

  In realtime/agent.py, change:
    stt = WhisperSTTService(model="large-v3", device="cuda", compute_type="float16")

D) GPU DEPENDENCIES
-------------------
  Make sure CUDA is installed:
    nvidia-smi    (should show your GPU)

  PyTorch with CUDA:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


================================================================================
PART 6 — THE PROPOSAL (what the project is about)
================================================================================

TITLE: "A Knowledge-Grounded and Self-Prompting LLM Framework for
        Personalised AI Tutoring"

The project builds an AI tutor that solves three problems with LLMs in education:

PROBLEM 1: HALLUCINATION
  LLMs make up facts. A student learning recursion might get wrong information.
  SOLUTION: Knowledge Graph + RAG. We ground every answer in verified knowledge
  stored in Neo4j and FAISS. The LLM can only use retrieved facts.

PROBLEM 2: NO REASONING TRANSPARENCY
  LLMs give black-box answers. Students can't see HOW the answer was derived.
  SOLUTION: Chain-of-Thought prompting. The tutor shows its reasoning steps
  before giving the final answer. Each step is validated against the KG.

PROBLEM 3: NO PERSONALIZATION
  LLMs give the same answer to everyone regardless of skill level.
  SOLUTION: Adaptive learner levels (Beginner/Intermediate/Advanced) that
  change the depth and complexity of explanations.

KEY TECHNIQUES (from the proposal):

1. KG-RAG (Section 4.1):
   - Knowledge Graph stores entity-relationship triples
   - Hybrid retrieval: graph traversal + vector similarity
   - Reduces hallucinations, ensures factual grounding

2. Chain-of-Thought (Section 4.2):
   - Step-by-step reasoning before final answer
   - Each step validated against knowledge graph
   - Inline display (steps first, conclusion last)

3. Prompt Engineering (Section 4.3):
   - Ensemble prompting (multiple prompts, aggregated)
   - S2A context filtering (remove noise from student questions)
   - Instruction-Output templating

4. Evaluation (Section 5.2):
   - 20 students, Group A (baseline RAG) vs Group B (full KG-RAG + CoT)
   - Metrics: BERTScore, BLEU, quiz improvement, trust survey
   - Target: 25% improvement in quiz scores

WHAT WE BUILT vs WHAT THE PROPOSAL ASKED FOR:

  Proposal Section              | Our Implementation          | Status
  ------------------------------|-----------------------------|--------
  KG-RAG (4.1)                 | Neo4j + FAISS + Groq        | DONE
  CoT Prompting (4.2)          | Steps-first inline display  | DONE
  Ensemble Prompting (4.3)     | Ensemble mode in engine     | DONE
  S2A Context Filtering (4.3)  | Input preprocessing         | DONE
  Adaptive Learner Levels      | 3 levels in prompt builder  | DONE
  CoT Visualizer               | Per-step KG validation      | DONE
  Streamlit UI                 | Full interface              | DONE
  Evaluation Metrics           | BERTScore, BLEU, ROUGE     | DONE
  Content Safety Guard         | Regex-based filter          | DONE
  Multimodal (voice + vision)  | STT + TTS + LiveKit + Vision| DONE
  Avatar (MuseTalk)            | GPU only, not on CPU laptop  | TODO (GPU)
  PostgreSQL Memory            | Session persistence         | DONE (optional)


================================================================================
PART 7 — HOW EVERYTHING CONNECTS (data flow)
================================================================================

STUDENT ASKS A QUESTION (text):

  Browser -> POST /api/chat { question: "What is recursion?" }
    -> backend/main.py
    -> src/rag/hybrid_retriever.py
       -> Neo4j: graph traversal, find related concepts (base case, stack, etc.)
       -> FAISS: vector search, find relevant text chunks
       -> Combine both contexts
    -> src/prompts/prompt_builder.py
       -> Build prompt with context + CoT trigger + learner level
    -> Groq API (gpt-oss-120b)
       -> Generate answer with reasoning steps
    -> src/evaluation/evaluation_metrics.py
       -> Validate CoT steps against KG
    -> Return { answer: "...", debug: { cot_steps: [...], cot_validation: {...} } }
  <- Browser displays answer with CoT visualizer

STUDENT SPEAKS (voice):

  Browser mic -> MediaRecorder captures 5s audio
    -> POST /api/stt (audio blob)
    -> realtime/stt.py: faster-whisper transcribes
    <- Return { text: "What is recursion?" }
  Browser puts text in input -> student clicks Send
    -> (same as text flow above)
    -> POST /api/tts { text: "Recursion is when..." }
    -> realtime/tts.py: Windows System.Speech generates WAV
    <- Return WAV audio
  Browser plays audio

LIVE ROOM (full realtime via Pipecat):

  Browser joins LiveKit room (mic + camera)
    -> Pipecat agent also joins room as "AI-Tutor" bot
    -> User speaks -> LiveKit streams audio to agent
    -> realtime/agent.py pipeline:
       transport.input() -> faster-whisper STT -> TutorProcessor (KG-RAG)
       -> WindowsTTSProcessor -> transport.output()
    -> Audio streams back to browser via LiveKit
    <- User hears AI tutor voice in real-time


================================================================================
PART 8 — FILE MAP (what everything is)
================================================================================

AI_TUTOR/
|
|-- src/                          THE BRAIN (KG-RAG engine)
|   |-- config.py                 All settings from .env
|   |-- tutor/tutor_engine.py     ask_tutor() — the main function
|   |-- rag/
|   |   |-- hybrid_retriever.py   Combines Neo4j + FAISS search
|   |   |-- vector_search.py      FAISS similarity search
|   |   |-- embed_documents.py    Build/load FAISS index
|   |-- kg/
|   |   |-- kg_query.py           Neo4j graph queries
|   |   |-- kg_import.py          Import CSV triples into Neo4j
|   |-- prompts/
|   |   |-- prompt_builder.py     Build prompts with CoT + context
|   |-- evaluation/
|       |-- evaluation_metrics.py CoT validation, BERTScore, BLEU
|
|-- backend/                      THE SERVER (FastAPI)
|   |-- main.py                   All API endpoints
|   |-- realtime.py               LiveKit token generation
|   |-- upload.py                 Document upload + chunking
|   |-- memory.py                 PostgreSQL session memory
|   |-- tools.py                  Function calling registry
|
|-- realtime/                     THE VOICE LAYER (Pipecat agent)
|   |-- agent.py                  Pipecat voice agent (main) ***
|   |-- stt.py                    faster-whisper STT
|   |-- tts.py                    Windows System.Speech TTS
|   |-- vision.py                 Webcam frame capture
|   |-- pipeline.py               HTTP-based fallback pipeline
|
|-- frontend/                     THE UI (React)
|   |-- src/App.jsx               Chat + Live Room + Upload
|   |-- src/styles.css            Styling
|   |-- package.json              Node dependencies
|
|-- data/                         THE KNOWLEDGE BASE
|   |-- documents/                24 CS text files (recursion, trees, etc.)
|   |-- knowledge_graph/          knowledge_triples.csv (112 triples)
|
|-- tools/livekit/                LiveKit server binary (gitignored)
|-- .env.example                  Environment template
|-- requirements.txt              Python dependencies
|-- docker-compose.yml            PostgreSQL + LiveKit containers
|-- README.md                     Project overview
|-- REQUIREMENTS.md               Setup guide (step by step)
|-- SETUP_AND_PLAN.md             THIS FILE


================================================================================
PART 9 — QUICK REFERENCE (all commands)
================================================================================

# ONE-TIME SETUP:
  git clone <repo-url>
  cd AI_TUTOR
  git checkout multimodal
  python -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt
  copy .env.example .env
  # Edit .env -> set GROQ_API_KEY and NEO4J_PASSWORD

# START NEO4J (required):
  docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5
  # Then import knowledge graph:
  python -c "from src.kg.kg_import import import_csv; import_csv()"

# START SERVICES (3 terminals):
  # Terminal 1 - Backend:
  python -m uvicorn backend.main:app --port 8000

  # Terminal 2 - Frontend:
  cd frontend && npm install && npm run dev

  # Terminal 3 - Voice Agent:
  python -m realtime.agent --room tutor-room

  # LiveKit must be running separately:
  tools/livekit/livekit-server.exe --dev

# OPEN BROWSER:
  http://localhost:5173

# TEST:
  Type a question -> see answer with CoT
  Click mic -> speak -> hear reply
  Live Room -> join -> speak -> AI responds


================================================================================
END OF DOCUMENT
================================================================================
