================================================================================
  PROJECT PLAN — AI TUTOR WITH KG-RAG + VOICE + AVATAR
  (What we are building, why, and which models at each step)
================================================================================

This document explains:
  1. THE PROBLEM we are solving
  2. WHAT we are building (the full system)
  3. WHICH MODELS are used and where
  4. STEP-BY-STEP: what happens when the student asks a question
  5. STEP-BY-STEP: what happens when the student speaks
  6. STEP-BY-STEP: what happens on the GPU laptop (avatar)
  7. THE PROPOSAL details (for your report/presentation)


================================================================================
1. THE PROBLEM
================================================================================

LLMs like GPT-4, DeepSeek, etc. are powerful but BAD at tutoring because:

  HALLUCINATION:
    They make up facts. A student asking "What is recursion?" might get
    wrong information that sounds correct.

  NO REASONING TRANSPARENCY:
    They give black-box answers. Students can't see HOW the answer was
    derived, which defeats the purpose of learning.

  NO PERSONALIZATION:
    Same answer for everyone — beginner and expert get the same response.

  NO VOICE/VISUAL:
    Text-only chatbots are boring. Real tutors talk and have a face.


================================================================================
2. WHAT WE ARE BUILDING
================================================================================

An AI tutoring system with THREE layers:

LAYER 1 — THE BRAIN (KG-RAG):
  - Knowledge Graph (Neo4j) stores relationships between CS concepts
  - Vector Database (FAISS) stores embeddings of 24 CS text files
  - When student asks a question, we search BOTH the graph AND vectors
  - Combined context is sent to the LLM
  - Answer is GROUNDED in real knowledge, not hallucinated

LAYER 2 — THE VOICE (Realtime):
  - Student speaks into microphone
  - Speech-to-Text (faster-whisper) transcribes what they said
  - Brain generates an answer
  - Text-to-Speech (Windows System.Speech) speaks the answer back
  - All in real-time via LiveKit WebRTC

LAYER 3 — THE FACE (Avatar, GPU only):
  - MuseTalk or LiveTalking takes the TTS audio
  - Generates a lip-synced video of the tutor face
  - Student sees a talking tutor, not just text


================================================================================
3. MODELS USED — WHERE AND WHY
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  COMPONENT          MODEL                    WHERE              WHY         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LLM (Brain)        Groq gpt-oss-120b        Backend API        Main tutor │
│                     (OpenAI open model)      /api/chat          reasoning   │
│                                             Groq Cloud         Free tier   │
│                                                                             │
│  Embeddings         all-MiniLM-L6-v2         Local (Python)     Convert    │
│                     (sentence-transformers)  src/rag/           text to    │
│                                             embed_documents.py vectors    │
│                                                                             │
│  Vector Search      FAISS                    Local              Fast        │
│                     (faiss-cpu)              src/rag/           similarity │
│                                             vector_search.py   search     │
│                                                                             │
│  Knowledge Graph    Neo4j 5                  Local/Docker       Store CS   │
│                     (graph database)         src/kg/            concept    │
│                                             kg_query.py        relations  │
│                                                                             │
│  Speech-to-Text     faster-whisper           Local (CPU)        Transcribe │
│                     (base model, int8)       realtime/stt.py    speech     │
│                                                                             │
│  Text-to-Speech     Windows System.Speech    Local (offline)    Speak      │
│                     (PowerShell bridge)      realtime/tts.py    answers    │
│                                                                             │
│  Voice Pipeline     Pipecat 0.0.108          Local              Orchestrate│
│                     (by Daily)               realtime/agent.py  STT→LLM→TTS│
│                                                                             │
│  WebRTC Transport   LiveKit (self-hosted)    Local              Real-time  │
│                     tools/livekit/           :7880              audio/vid  │
│                                                                             │
│  AVATAR (GPU)       MuseTalk or LiveTalking  GPU only           Lip-sync   │
│                     (future)                 realtime/avatar.py face video │
│                                                                             │
│  NEURAL TTS (GPU)   Piper TTS                GPU only (future)  Better     │
│                     (future)                 realtime/tts.py    voice      │
│                                                                             │
│  BETTER STT (GPU)   faster-whisper large-v3  GPU only (future)  More       │
│                     (future)                 realtime/stt.py    accurate   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Model details:

  Groq gpt-oss-120b:
    - OpenAI's open-source model, hosted on Groq cloud
    - Free API key from console.groq.com
    - Fast inference (Groq's custom hardware)
    - Used for: generating tutor answers + Chain-of-Thought reasoning

  all-MiniLM-L6-v2:
    - Sentence transformer model (22M params)
    - Converts text to 384-dimensional vectors
    - Used for: embedding CS text files into FAISS index
    - Runs locally, no API needed

  faster-whisper (base):
    - OpenAI Whisper re-implemented in CTranslate2
    - Much faster than original Whisper, works on CPU
    - Model: "base" (74M params), int8 quantization
    - Used for: transcribing student speech to text

  Neo4j:
    - Graph database storing CS concept relationships
    - 112 triples (e.g., "recursion" --uses--> "base case")
    - Used for: knowledge graph traversal in KG-RAG

  FAISS:
    - Facebook's similarity search library
    - Stores vector embeddings of 24 CS text files
    - Used for: fast nearest-neighbor search


================================================================================
4. STEP-BY-STEP: Student types a question
================================================================================

  STEP 1: Browser sends question
    Student types "What is recursion?" in the chat
    Browser sends POST /api/chat { question: "What is recursion?" }

  STEP 2: Backend receives request
    FastAPI backend at port 8000 receives the request
    backend/main.py:chat() function handles it

  STEP 3: Hybrid retrieval (the KG-RAG magic)
    src/rag/hybrid_retriever.py runs TWO searches:

    SEARCH A — Knowledge Graph (Neo4j):
      - Extract keywords from question: ["recursion"]
      - Query Neo4j: find nodes matching "recursion"
      - Traverse 2 hops: find related concepts (base case, stack, function)
      - Return: graph context (concept relationships)

    SEARCH B — Vector Search (FAISS):
      - Embed question using all-MiniLM-L6-v2
      - Search FAISS for top-5 most similar text chunks
      - Return: document context (relevant paragraphs)

    COMBINE:
      - Merge graph context + document context
      - This is the "KG-RAG" — knowledge-grounded retrieval

  STEP 4: Build prompt
    src/prompts/prompt_builder.py builds the prompt:
      - System instruction: "You are a CS tutor for [beginner] students"
      - Context: the retrieved KG + FAISS content
      - CoT trigger: "Let's break this down step by step"
      - Student question: "What is recursion?"

  STEP 5: LLM generates answer
    Groq API (gpt-oss-120b) processes the prompt
    Returns: reasoning steps + final answer
    Example output:
      Step 1: Recursion is when a function calls itself
      Step 2: It needs a base case to stop
      Step 3: Each call pushes onto the call stack
      Final: Recursion is a technique where a function calls itself...

  STEP 6: Validate CoT against KG
    src/evaluation/evaluation_metrics.py checks:
      - Do the reasoning steps reference KG concepts?
      - Is the reasoning logically consistent?
      - Calculate "grounded fraction" (% of steps backed by KG)

  STEP 7: Return to browser
    Backend returns:
      { answer: "Recursion is...",
        debug: { cot_steps: [...], cot_validation: {...} } }
    Browser displays answer with CoT visualizer


================================================================================
5. STEP-BY-STEP: Student speaks (voice mode)
================================================================================

  STEP 1: Browser captures audio
    Student clicks mic button
    MediaRecorder captures 5 seconds of audio (WebM format)

  STEP 2: Send to STT endpoint
    Browser sends POST /api/stt with audio blob

  STEP 3: faster-whisper transcribes
    realtime/stt.py:
      - Receives audio file
      - Saves to temp file
      - Calls faster-whisper (base model, CPU, int8)
      - Returns: { text: "What is recursion?" }

  STEP 4: Send transcribed text to chat
    Browser puts text in input field
    Student clicks Send (or auto-send)

  STEP 5: Same as text flow (Steps 3-7 above)
    Backend runs KG-RAG → LLM → returns answer

  STEP 6: TTS generates speech
    Browser sends POST /api/tts { text: "Recursion is..." }
    realtime/tts.py:
      - Calls Windows System.Speech via PowerShell
      - Generates WAV file
      - Returns audio bytes

  STEP 7: Browser plays audio
    Student hears the tutor's voice reply


================================================================================
6. STEP-BY-STEP: Live Room (full realtime via Pipecat)
================================================================================

  STEP 1: Browser joins LiveKit room
    Frontend clicks "Join" in Live Room
    Gets JWT token from /api/session/token
    Connects to LiveKit WebRTC on port 7880
    Mic + camera streams to room

  STEP 2: Pipecat agent joins same room
    realtime/agent.py starts
    Connects to LiveKit as "AI-Tutor" bot participant
    Listens for audio from students

  STEP 3: Student speaks in room
    Audio flows through LiveKit to Pipecat agent

  STEP 4: Pipecat pipeline processes
    transport.input() → audio frame
    WhisperSTTService → text (faster-whisper)
    TutorProcessor → answer (KG-RAG: Neo4j + FAISS + Groq)
    WindowsTTSProcessor → audio (System.Speech)
    transport.output() → audio back to room

  STEP 5: Student hears reply
    Audio streams back through LiveKit
    Student hears AI tutor voice in real-time


================================================================================
7. STEP-BY-STEP: GPU Laptop (avatar face)
================================================================================

  This is the ADD-ON for the GPU laptop. Everything above works on CPU.

  STEP 1: MuseTalk receives TTS audio
    The TTS processor generates WAV audio
    Avatar processor receives the audio

  STEP 2: Lip-sync generation
    MuseTalk takes:
      - Source image/video of tutor face
      - TTS audio
    Generates: lip-synced video frames

  STEP 3: Stream video back
    Video frames sent to LiveKit room
    Browser displays the talking face

  Models needed on GPU:
    - MuseTalk: requires CUDA GPU
    - Piper TTS: optional, better voice than SAPI
    - Whisper large-v3: optional, better accuracy


================================================================================
8. THE PROPOSAL (for your report/presentation)
================================================================================

TITLE: "A Knowledge-Grounded and Self-Prompting LLM Framework for
        Personalised AI Tutoring"

WHAT THE PROPOSAL SAYS:

  SECTION 1 — INTRODUCTION:
    LLMs are powerful but unsuitable for tutoring due to hallucination,
    no reasoning transparency, and no personalization. This project
    combines KG-RAG + CoT + Prompt Engineering to fix these problems.

  SECTION 2 — OBJECTIVES:
    1. Design a modular tutoring framework
    2. Mitigate hallucinations via KG-guided retrieval
    3. Enable reasoning with Chain-of-Thought prompting
    4. Enhance output predictability and safety
    5. Conduct end-to-end evaluation

  SECTION 3 — BACKGROUND:
    3.1: Problems with current LLMs (hallucination, bias, no memory)
    3.2: Limitations in tutoring (no personalization, no progress tracking)
    3.3: Opportunity — KG-RAG + CoT + Prompt Engineering

  SECTION 4 — TECHNICAL FRAMEWORK:
    4.1: KG-RAG Architecture
         - Knowledge Graph construction (Neo4j)
         - Query interpretation and node matching
         - Hybrid context retrieval (KG + vectors)
         - Augmented prompt generation

    4.2: Chain-of-Thought Prompting
         - Step-by-step reasoning
         - KG validation at each step
         - Inline display (steps first, conclusion last)
         - We implemented this per §4.2

    4.3: Prompt Engineering
         - Ensemble prompting
         - S2A context filtering
         - Instruction-Output templating

  SECTION 5 — METHODOLOGY:
    5.1: Development pipeline
         - Curriculum data (CS texts)
         - KG construction (112 triples in Neo4j)
         - KG-RAG implementation
         - CoT integration
         - Prompt engineering
         - Web interface

    5.2: Evaluation plan
         - 20 students, Group A vs Group B
         - Metrics: BERTScore, BLEU, ROUGE, quiz improvement
         - Target: 25% improvement in quiz scores


================================================================================
9. SYSTEM ARCHITECTURE (visual)
================================================================================

  STUDENT
     │
     ▼
  ┌──────────────────────────────────────────────────────────┐
  │                    BROWSER (React)                        │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
  │  │ Chat View │  │Live Room │  │  Upload  │  │ Settings│ │
  │  │ (text)   │  │ (WebRTC) │  │ (PDF/TXT)│  │ (CoT,  │ │
  │  │          │  │ mic+cam  │  │          │  │  level) │ │
  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────────┘ │
  └───────┼──────────────┼──────────────┼────────────────────┘
          │              │              │
          ▼              ▼              ▼
  ┌──────────────────────────────────────────────────────────┐
  │                  FASTAPI BACKEND (:8000)                  │
  │                                                          │
  │  /api/chat ──→ KG-RAG Engine                             │
  │  /api/stt  ──→ faster-whisper (STT)                      │
  │  /api/tts  ──→ Windows System.Speech (TTS)               │
  │  /api/upload──→ chunk + embed + add to FAISS             │
  │  /api/vision──→ webcam frame context                      │
  │  /api/tools ──→ function calling                         │
  └──────────────────────┬───────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
  ┌──────────────┐ ┌──────────┐ ┌────────────────────┐
  │   Neo4j      │ │  FAISS   │ │  Groq Cloud        │
  │ (Knowledge   │ │ (Vector  │ │  gpt-oss-120b      │
  │  Graph)      │ │  Search) │ │  (LLM)             │
  │              │ │          │ │                     │
  │ 112 triples  │ │ 24 texts │ │  Free API key      │
  │ CS concepts  │ │ embeddings│ │                    │
  └──────────────┘ └──────────┘ └────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────┐
  │              PIPECAT VOICE AGENT (:7880)                 │
  │                                                          │
  │  LiveKit Transport ←→ STT ←→ KG-RAG ←→ TTS             │
  │                                                          │
  │  User speaks → transcribe → tutor answers → speak back   │
  └──────────────────────────────────────────────────────────┘

  GPU LAPTOP ONLY (below):
  ┌──────────────────────────────────────────────────────────┐
  │  MuseTalk Avatar                                         │
  │  TTS audio + face image → lip-synced video              │
  └──────────────────────────────────────────────────────────┘


================================================================================
10. WHAT EACH FILE DOES
================================================================================

src/                        THE BRAIN
  config.py                 All settings from .env
  tutor/tutor_engine.py     ask_tutor() — main function
  rag/hybrid_retriever.py   Combines Neo4j + FAISS search
  rag/vector_search.py      FAISS similarity search
  rag/embed_documents.py    Build/load FAISS index
  kg/kg_query.py            Neo4j graph queries
  kg/kg_import.py           Import CSV triples into Neo4j
  prompts/prompt_builder.py Build prompts with CoT + context
  evaluation/               CoT validation, BERTScore, BLEU

backend/                    THE SERVER
  main.py                   All API endpoints (FastAPI)
  realtime.py               LiveKit token generation
  upload.py                 Document upload + chunking
  memory.py                 PostgreSQL session memory
  tools.py                  Function calling registry

realtime/                   THE VOICE LAYER
  agent.py                  Pipecat voice agent (main) ***
  stt.py                    faster-whisper STT
  tts.py                    Windows System.Speech TTS
  vision.py                 Webcam frame capture
  pipeline.py               HTTP-based fallback pipeline

frontend/                   THE UI
  src/App.jsx               Chat + Live Room + Upload
  src/styles.css            Styling
  package.json              Node dependencies

data/                       THE KNOWLEDGE BASE
  documents/                24 CS text files
  knowledge_graph/          knowledge_triples.csv (112 triples)


================================================================================
END OF DOCUMENT
