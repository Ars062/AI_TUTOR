# START HERE — read this first

**Repo:** https://github.com/Ars062/AI_TUTOR
**Branch:** `multimodal`

---

## Which document do I read?

| File | Read it when you want to know... |
|------|-------------------------------|
| **START_HERE.md** (this file) | The 5 commands to get it running |
| `PROJECT_PLAN.md` | What we're building, which models, why |
| `REQUIREMENTS.md` | Every prerequisite + troubleshooting |
| `SETUP_AND_PLAN.md` | Detailed walkthrough of the whole plan |
| `README.md` | Architecture overview |

---

## ⚠️ 3 things that WILL trip you up

**1. `tools/` is gitignored — the LiveKit binary is NOT in the repo.**
You must download it yourself (see Step 2 below).

**2. Neo4j must be running BEFORE you load the knowledge graph.**
Otherwise the tutor silently falls back to plain RAG and you lose KG grounding.

**3. PostgreSQL is optional.** It only stores chat history. Skip it if you like —
the system prints `[memory] PostgreSQL unavailable — in-memory only` and works fine.

---

## The 5 commands

### Step 0 — Prerequisites
- Python 3.10 or 3.11
- Node.js 18+
- Git
- Docker Desktop (for Neo4j + LiveKit)
- A free Groq API key → https://console.groq.com/keys

### Step 1 — Clone + Python deps
```bash
git clone https://github.com/Ars062/AI_TUTOR.git
cd AI_TUTOR
git checkout multimodal

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
> If pip is slow: `pip install uv` then `uv pip install -r requirements.txt`

> **Want the evaluation metrics too** (BLEU / ROUGE / BERTScore, study harness)?
> Those need extra heavy packages. Install them separately so the base setup
> stays fast: `pip install -r requirements-eval.txt`
> Everything else — chat, KG-RAG, voice, Live Room — works without it.

### Step 2 — Get LiveKit (NOT in the repo)
```bash
mkdir -p tools/livekit
# Download livekit-server-windows-amd64.zip from:
#   https://github.com/livekit/livekit/releases/download/v1.13.5/livekit-server-windows-amd64.zip
# Extract livekit-server.exe into tools/livekit/

# Verify:
tools/livekit/livekit-server.exe --version   # expect: 1.13.5
```
> Using Docker instead? Skip this step and use the Docker command in Step 3.

### Step 3 — Start Neo4j (REQUIRED)
```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/your_password neo4j:5
```
Wait ~15 seconds. Verify: open http://localhost:7474 → login `neo4j` / `your_password`

> **If you change the password in that command, change `NEO4J_PASSWORD` in
> Step 4 to the exact same value.** They must match or you get `AuthError`.

### Step 4 — Configure + load knowledge graph
```bash
copy .env.example .env      # Windows
```
Edit `.env`:
```bash
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx      # REQUIRED — from console.groq.com/keys
NEO4J_PASSWORD=your_password          # must EXACTLY match Step 3's docker password
```
Then load the 112 knowledge triples:
```bash
python -c "from src.kg.kg_loader import load_kg; load_kg()"
```
Expected output: `Knowledge graph loaded successfully.`

### Step 5 — Start the app (3 terminals)

**Terminal 1 — LiveKit:**
```bash
tools/livekit/livekit-server.exe --dev
```

**Terminal 2 — Backend:**
```bash
python -m uvicorn backend.main:app --port 8000
```
> First run takes ~30s (builds the FAISS vector index).
> If it hangs on startup, you're in the project root — the `import regex` NLTK
> error happens when CWD contains a conflicting file. Run `python -m uvicorn
> backend.main:app --port 8000` from the `AI_TUTOR/` folder.

**Terminal 3 — Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**Terminal 4 — Voice agent (optional, only for the Live Room):**
```bash
python -m realtime.agent --room tutor-room
```

### Step 6 — Open it
```
http://localhost:5173
```

**Test the tutor is actually grounded:**
1. Chat view → ask *"What is recursion?"*
2. Expand **CoT Visualizer** under the answer
3. You should see *"N% KG-grounded"* and named CS concepts
   → If it says 0% grounded, Neo4j isn't loaded. Redo Step 4.

---

## What works where

| Feature | Needs GPU? |
|---------|-----------|
| KG-RAG tutor, Chain-of-Thought, CoT Visualizer | No |
| Text chat | No |
| Mic → STT → tutor → TTS voice loop | No |
| LiveKit room (WebRTC mic/camera) | No |
| PDF/TXT upload | No |
| Webcam vision capture | No |
| **Lip-synced avatar (MuseTalk/LiveTalking)** | **YES — GPU only** |
| Piper neural TTS (better voice) | Optional |
| Whisper large-v3 (better STT) | Optional |

Everything except the avatar face works on a plain CPU laptop today.

---

## If something breaks

| Symptom | Cause / fix |
|---------|-------------|
| CoT Visualizer shows 0% grounded | Neo4j not running, or KG not loaded → redo Steps 3–4 |
| `Connection refused` port 7880 | LiveKit not started → Step 5 Terminal 1 |
| `Connection refused` port 8000 | Backend not started → Step 5 Terminal 2 |
| `GROQ_API_KEY not set` | `.env` not saved, or key missing |
| `neo4j.AuthError` | `.env` `NEO4J_PASSWORD` ≠ the Docker password |
| `ModuleNotFoundError: backend` | Run from the `AI_TUTOR/` folder, not `frontend/` |
| `Blocked import of regex ... nltk` | CWD conflict — run from `AI_TUTOR/` root |
| Live Room joins but no AI voice | Terminal 4 (voice agent) not running |

More in `REQUIREMENTS.md` §11.