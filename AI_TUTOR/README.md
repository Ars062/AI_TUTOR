# AI Tutor

A knowledge-grounded, chain-of-thought AI tutoring system built with Groq (gpt-oss-120b), Neo4j, FAISS, and Streamlit.

## Demo

Watch the full walkthrough: **[AI Tutor Demo Video](https://drive.google.com/file/d/1nVzWK7juqBpJq6PGKZ3S71qMdaXFhziV/view?usp=sharing)**

## Multimodal v2 (`multimodal` branch)

The KG-RAG research core in `src/` is unchanged. New presentation layers are built around it:

```text
frontend/   React (Vite) chat UI          <- replaces Streamlit
backend/    FastAPI wrapper around ask_tutor
realtime/   Phase 2+: Pipecat + LiveKit, faster-whisper STT, TTS, vision
avatar/     GPU machine only: LiveTalking + MuseTalk lip-sync adapter
```

Run the v2 stack locally:

```bash
# terminal 1 - API server
.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000

# terminal 2 - web UI
cd frontend && npm install && npm run dev
# open http://localhost:5173
```

GPU rule: everything above runs on CPU; only realtime avatar inference (LiveTalking/MuseTalk) needs an NVIDIA GPU later.

### Milestone 1 — Live realtime room (done)

Browser connects over WebRTC through a self-hosted [LiveKit](https://github.com/livekit/livekit-server) server:

```bash
# terminal 3 - realtime media server (or use docker compose up livekit)
tools\livekit\livekit-server.exe --dev
```

In the web UI, switch the sidebar to **🎥 Live Room → Join**. Grant mic/camera
permissions; your stream publishes into the `tutor-room`. Session tokens are
issued server-side by `POST /api/session/token` so LiveKit secrets never
reach the browser. Later milestones attach STT/vision/avatar participants to
the same room.

## Architecture

```
Student Question
       |
Hybrid Retriever (KG + Vector Search)
       |
Prompt Builder (CoT + Ensemble + S2A)
       |
Groq LLM (gpt-oss-120b)
       |
Answer + Debug Info
```

### Components
| Module | Purpose |
|---|---|
| `src/kg/` | Neo4j knowledge graph: load CSV triples, query by keyword/entity |
| `src/rag/` | FAISS vector index: embed documents, search, hybrid KG-RAG retriever |
| `src/prompts/` | Prompt engineering: CoT templates, ensemble prompts, S2A filtering |
| `src/tutor/` | Orchestrator: conversation memory, ensemble aggregation |
| `src/evaluation/` | Metrics: similarity, concept coverage, CoT analysis |
| `app/` | Streamlit web UI with debug panel |

## Setup

1. Clone the repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.template` to `.env` and fill in:
   - `GROQ_API_KEY` (required)
   - `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (if using Neo4j)

4. (Optional) Load the knowledge graph:
   ```python
   from src.kg.kg_loader import load_kg
   load_kg("data/knowledge_graph/knowledge_triples.csv")
   ```

5. Start Neo4j (optional but recommended for KG-RAG):
   ```bash
   C:\Users\akju0\.neo4j\start-neo4j.cmd
   ```

6. Run the app:
   ```bash
   run-app.cmd
   ```
   Or manually:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Configuration

See `.env.template` for all config options:
- `DOCUMENTS_DIR` — folder with `.txt`/`.md` documents
- `FAISS_INDEX_PATH` — where to cache the vector index
- `TOP_K_DOCS` — number of documents to retrieve (default: 5)
- `KG_MAX_HOPS` — graph expansion depth (default: 2)
- `USE_ENSEMBLE` — enable ensemble prompting (default: true)
- `USE_S2A` — enable S2A context filtering (default: true)
- `MAX_HISTORY` — conversation memory length (default: 10)

## Testing

```bash
python tests/run_all.py
```

## Domain Data

Add `.txt` or `.md` files to `data/documents/` and triples to `data/knowledge_graph/knowledge_triples.csv`.
