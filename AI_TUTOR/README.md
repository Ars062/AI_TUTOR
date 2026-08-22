# AI Tutor

A knowledge-grounded, chain-of-thought AI tutoring system built with Gemini, Neo4j, FAISS, and Streamlit.

## Architecture

```
Student Question
       |
Hybrid Retriever (KG + Vector Search)
       |
Prompt Builder (CoT + Ensemble + S2A)
       |
Gemini 1.5 Pro
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
   - `GEMINI_API_KEY` (required)
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
