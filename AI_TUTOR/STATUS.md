# AI Tutor — Work Status

Last updated: 2026-08-20

## Project: Knowledge-Grounded + Chain-of-Thought AI Tutoring System
Proposal: `D:\AI_TUTOR\Proposal (1).docx`

---

## Bug Fixes (this session)

| # | Bug | Fix | File |
|---|-----|-----|------|
| 1 | CoT answer had spaced-out characters (R\ne\nc...) | Removed duplicate `## Final Answer` section; renamed Step 5 → Conclude | `src/prompts/prompt_builder.py` |
| 2 | Test failed: `GEMINI_API_KEY` import error | Renamed to `GROQ_API_KEY` | `tests/test_config.py` |
| 3 | KG returned empty for natural questions (e.g. "what is recursion") | Added keyword extraction + multi-keyword querying | `src/kg/kg_query.py` |
| 4 | LLM 404: `llama-3.3-70b-versatile` no longer exists on Groq | Switched to `openai/gpt-oss-120b` | `src/config.py`, `.env`, `.env.template` |
| 5 | Streamlit torchvision import error spam | Harmless noise from transformers watcher; launch with `--server.fileWatcherType none` | (no code change) |
| 6 | KG node matching was keyword-only (missed paraphrased queries) | Implemented embedding-based anchor nodes (`all-MiniLM-L6-v2` cosine similarity + threshold, keyword fallback) | `src/kg/kg_query.py`, `src/rag/embeddings.py`, `src/config.py` |

## Setup (this session)

- ygrep installed for opencode (skill + binary) — indexed the project
- Neo4j 5.26.28 installed (portable) → `C:\Users\akju0\.neo4j\neo4j-unpacked\`
- JDK 17 installed (portable) → `C:\Users\akju0\.neo4j\jdk17\`
- Helper scripts created:
  - `C:\Users\akju0\.neo4j\start-neo4j.cmd` — starts Neo4j
  - `D:\AI_TUTOR\AI_TUTOR\run-app.cmd` — starts Streamlit app

## Prior Session Work (already in repo, uncommitted)

- Evaluation metrics: BLEU, ROUGE, BERTScore (lazy-loaded) — `src/evaluation/evaluation_metrics.py`
- Batch evaluation + report formatting — `run_batch_evaluation()`, `format_report()`
- Rate-limit retry with backoff in LLM calls — `src/tutor/tutor_engine.py`
- Evaluation runner script — `scripts/run_evaluation.py`
- Quiz dataset (12 CS questions + references) — `data/evaluation/quiz.json`
- One-click Windows launcher — `start_windows.bat` (JAVA_HOME paths point to `C:\Users\akju0\dev\tools`, may need updating)
- One evaluation run done with OLD model — `data/evaluation/run_log.txt`, `report_cot.json`
  - CoT: BLEU 0.05, ROUGE-L 0.15, BERTScore F1 0.79, coverage 0.88
  - Was extremely slow (rate-limited, ~88 min/question)
- Evaluation re-run with NEW model — `data/evaluation/report_new_model_heavy.json`
  - CoT: BLEU 0.031, ROUGE-1 0.155, ROUGE-L 0.133, BERTScore F1 0.755, coverage 0.94, latency 18.3s/q
  - No-CoT: coverage 0.885, similarity 0.027, latency 13.3s/q
  - ~300x faster than old run (no rate limiting at current usage); coverage improved, lexical metrics slightly lower

## Verified Working

- 26/26 tests pass: `python tests/run_all.py`
- Hybrid retriever returns: KG facts + vector docs + KG-guided docs
- `query_kg("what is recursion")` → 5 facts
- New model `openai/gpt-oss-120b` responds correctly
- Embedding-based KG anchor matching: `find_anchor_nodes()` finds semantically close Concept nodes (e.g. "tell me about bubble sort" → Bubble Sort, Sorting Algorithms, Time Complexity)

## This Session (remaining proposal items)

- Per-CoT-step **KG validation** — `validate_cot_steps_against_kg()` flags ungrounded steps (`src/evaluation/evaluation_metrics.py`), wired into `ask_tutor` debug output
- **CoT visualizer** + **feedback rating** in Streamlit UI — collapsible step-by-step view with KG-grounding %, `st.feedback` thumbs logged to `data/evaluation/feedback.jsonl`
- **Rubric scores** — `score_logical_consistency()` (0-5) & `score_explainability()` (1-5) heuristics added to batch reports (human rubric still recommended for final submission)
- **Human study harness** — `src/evaluation/study_harness.py` + `scripts/run_study.py`:
  - `split_pre_post()` balances topics across pre/post quizzes
  - `score_answer()` auto proxy (extractive + concept coverage, 0-100)
  - `analyze_study()` paired t-test learning gain; `analyze_survey()` Likert means
  - survey template at `data/study/survey.json`; pilot runs end-to-end (post scores 60-98)
- **Content-safety guard** — `_is_unsafe()` blocklist in `tutor_engine.py` (config `ENABLE_CONTENT_SAFETY`)

## How to Run

1. Start Neo4j (keep window open):
   ```
   C:\Users\akju0\.neo4j\start-neo4j.cmd
   ```
2. Start app:
   ```
   cd D:\AI_TUTOR\AI_TUTOR
   run-app.cmd
   ```
3. Open `http://localhost:8501`

## Remaining Work (~15% of proposal)

- Real human study with ~20 participants (harness is built & pilot-tested; needs real users)
- Full multi-hop semantic KG traversal
- Human-in-the-loop prompt refinement sessions
- Ethical/safety review write-up (code-level content guard is in place)

## Notes

- Groq free tier: 100k tokens/day. Ensemble mode = 4 LLM calls per question → rate limits hit fast. Consider `USE_ENSEMBLE=false` in `.env` for testing.
- Neo4j launched from this agent's shell does NOT persist between commands. It must be started from a user terminal window.