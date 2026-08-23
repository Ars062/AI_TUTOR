"""FastAPI backend for the multimodal AI Tutor (v2 presentation layer).

Wraps the existing KG-RAG tutor engine unchanged:
    question -> memory -> Neo4j KG + FAISS hybrid retrieval
             -> prompt engine -> Groq -> answer (+ debug info)

Phase 1: text chat. Later phases attach STT/vision inputs and TTS/avatar
outputs around the same `ask_tutor` call.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag.embed_documents import load_index
from src.tutor.tutor_engine import ask_tutor, get_memory
from src.prompts.prompt_builder import LEARNER_LEVELS
from backend.realtime import router as realtime_router

app = FastAPI(title="AI Tutor Backend", version="2.0")

app.include_router(realtime_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {"index": None, "documents": None, "filenames": None}


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    use_cot: bool = True
    learner_level: str = "beginner"
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    debug: dict


@app.on_event("startup")
def _load_resources():
    index, documents, filenames = load_index()
    if index.ntotal == 0:
        from src.rag.embed_documents import build_vector_index, save_index

        index, documents, filenames = build_vector_index()
        save_index(index, documents, filenames)
    _state.update(index=index, documents=documents, filenames=filenames)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "documents": len(_state["documents"] or []),
        "index_size": int(_state["index"].ntotal) if _state["index"] else 0,
        "learner_level": get_memory().student_level,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer, debug = ask_tutor(
        question=req.question,
        index=_state["index"],
        documents=_state["documents"],
        filenames=_state["filenames"],
        session_id=req.session_id,
        use_cot=req.use_cot,
        learner_level=req.learner_level,
    )
    return {"answer": answer, "debug": debug}
