"""FastAPI backend for the multimodal AI Tutor (v2 presentation layer).

Wraps the existing KG-RAG tutor engine unchanged:
    question -> memory -> Neo4j KG + FAISS hybrid retrieval
             -> prompt engine -> Groq -> answer (+ debug info)

Phase 1: text chat. Later phases attach STT/vision inputs and TTS/avatar
outputs around the same `ask_tutor` call.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.rag.embed_documents import load_index
from src.tutor.tutor_engine import ask_tutor, get_memory
from src.prompts.prompt_builder import LEARNER_LEVELS
from backend.realtime import router as realtime_router
from realtime.stt import transcribe_file
from realtime.tts import synthesize as tts_synthesize
from realtime.vision import update_frame, get_visual_context, describe_placeholder
from realtime.pipeline import process_audio_chunk, turn
from backend.tools import registry as tool_registry
from backend.upload import extract_text, chunk_text, add_documents_to_index, save_uploaded_file
from backend.memory import init_db, save_message, get_history

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
    visual_context: str = ""


class VisionFrame(BaseModel):
    description: str = Field(min_length=1)
    confidence: float = 0.8
    frame_b64: str = ""


class ToolCall(BaseModel):
    name: str = Field(min_length=1)
    args: dict = {}


class ChatResponse(BaseModel):
    answer: str
    debug: dict


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1)


@app.on_event("startup")
def _load_resources():
    init_db()
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
    vis = req.visual_context or describe_placeholder()
    question = req.question
    if vis:
        question = f"{question}\n\n[Visual context from webcam: {vis}]"
    answer, debug = ask_tutor(
        question=question,
        index=_state["index"],
        documents=_state["documents"],
        filenames=_state["filenames"],
        session_id=req.session_id,
        use_cot=req.use_cot,
        learner_level=req.learner_level,
    )
    grounded = (debug.get("cot_validation") or {}).get("grounded_fraction")
    save_message(req.session_id, "user", req.question)
    save_message(req.session_id, "assistant", answer, grounded)
    return {"answer": answer, "debug": debug}


@app.get("/api/history/{session_id}")
def history(session_id: str, limit: int = 50):
    return {"messages": get_history(session_id, limit)}


@app.post("/api/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """Milestone 2: mic audio -> faster-whisper -> transcript text."""
    ext = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"
    data = await file.read()
    if not data:
        return {"text": ""}
    fd, path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return {"text": transcribe_file(path)}
    finally:
        os.unlink(path)


@app.post("/api/tts")
async def text_to_speech(req: SpeakRequest):
    """Milestone 4: answer text -> spoken WAV (offline voice)."""
    wav = await tts_synthesize(req.text[:2000])
    if not wav:
        return Response(status_code=500)
    return Response(content=wav, media_type="audio/wav")


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Milestone 9: upload PDF/TXT → chunk → embed → tutor can answer from it."""
    content = await file.read()
    if not content:
        return {"error": "empty file", "chunks": 0}

    saved = save_uploaded_file(content, file.filename or "document.txt")
    raw = extract_text(saved, file.filename or "document.txt")
    chunks = chunk_text(raw)

    if not chunks:
        return {"error": "no extractable text", "chunks": 0}

    names = [f"upload:{os.path.basename(saved)}:{i}" for i in range(len(chunks))]
    _state["index"], _state["documents"], _state["filenames"] = add_documents_to_index(
        _state["index"], _state["documents"], _state["filenames"],
        chunks, names,
    )
    return {"ok": True, "chunks": len(chunks), "total_docs": len(_state["documents"])}


@app.post("/api/vision")
def vision_frame(req: VisionFrame):
    """Milestone 7: browser sends webcam frame description → tutor uses it."""
    update_frame(req.description, req.confidence, req.frame_b64)
    return {"ok": True}


@app.get("/api/tools")
def list_tools():
    """List available tools the agent can call."""
    return {"tools": tool_registry.schema()}


@app.post("/api/tools/call")
def call_tool(req: ToolCall):
    """Execute a tool by name."""
    return {"result": tool_registry.call(req.name, **req.args)}


@app.post("/api/realtime/stream")
async def realtime_stream(file: UploadFile = File(...)):
    """Milestone 6: mic audio → STT → tutor → TTS → audio back.

    Full realtime loop. Supports barge-in via turn manager.
    """
    content = await file.read()
    if not content:
        return {"error": "empty audio"}

    def _get_answer(text: str) -> str:
        answer, _ = ask_tutor(
            question=text,
            index=_state["index"],
            documents=_state["documents"],
            filenames=_state["filenames"],
            session_id="realtime",
            use_cot=True,
            learner_level="beginner",
        )
        return answer

    result = await process_audio_chunk(
        content, file.filename or "chunk.webm", get_answer_fn=_get_answer
    )
    if not result:
        return {"error": "no speech detected", "text": ""}

    return Response(content=result["audio"], media_type="audio/wav")
