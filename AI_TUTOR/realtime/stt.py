"""Milestone 2: speech-to-text via faster-whisper (CPU, no training).

Browser records mic audio (MediaRecorder webm/opus), the backend transcribes
it and returns text that flows into the existing tutor pipeline. The model
lazy-loads on first use so server startup stays fast.
"""
import os

from dotenv import load_dotenv

load_dotenv()

STT_MODEL = os.getenv("STT_MODEL", "base")

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(STT_MODEL, device="cpu", compute_type="int8")
    return _model


def transcribe_file(path: str) -> str:
    """Transcribe an audio file; VAD filters silence automatically."""
    segments, info = _get_model().transcribe(path, vad_filter=True)
    text = " ".join(s.text.strip() for s in segments if s.text.strip())
    return text.strip()
