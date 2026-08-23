"""Milestone 6: realtime streaming pipeline.

Processes audio chunks through the full pipeline:
    mic audio → STT → tutor (LLM + KG-RAG) → TTS → audio

Handles turn-taking: detects when the user starts/stops speaking,
waits for silence before processing, and supports barge-in (user
interrupts while AI is speaking).
"""
import asyncio
import io
import time
from typing import Optional

from realtime.stt import transcribe_file
from realtime.tts import synthesize as tts_synthesize
from realtime.vision import describe_placeholder


class TurnManager:
    """Simple rule-based turn manager (MVP: deterministic rules).

    States: LISTENING → PROCESSING → SPEAKING → LISTENING
    Supports barge-in: if user speaks during SPEAKING, abort and return to LISTENING.
    """

    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"

    def __init__(self, silence_threshold: float = 1.5):
        self.state = self.LISTENING
        self.silence_threshold = silence_threshold
        self.last_speech_time = 0.0
        self._abort = asyncio.Event()

    def on_speech_start(self):
        self.last_speech_time = time.time()
        if self.state == self.SPEAKING:
            self._abort.set()
            self.state = self.LISTENING
            return True  # barge-in detected
        return False

    def on_speech_end(self):
        self.last_speech_time = time.time()

    def is_silence(self) -> bool:
        return (time.time() - self.last_speech_time) >= self.silence_threshold

    def request_abort(self):
        self._abort.set()

    def clear_abort(self):
        self._abort.clear()

    def should_abort(self) -> bool:
        return self._abort.is_set()


turn = TurnManager()


async def process_audio_chunk(
    audio_bytes: bytes,
    filename: str = "chunk.webm",
    get_answer_fn=None,
) -> Optional[dict]:
    """Full pipeline: audio → text → tutor → speech.

    Returns {"text": "...", "audio": bytes} or None if processing was aborted.
    """
    import tempfile, os

    ext = os.path.splitext(filename)[1] or ".webm"
    fd, path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        text = await asyncio.to_thread(transcribe_file, path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    if not text.strip():
        return None

    turn.state = turn.PROCESSING
    turn.clear_abort()

    if get_answer_fn:
        answer = await asyncio.to_thread(get_answer_fn, text)
    else:
        answer = f"Echo: {text}"

    if turn.should_abort():
        return None

    turn.state = turn.SPEAKING
    audio = await tts_synthesize(answer[:2000])
    turn.state = turn.LISTENING

    return {"text": answer, "audio": audio, "user_text": text}
