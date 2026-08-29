"""Pipecat voice agent: LiveKit transport + faster-whisper STT + KG-RAG tutor + Windows TTS.

Runs as a standalone process alongside the FastAPI backend.
Joins a LiveKit room, listens for user speech, routes through the tutor,
and speaks the reply back.

Usage:
    python -m realtime.agent                      # auto-configure from env
    python -m realtime.agent --room tutor-room    # specify room

Env vars (from .env):
    LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
    GROQ_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""
import asyncio
import io
import os
import struct
import sys
import wave

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv()

from pipecat.frames.frames import (
    OutputAudioRawFrame,
    TTSSpeakFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

from realtime.tts import synthesize as tts_synthesize


# ---------------------------------------------------------------------------
# Custom processors
# ---------------------------------------------------------------------------

class TutorProcessor(FrameProcessor):
    """Takes TranscriptionFrame → calls KG-RAG tutor → outputs TTSSpeakFrame."""

    def __init__(self):
        super().__init__()
        self._index = None
        self._documents = None
        self._filenames = None

    def _ensure_loaded(self):
        if self._index is not None:
            return
        from src.rag.embed_documents import load_index
        self._index, self._documents, self._filenames = load_index()
        if self._index.ntotal == 0:
            from src.rag.embed_documents import build_vector_index, save_index
            self._index, self._documents, self._filenames = build_vector_index()
            save_index(self._index, self._documents, self._filenames)

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            text = frame.text.strip()
            if not text:
                return

            print(f"[tutor] User: {text}")

            def _run():
                from src.tutor.tutor_engine import ask_tutor
                answer, _ = ask_tutor(
                    question=text,
                    index=self._index,
                    documents=self._documents,
                    filenames=self._filenames,
                    session_id="livekit-agent",
                    use_cot=False,
                    learner_level="beginner",
                )
                return answer

            answer = await asyncio.to_thread(_run)
            print(f"[tutor] Bot: {answer[:120]}...")
            await self.push_frame(TTSSpeakFrame(text=answer))

        else:
            await self.push_frame(frame, direction)


class WindowsTTSProcessor(FrameProcessor):
    """Takes TTSSpeakFrame → generates WAV via System.Speech → outputs OutputAudioRawFrame."""

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSSpeakFrame) and direction == FrameDirection.DOWNSTREAM:
            text = frame.text
            if not text:
                return

            wav_bytes = await tts_synthesize(text[:2000])
            if not wav_bytes:
                return

            audio_frame = self._wav_to_audio_frame(wav_bytes)
            if audio_frame:
                await self.push_frame(audio_frame)
        else:
            await self.push_frame(frame, direction)

    @staticmethod
    def _wav_to_audio_frame(wav_bytes: bytes):
        """Convert WAV bytes to OutputAudioRawFrame."""
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                sample_width = wf.getsampwidth()
                pcm_data = wf.readframes(wf.getnframes())
        except Exception:
            return None

        return OutputAudioRawFrame(
            audio=pcm_data,
            sample_rate=sample_rate,
            num_channels=channels,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_agent(room_name: str = "tutor-room"):
    url = os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880")
    api_key = os.environ.get("LIVEKIT_API_KEY", "devkey")
    api_secret = os.environ.get("LIVEKIT_API_SECRET", "secret")

    from pipecat.runner.livekit import generate_token_with_agent
    token = generate_token_with_agent(room_name, "AI-Tutor", api_key, api_secret)

    print(f"[agent] Connecting to {url} room={room_name}")

    transport = LiveKitTransport(
        url=url,
        token=token,
        room_name=room_name,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    stt = WhisperSTTService(model="base", device="cpu", compute_type="int8")
    tutor = TutorProcessor()
    tts = WindowsTTSProcessor()

    pipeline = Pipeline([
        transport.input(),
        stt,
        tutor,
        tts,
        transport.output(),
    ])

    task = PipelineTask(
        pipeline,
        PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=False,
        ),
    )

    runner = PipelineRunner()

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant_id):
        print(f"[agent] Participant joined: {participant_id}")
        await asyncio.sleep(1)
        await task.queue_frame(
            TTSSpeakFrame(text="Hello! I'm your AI tutor. What would you like to learn today?")
        )

    @transport.event_handler("on_participant_disconnected")
    async def on_participant_disconnected(transport, participant_id):
        print(f"[agent] Participant left: {participant_id}")

    await runner.run(task)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pipecat voice tutor agent")
    parser.add_argument("--room", default="tutor-room", help="LiveKit room name")
    args = parser.parse_args()

    asyncio.run(run_agent(args.room))


if __name__ == "__main__":
    main()
