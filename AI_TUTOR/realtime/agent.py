import triton
import asyncio
import io
import os
import threading
import time
import wave
from typing import Optional

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

from realtime.avatar import AVATAR_FPS, AVATAR_HEIGHT, AVATAR_WIDTH, AvatarProcessor
from realtime.tts import synthesize as tts_synthesize

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

AUDIO_OUT_SAMPLE_RATE = int(os.getenv("AVATAR_AUDIO_SAMPLE_RATE", "48000"))
WHISPER_MODEL = os.getenv("STT_MODEL", "base")
WHISPER_DEVICE = os.getenv("STT_DEVICE", "cpu")
_WHISPER_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def _stt_ready() -> bool:
    import glob

    model_dir = os.path.join(_WHISPER_CACHE_DIR, "models--Systran--faster-whisper-base")
    if not os.path.exists(model_dir):
        return False
    if glob.glob(os.path.join(model_dir, "**", "*.incomplete"), recursive=True):
        return False
    return True


_INDEX_LOAD_LOCK = threading.Lock()


class TutorProcessor(FrameProcessor):
    """Turns student speech (TranscriptionFrame) into a spoken tutor answer.

    Uses the same RAG+CoT engine as the text chat (src.tutor.tutor_engine),
    with an HTTP fallback to the backend /api/chat endpoint if the index
    cannot be loaded in-process.
    """

    def __init__(self):
        super().__init__()
        self._index = None
        self._documents = []
        self._filenames = []
        self._index_state = "unloaded"
        self._answer_task: Optional[asyncio.Task] = None

    def _load_index(self):
        if self._index_state == "loaded":
            return
        with _INDEX_LOAD_LOCK:
            if self._index_state == "loaded":
                return
            self._index_state = "loading"
            try:
                from src.rag.embed_documents import load_index
                self._index, self._documents, self._filenames = load_index()
                n = len(self._documents or [])
                self._index_state = "loaded" if n else "empty"
                logger.info(f"Voice tutor RAG index loaded: {n} docs")
            except Exception as e:
                logger.warning(f"Voice tutor RAG index load failed: {e}")
                self._index_state = "failed"

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            text = (frame.text or "").strip()
            print(f"[tutor] got transcript: {text!r}", flush=True)
            if len(text) >= 3:
                if self._answer_task and not self._answer_task.done():
                    self._answer_task.cancel()
                self._answer_task = asyncio.create_task(self._answer(text))
            return
        await self.push_frame(frame, direction)

    async def _answer(self, question: str):
        print(f"[tutor] answer start: {question[:60]!r}", flush=True)
        try:
            answer = ""
            if self._index_state != "loaded":
                await asyncio.to_thread(self._load_index)
            if self._index_state == "loaded":
                try:
                    answer = await asyncio.to_thread(self._ask_inprocess, question)
                except Exception as e:
                    logger.warning(f"in-process ask failed: {e}; falling back to backend API")
            if not answer:
                answer = await self._ask_backend(question)
            if answer:
                print(f"[tutor] Q: {question[:80]} -> A: {len(answer)} chars", flush=True)
                await self.push_frame(TTSSpeakFrame(text=answer[:1500]))
                print("[tutor] pushed TTSSpeakFrame downstream", flush=True)
            else:
                print(f"[tutor] no answer for: {question[:80]}", flush=True)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"Tutor answer failed: {e}")

    def _ask_inprocess(self, question: str) -> str:
        from src.tutor.tutor_engine import ask_tutor
        answer, _debug = ask_tutor(
            question=question,
            index=self._index,
            documents=self._documents,
            filenames=self._filenames,
            session_id="live-voice",
            use_cot=False,
            use_kg=True,
        )
        return answer or ""

    async def _ask_backend(self, question: str) -> str:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://127.0.0.1:8000/api/chat",
                    json={"question": question, "session_id": "live-voice", "use_cot": False},
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("answer") or ""
        except Exception as e:
            logger.warning(f"backend chat fallback failed: {e}")
        return ""


class TTSProcessor(FrameProcessor):
    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSSpeakFrame) and direction == FrameDirection.DOWNSTREAM:
            text = frame.text
            if not text:
                return
            wav_bytes = await tts_synthesize(text[:2000])
            if not wav_bytes:
                print("[tts] synthesis returned no audio", flush=True)
                return
            audio_frame = self._wav_to_audio_frame(wav_bytes)
            if audio_frame:
                print(f"[tts] pushing audio frame {len(audio_frame.audio)}B sr={audio_frame.sample_rate}", flush=True)
                await self.push_frame(audio_frame)
        else:
            await self.push_frame(frame, direction)

    @staticmethod
    def _wav_to_audio_frame(wav_bytes: bytes):
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                sample_width = wf.getsampwidth()
                pcm_data = wf.readframes(wf.getnframes())
        except Exception:
            return None
        return OutputAudioRawFrame(audio=pcm_data, sample_rate=sample_rate, num_channels=channels)


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
            audio_out_sample_rate=AUDIO_OUT_SAMPLE_RATE,
            video_out_enabled=True,
            video_out_is_live=True,
            video_out_width=AVATAR_WIDTH,
            video_out_height=AVATAR_HEIGHT,
            video_out_framerate=AVATAR_FPS,
            video_out_color_format="RGB",
        ),
    )
    processors: list = []
    if _stt_ready():
        vad_analyzer = SileroVADAnalyzer(params=VADParams(confidence=0.6, min_volume=0.1))
        _orig_run = vad_analyzer._run_analyzer
        _orig_conf = vad_analyzer.voice_confidence
        _dbg = {"last": "", "t": 0.0, "conf": 0.0}

        def _conf(buf):
            c = _orig_conf(buf)
            _dbg["conf"] = c
            return c

        vad_analyzer.voice_confidence = _conf

        def _f(x):
            try:
                return float(x)
            except Exception:
                try:
                    return float(x.reshape(-1)[0])
                except Exception:
                    return -1.0

        _cap = {"b": bytearray(), "sr": 0, "done": False}

        def _capture(buffer):
            if _cap["done"]:
                return
            _cap["b"] += buffer
            try:
                _cap["sr"] = int(vad_analyzer.sample_rate) or 16000
            except Exception:
                _cap["sr"] = 16000
            if len(_cap["b"]) >= _cap["sr"] * 2 * 15:
                try:
                    with wave.open("/tmp/opencode/vad_cap.wav", "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(_cap["sr"])
                        w.writeframes(bytes(_cap["b"]))
                    print(f"[vad] captured {len(_cap['b'])}B sr={_cap['sr']} -> /tmp/opencode/vad_cap.wav", flush=True)
                except Exception as exc:
                    print(f"[vad] capture failed: {exc}", flush=True)
                _cap["done"] = True

        def _run_dbg(buffer):
            _capture(buffer)
            state = _orig_run(buffer)
            now = time.monotonic()
            tag = f"{state}"
            if tag != _dbg["last"] or now - _dbg["t"] >= 2.0:
                print(
                    f"[vad] vol={_f(vad_analyzer._prev_volume):.3f} conf={_f(_dbg['conf']):.2f} state={tag}",
                    flush=True,
                )
                _dbg["last"] = tag
                _dbg["t"] = now
            return state

        vad_analyzer._run_analyzer = _run_dbg
        processors += [
            VADProcessor(vad_analyzer=vad_analyzer),
            WhisperSTTService(
                device=WHISPER_DEVICE,
                compute_type="int8",
                settings=WhisperSTTService.Settings(model=WHISPER_MODEL),
                audio_passthrough=False,
            ),
        ]
    else:
        print("[agent] faster-whisper model not downloaded yet: lecture mode (no voice input). Run realtime/ensure_stt_model.py and restart.")
    tutor_proc = TutorProcessor()
    processors += [tutor_proc, TTSProcessor(), AvatarProcessor(width=AVATAR_WIDTH, height=AVATAR_HEIGHT, fps=AVATAR_FPS)]
    tutor_proc._preload_task = asyncio.create_task(asyncio.to_thread(tutor_proc._load_index))
    out = transport.output()
    _orig_write = out.write_audio_frame
    async def _dbg_write(frame):
        ok = await _orig_write(frame)
        print(f"[audio-out] {len(frame.audio)}B sr={frame.sample_rate} ok={ok}", flush=True)
        return ok
    out.write_audio_frame = _dbg_write
    pipeline = Pipeline([transport.input(), *processors, out])
    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=False), setup_timeout_secs=90, enable_rtvi=False, idle_timeout_secs=None)
    runner = PipelineRunner()
    greeting_task = None
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant_id):
        nonlocal greeting_task
        print(f"[agent] Participant joined: {participant_id}")
        async def say_hi():
            try:
                await asyncio.sleep(1.0)
                await task.queue_frame(TTSSpeakFrame(text="Hello! I'm your AI tutor. What would you like to learn today?"))
                print("[agent] Greeting queued", flush=True)
            except Exception as e:
                print(f"[agent] Greeting failed: {e}", flush=True)
        greeting_task = asyncio.create_task(say_hi())
    @transport.event_handler("on_participant_disconnected")
    async def on_participant_disconnected(transport, participant_id):
        print(f"[agent] Participant left: {participant_id}")
    await runner.run(task)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Tutor LiveKit Agent")
    parser.add_argument("--room", default="tutor-room", help="LiveKit room name")
    args = parser.parse_args()
    try:
        asyncio.run(run_agent(args.room))
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise


if __name__ == "__main__":
    main()
