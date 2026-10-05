"""Realtime talking-teacher avatar (pipecat FrameProcessor).

Sits between the TTS and the transport in the voice agent pipeline:

    tts -> AvatarProcessor -> transport.output()   (audio + video)

AvatarProcessor consumes ``OutputAudioRawFrame`` (16-bit PCM) emitted by the
TTS, renders one RGB image per video-frame via a pluggable lipsync engine,
and pushes both audio slices and video frames downstream, paced on the wall
clock so the published LiveKit video track looks live.

Engines (AVATAR_ENGINE):
  auto     (default)  pick the best available engine
  puppet   amplitude-driven mouth on a fixed teacher portrait; always ready
  musetalk real neural lip-sync (TMElyralab/MuseTalk); used when installed AND
           weights present under models/musetalk (see ENGINES section below)

Env vars:
  AVATAR_ENGINE   auto | puppet | musetalk            (default auto)
  AVATAR_FACE     path to teacher portrait PNG        (default avatar/teacher.png)
  AVATAR_FPS      output video frame rate             (default 25)
  AUDIO_OUT_SAMPLE_RATE  transport output rate        (default 48000)

Usage as a standalone preview (writes a rendered .mp4 of a sample sentence):
    python -m realtime.avatar --preview out.mp4
"""
import asyncio
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv()

from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_FACE = os.path.join(ROOT, "avatar", "teacher.png")
MOUTH_BOX = (437, 448, 150, 110)  # matches avatar/make_teacher_avatar.py
EYE_BOXES = [(418, 328, 68, 40), (538, 328, 68, 40)]

AVATAR_WIDTH = int(os.getenv("AVATAR_WIDTH", "1024"))
AVATAR_HEIGHT = int(os.getenv("AVATAR_HEIGHT", "768"))
AVATAR_FPS = int(os.getenv("AVATAR_FPS", "25"))
AUDIO_OUT_SAMPLE_RATE = int(os.getenv("AUDIO_OUT_SAMPLE_RATE", "48000"))


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def resample_int16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-resample 16-bit PCM (handles e.g. piper 22050 -> 48000)."""
    if src_rate == dst_rate or not pcm:
        return pcm
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n = len(a)
    if n < 2:
        return a.astype(np.int16).tobytes()
    target = max(1, int(round(n * dst_rate / src_rate)))
    x_old = np.linspace(0.0, n - 1.0, n)
    x_new = np.linspace(0.0, n - 1.0, target)
    return np.interp(x_new, x_old, a).astype(np.int16).tobytes()


def pcm_rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


# ---------------------------------------------------------------------------
# Lipsync engine base + implementations
# ---------------------------------------------------------------------------

class LipsyncEngine:
    """Renders one RGB ndarray (H, W, 3) per video frame for a speech chunk.

    ``render`` receives the utterance audio as float32 mono at ``sample_rate``
    and returns one image per output video frame.  Frames are indexed on the
    wall clock by the processor, so engines may return exactly the right
    number of frames for the audio duration.
    """

    name = "base"

    def __init__(self, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps

    def is_ready(self) -> bool:
        return True

    async def render(self, audio: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        raise NotImplementedError

    def idle_frame(self, t: float, blinking: bool) -> np.ndarray:
        raise NotImplementedError

    async def close(self):
        pass


class PuppetEngine(LipsyncEngine):
    """Amplitude-driven lip-sync over the generated teacher portrait.

    Pre-renders a strip of mouth-open variants (one per frame rate of the
    rendered canvas), then for each video frame overlays the variant whose
    opening matches the smoothed RMS level of that frame's audio slice.  A
    gentle vertical bob and periodic blink keep it feeling alive while idle.
    """

    name = "puppet"

    def __init__(self, width: int, height: int, fps: int, face_path: str = ""):
        super().__init__(width, height, fps)
        self._face_path = face_path or os.getenv("AVATAR_FACE", DEFAULT_FACE)
        if not os.path.exists(self._face_path):
            self._generate_face(self._face_path)
        self._base = self._load_face(self._face_path)
        self._mouth_levels, self._blink_patch = self._build_props(self._base)
        self._t0 = time_monotonic()

    # -- asset loading -----------------------------------------------------

    def _generate_face(self, path: str):
        from avatar.make_teacher_avatar import draw_teacher  # local authoring

        draw_teacher()
        if not os.path.exists(path):
            raise RuntimeError(f"avatar face not found: {path}")

    def _load_face(self, path: str) -> np.ndarray:
        from PIL import Image

        img = Image.open(path).convert("RGB").resize((self.width, self.height))
        return np.asarray(img, dtype=np.uint8)

    def _build_props(self, base: np.ndarray):
        mx, my, mw, mh = MOUTH_BOX
        # Skin block used to erase the closed smile before painting an open mouth.
        from PIL import Image, ImageDraw

        skin = (232, 185, 139)
        lip_dark = (98, 28, 36)
        lip_line = (150, 52, 52)
        teeth = (248, 244, 238)

        levels: list[np.ndarray] = []
        n = max(1, int(mh // 4))
        for i in range(n + 1):
            open_px = int(4 + i * (mh - 18) * 0.8 / n)
            canv = base[my:my + mh, mx:mx + mw].copy()
            canv[:] = skin
            pil = Image.fromarray(canv)
            pd = ImageDraw.Draw(pil)
            # upper (closed) lip line
            pd.rounded_rectangle((14, 2, mw - 14, 12), radius=6, fill=lip_line)
            # mouth opening
            top = 14
            h = max(2, open_px)
            pd.rounded_rectangle((20, top, mw - 20, top + h), radius=max(4, h // 3),
                                 fill=lip_dark)
            # teeth hint
            pd.rectangle((24, top + 2, mw - 24, top + 10), fill=teeth)
            levels.append(np.asarray(pil, dtype=np.uint8))

        # Blink patch: skin + lid line over each eye.
        from PIL import Image, ImageDraw

        blink = base.copy()
        bp = Image.fromarray(blink)
        bd = ImageDraw.Draw(bp)
        for ex, ey, ew, eh in EYE_BOXES:
            bd.ellipse((ex, ey, ex + ew, ey + eh), fill=skin)
            bd.arc((ex, ey - 2, ex + ew, ey + eh), start=20, end=200, fill=(180, 120, 90), width=4)
        blink = np.asarray(bp, dtype=np.uint8)

        return levels, blink

    # -- engine interface --------------------------------------------------

    async def render(self, audio: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        if audio.size == 0:
            return []
        dur = audio.size / sample_rate
        n_frames = max(1, int(round(dur * self.fps)))
        rms_all = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
        gate = max(24.0, rms_all * 0.08)
        mx, my, mw, mh = MOUTH_BOX

        frames: list[np.ndarray] = []
        level = 0.0
        n_levels = len(self._mouth_levels) - 1
        slices = np.array_split(audio, n_frames) if n_frames > 1 else [audio]
        for chunk in slices:
            if chunk.size:
                r = float(np.sqrt(np.mean(chunk * chunk)))
            else:
                r = 0.0
            target = 0.0 if r < gate else min(1.0, (r - gate) / max(1.0, rms_all))
            level += (target - level) * (0.55 if target > level else 0.10)
            idx = int(round(level * n_levels))
            canv = self._base.copy()
            canv[my:my + mh, mx:mx + mw] = self._mouth_levels[idx]
            frames.append(canv)
        return frames

    def idle_frame(self, t: float, blinking: bool) -> np.ndarray:
        if blinking:
            return self._blink_patch.copy()
        dy = int(1.0 * math.sin(t * 1.1))
        return np.roll(self._base, dy, axis=0)

    async def close(self):
        pass


class MuseTalkEngine(LipsyncEngine):
    """Real neural lip-sync via TMElyralab/MuseTalk (GPU)."""

    name = "musetalk"

    def __init__(self, width: int, height: int, fps: int):
        super().__init__(width, height, fps)
        self._reason = ""
        weights = os.getenv("MUSETALK_WEIGHTS", os.path.join(ROOT, "models", "musetalk"))
        self._weights = weights

    def is_ready(self) -> bool:
        try:
            import musetalk  # noqa: F401
        except ImportError as exc:
            self._reason = f"musetalk package not installed ({exc})"
            return False
        if not os.path.isdir(self._weights):
            self._reason = f"MuseTalk weights missing at {self._weights}"
            return False
        return True

    async def render(self, audio: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        if not self.is_ready():
            raise RuntimeError(self._reason)
        raise NotImplementedError("MuseTalkEngine.render: future work")


def time_monotonic() -> float:
    import time

    return time.monotonic()


def build_engine(width: int, height: int, fps: int, kind: str = "") -> LipsyncEngine:
    kind = (kind or os.getenv("AVATAR_ENGINE", "auto")).lower()
    if kind == "auto":
        musetalk = MuseTalkEngine(width, height, fps)
        if musetalk.is_ready():
            print(f"[avatar] engine: musetalk ({musetalk._weights})")
            return musetalk
        if musetalk._reason:
            print(f"[avatar] musetalk unavailable ({musetalk._reason}); using puppet")
        return PuppetEngine(width, height, fps)
    if kind == "musetalk":
        eng = MuseTalkEngine(width, height, fps)
        if not eng.is_ready():
            raise RuntimeError(eng._reason)
        return eng
    if kind == "puppet":
        return PuppetEngine(width, height, fps)
    raise ValueError(f"unknown AVATAR_ENGINE: {kind}")


# ---------------------------------------------------------------------------
# AvatarProcessor
# ---------------------------------------------------------------------------

class AvatarProcessor(FrameProcessor):
    """Turns TTS PCM into paced RGB frames + matching PCM slices downstream.

    Owns audio rate conversion: it resamples incoming audio to
    ``self.audio_sample_rate`` (== transport output rate) so the audio slices
    it emits already match what the LiveKit transport expects.
    """

    def __init__(
        self,
        *,
        engine: LipsyncEngine | None = None,
        width: int = 1024,
        height: int = 768,
        fps: int = 25,
        audio_sample_rate: int = 48000,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps
        self.audio_sample_rate = audio_sample_rate
        self.engine = engine or build_engine(width, height, fps)
        self._speaking = False
        self._emit_task: asyncio.Task | None = None
        self._idle_task: asyncio.Task | None = None
        self._first_frame_seen = False
        self._blink_t = time_monotonic()
        self._idle_enabled = os.getenv("AVATAR_IDLE", "0") == "1"

    # -- lifecycle ---------------------------------------------------------

    async def _run_idle(self):
        try:
            step = 1.0 / max(2, int(self.fps / 4))
            t = 0.0
            while True:
                await asyncio.sleep(step)
                if self._speaking:
                    continue
                t += step
                now = time_monotonic()
                blinking = (now - self._blink_t) % 4.2 < 0.18
                frame = self.engine.idle_frame(t, blinking)
                await self.push_frame(
                    OutputImageRawFrame(
                        image=frame.astype(np.uint8, copy=False).tobytes(),
                        size=(self.width, self.height),
                        format="RGB",
                    )
                )
        except asyncio.CancelledError:
            pass

    # -- frame handling ----------------------------------------------------

    async def process_frame(self, frame, direction: FrameDirection):
        # Audio frames are consumed: the avatar emits its own paced, resampled
        # audio slices alongside the rendered video frames.  Everything else is
        # passed through.
        if direction == FrameDirection.DOWNSTREAM and isinstance(
            frame, (OutputAudioRawFrame, TTSAudioRawFrame, AudioRawFrame)
        ):
            if frame.audio and frame.sample_rate > 0:
                if self._emit_task and not self._emit_task.done():
                    self._emit_task.cancel()
                self._emit_task = asyncio.create_task(self._emit_utterance(frame))
            return

        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if not self._first_frame_seen:
            self._first_frame_seen = True
            if self._idle_enabled and self._idle_task is None:
                self._idle_task = asyncio.create_task(self._run_idle())

        if isinstance(frame, StartFrame):
            if self._idle_enabled and (self._idle_task is None or self._idle_task.done()):
                self._idle_task = asyncio.create_task(self._run_idle())

        if isinstance(frame, EndFrame):
            if self._emit_task:
                self._emit_task.cancel()
                self._emit_task = None
            self._speaking = False
            if self._idle_task:
                self._idle_task.cancel()
                self._idle_task = None

        if isinstance(frame, InterruptionFrame):
            if self._emit_task:
                self._emit_task.cancel()
                self._emit_task = None
            self._speaking = False

    async def _emit_utterance(self, frame):
        try:
            self._speaking = True
            pcm = resample_int16(frame.audio, frame.sample_rate, self.audio_sample_rate)
            n_audio = len(pcm) // 2
            if n_audio <= 0:
                self._speaking = False
                return

            audio_f32 = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            video_frames = await self.engine.render(audio_f32, self.audio_sample_rate)
            n_vid = len(video_frames)
            print(f"[avatar] emit: audio={n_audio} samples, video_frames={n_vid}", flush=True)
            if n_vid == 0:
                await self.push_frame(
                    OutputAudioRawFrame(audio=audio_f32.astype(np.int16).tobytes(),
                                        sample_rate=self.audio_sample_rate, num_channels=1)
                )
                self._speaking = False
                return

            # Slice audio into per-video-frame PCM chunks.
            pts = [int(round(n_audio * i / n_vid)) for i in range(n_vid + 1)]

            for i in range(n_vid):
                if not self._speaking:
                    break
                lo, hi = pts[i], pts[i + 1]
                slice_bytes = pcm[2 * lo:2 * hi]
                await self.push_frame(
                    OutputImageRawFrame(
                        image=video_frames[i].astype(np.uint8, copy=False).tobytes(),
                        size=(self.width, self.height),
                        format="RGB",
                    )
                )
                await self.push_frame(
                    OutputAudioRawFrame(audio=slice_bytes,
                                        sample_rate=self.audio_sample_rate, num_channels=1)
                )
                await asyncio.sleep(1.0 / self.fps)
        except asyncio.CancelledError:
            raise
        finally:
            self._speaking = False


# ---------------------------------------------------------------------------
# Standalone preview (render a sample sentence to an .mp4 via ffmpeg)
# ---------------------------------------------------------------------------

async def _preview(out_path: str = "/tmp/opencode/avatar_preview.mp4"):
    from realtime.tts import synthesize

    print("[preview] synthesizing sample speech...")
    wav = await synthesize("Welcome to your AI tutor. Ask me anything about computer science, "
                           "and I will walk you through it step by step.")
    import io
    import wave

    with wave.open(io.BytesIO(wav), "rb") as wf:
        sr = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())

    engine = PuppetEngine(1024, 768, 25)
    audio_f32 = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    frames = await engine.render(audio_f32, sr)
    print(f"[preview] {len(frames)} frames, {audio_f32.size / sr:.2f}s audio")

    frames.append(engine.idle_frame(0.0, False))

    import subprocess

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "1024x768", "-r", "25", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_path],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    for f in frames:
        proc.stdin.write(f.astype(np.uint8, copy=False).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"[preview] wrote {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", nargs="?", const="/tmp/opencode/avatar_preview.mp4")
    args = parser.parse_args()
    if args.preview:
        asyncio.run(_preview(args.preview))
    else:
        print("avatar module imported OK (engine:", build_engine(1024, 768, 25).name, ")")