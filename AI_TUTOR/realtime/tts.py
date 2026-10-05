"""Milestone 4: text-to-speech via a provider abstraction.

TTS_PROVIDER selects the implementation:
  sapi   - Windows built-in voices (default, zero download, CPU)
  piper  - open-source Piper neural voice (offline, pip install piper-tts)

The interface mirrors the spec (synthesize / stream) so implementations can
be swapped without touching the tutor or frontend.
"""
import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "piper")
TTS_VOICE = os.getenv("TTS_VOICE", "")


class TTSProvider:
    async def synthesize(self, text: str) -> bytes:
        raise NotImplementedError

    def stop(self):
        pass


class SapiProvider(TTSProvider):
    """Windows built-in offline voice via System.Speech (PowerShell bridge).
    Chosen over pyttsx3/COM streams after both proved unreliable in server
    threads; this path is fully synchronous and battle-tested here."""

    def __init__(self, voice: str = "", rate: int = 0):
        self._voice = voice
        self._rate = rate

    async def synthesize(self, text: str, rate: int = 0) -> bytes:
        import asyncio
        import subprocess

        def render() -> bytes:
            import json

            txt_fd, txt_path = tempfile.mkstemp(suffix=".txt")
            wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(wav_fd)
            try:
                with os.fdopen(txt_fd, "w", encoding="utf-8") as f:
                    f.write(text)
                cfg = json.dumps(
                    {"txt": txt_path, "wav": wav_path, "rate": rate or self._rate}
                ).replace("'", "''")
                script = (
                    "$params = '" + cfg + "' | ConvertFrom-Json\n"
                    "Add-Type -AssemblyName System.Speech\n"
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
                    "if ($params.rate -ne 0) { $s.Rate = [int]$params.rate }\n"
                    "$s.SetOutputToWaveFile($params.wav)\n"
                    "$s.Speak((Get-Content -Raw -Encoding UTF8 $params.txt))\n"
                    "$s.Dispose()\n"
                )
                proc = subprocess.run(
                    [
                        "powershell", "-NoProfile", "-ExecutionPolicy",
                        "Bypass", "-Command", script,
                    ],
                    capture_output=True,
                    timeout=120,
                )
                if proc.returncode != 0:
                    raise RuntimeError(proc.stderr.decode(errors="replace")[-500:])
                with open(wav_path, "rb") as f:
                    return f.read()
            finally:
                for p in (txt_path, wav_path):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

        return await asyncio.to_thread(render)


class PiperProvider(TTSProvider):
    """Open-source Piper neural TTS via onnxruntime (offline, cross-platform).

    Model (voice .onnx + .onnx.json sidecar) lives under models/piper by
    default; override the path with PIPER_MODEL. Piper is the Linux path in
    place of the Windows-only SapiProvider.
    """

    def __init__(self, voice: str = "", rate: int = 175, model: str = ""):
        self._voice = voice
        self._rate = rate
        default_model = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "piper", "en_US-lessac-medium.onnx",
        )
        self._model_path = model or os.getenv("PIPER_MODEL", default_model)
        self._voice_model = None

    def _load(self):
        if self._voice_model is None:
            from piper import PiperVoice
            if not os.path.exists(self._model_path):
                raise RuntimeError(
                    f"Piper model not found at {self._model_path}. "
                    "Download en_US-lessac-medium.onnx + .onnx.json into models/piper/"
                )
            self._voice_model = PiperVoice.load(self._model_path)
        return self._voice_model

    async def synthesize(self, text: str, rate: int = 0) -> bytes:
        import asyncio
        import io
        import wave

        def render() -> bytes:
            voice = self._load()
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                voice.synthesize_wav(text, wf)
            return buf.getvalue()

        return await asyncio.to_thread(render)


_PROVIDERS = {
    "sapi": SapiProvider,
    "piper": PiperProvider,
}


def get_tts_provider() -> TTSProvider:
    cls = _PROVIDERS.get(TTS_PROVIDER, SapiProvider)
    return cls(voice=TTS_VOICE)


async def synthesize(text: str) -> bytes:
    return await get_tts_provider().synthesize(text)
