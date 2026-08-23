"""Milestone 4: text-to-speech via a provider abstraction.

TTS_PROVIDER selects the implementation:
  sapi   - Windows built-in voices (default, zero download, CPU)
  piper  - reserved for the open-source Piper neural voice on GPU laptop

The interface mirrors the spec (synthesize / stream) so implementations can
be swapped without touching the tutor or frontend.
"""
import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "sapi")
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
    """Placeholder for the open-source Piper neural TTS (future)."""

    async def synthesize(self, text: str) -> bytes:
        raise NotImplementedError(
            "Piper provider lands with the GPU-machine phase"
        )


_PROVIDERS = {
    "sapi": SapiProvider,
    "piper": PiperProvider,
}


def get_tts_provider() -> TTSProvider:
    cls = _PROVIDERS.get(TTS_PROVIDER, SapiProvider)
    return cls(voice=TTS_VOICE)


async def synthesize(text: str) -> bytes:
    return await get_tts_provider().synthesize(text)
