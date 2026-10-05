"""Background helper: download the faster-whisper STT model with retries.

The heavy HF downloads on this network stall, so we download the model
in a separate process with resume + retry until it is complete, then
exit.  realtime/agent.py enables voice input only once the model files
are present locally.

Model repo is taken from STT_MODEL (default Systran/faster-whisper-base).
"""
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

REPO_ID = os.getenv("STT_MODEL", "Systran/faster-whisper-base")


def is_complete(cache_root: str) -> bool:
    for root, _dirs, files in os.walk(cache_root):
        for name in files:
            if name.endswith(".incomplete"):
                return False
    return True


def main():
    import huggingface_hub

    cache_root = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub",
        "models--" + REPO_ID.replace("/", "--"),
    )
    attempt = 0
    while True:
        attempt += 1
        try:
            print(f"[stt] attempt {attempt}: snapshot_download({REPO_ID})", flush=True)
            huggingface_hub.snapshot_download(repo_id=REPO_ID)
            if os.path.isdir(cache_root) and is_complete(cache_root):
                print(f"[stt] download complete: {cache_root}", flush=True)
                return 0
            print("[stt] snapshot returned but files incomplete; retrying", flush=True)
        except Exception as exc:
            print(f"[stt] attempt {attempt} failed: {exc!r}", flush=True)
        print("[stt] retrying in 15s...", flush=True)
        time.sleep(15)


if __name__ == "__main__":
    sys.exit(main())