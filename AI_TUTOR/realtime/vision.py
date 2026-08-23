"""Milestone 7: vision / visual perception.

Captures webcam frames from the browser and produces visual context for the
tutor. Without a GPU, the vision analysis is a lightweight placeholder that
describes presence and basic scene info. Later this plugs into a real VLM
(Qwen-VL etc.) on the GPU machine.
"""
import os
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

VISION_INTERVAL = float(os.getenv("VISION_INTERVAL_SECONDS", "2"))

_latest_frame: dict = {
    "timestamp": 0.0,
    "description": "",
    "confidence": 0.0,
    "raw_b64": None,
}


def update_frame(description: str, confidence: float = 0.8, raw_b64: Optional[str] = None):
    _latest_frame["timestamp"] = time.time()
    _latest_frame["description"] = description
    _latest_frame["confidence"] = confidence
    if raw_b64:
        _latest_frame["raw_b64"] = raw_b64


def get_visual_context() -> dict:
    age = time.time() - _latest_frame["timestamp"]
    if age > VISION_INTERVAL * 10 or not _latest_frame["description"]:
        return {"available": False}
    return {
        "available": True,
        "description": _latest_frame["description"],
        "confidence": _latest_frame["confidence"],
        "age_seconds": round(age, 1),
    }


def describe_placeholder() -> str:
    """Simple heuristic description when no VLM is available."""
    ctx = get_visual_context()
    if not ctx["available"]:
        return ""
    return (
        f"[Visual context: {ctx['description']} "
        f"(confidence {ctx['confidence']:.0%}, {ctx['age_seconds']}s ago)]"
    )
