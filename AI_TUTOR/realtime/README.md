# Realtime layer (Phase 2+)

Pipecat orchestration around the unchanged tutor engine:

```text
mic -> faster-whisper STT -> ask_tutor() -> TTS -> LiveKit room
camera -> vision model (pretrained VLM) -/        |
                                        avatar adapter
```

Planned modules:

- `pipecat_pipeline.py` — LiveKit transport + STT/LLM/TTS frame pipeline
- `stt.py` — faster-whisper wrapper (CPU-friendly, CTranslate2)
- `tts.py` — pretrained open-source TTS (CPU-capable first)
- `vision.py` — VisionProvider interface (local VLM on GPU machine, API model acceptable on CPU)

No model training anywhere in this stack.
