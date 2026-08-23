# Avatar adapter (GPU machine only)

LiveTalking + MuseTalk provide realtime lip-synced video from TTS audio.

- Requires NVIDIA GPU (MuseTalk realtime reported ~42 FPS on RTX 3080 Ti,
  ~72 FPS on RTX 4090; weaker GPUs fall back to lighter models or lower FPS).
- Runs as a separate service: receives answer audio, returns talking-avatar
  video stream into the LiveKit room.
- Pretrained weights are downloaded at setup and never committed to git.

On CPU-only machines the UI shows a static avatar placeholder instead.
