# AI Tutor — A Real-Time Voice Tutor with a Neural Lip-Synced Teacher

## Personal Engineering Write-Up

**Author:** Abdur Rahaman Siddique (He/Him)
**Ownership:** This is entirely my project — designed, built, and operated by me.

**Scope** — AI_TUTOR, branch `musetalk` (voice + MuseTalk avatar), base branch `multimodal`
**Surface** — live voice tutoring in the browser, text chat, document-grounded Q&A, screen recording
**Stack** — Pipecat 1.11.0, LiveKit WebRTC, faster-whisper (STT), Piper (TTS), MuseTalk (CUDA fp16 lip-sync), Neo4j + FAISS + Groq (`openai/gpt-oss-120b`), FastAPI, React/Vite, Streamlit

This is a personal account of what I built, the problems I hit, and the reasoning behind the design decisions. It is written for a reader who wants to understand the engineering, not the marketing. Where a number is something I measured, I say so; where it is a target or an estimate, I say that too.

---

## 1. The problem I set out to solve

"Build an AI tutor you can talk to" sounds like a weekend of API calls until you look at what it actually requires. A single spoken question has to travel microphone → WebRTC → voice-activity detection → speech-to-text → knowledge-grounded LLM → text-to-speech → a lip-synced face → back to the browser, all inside a few seconds, while a knowledge graph and a 528-document FAISS index keep the answer honest instead of letting the model improvise.

The naive version of this is five independent services glued with HTTP, and it fails in five independent ways. My decision was one continuous frame pipeline with strict stage ownership: audio enters as raw PCM frames, every processor consumes, transforms, or passes frames, and the avatar is just the last processor before the transport — not a separate rendering daemon. Pipecat provides that pipeline; almost all of my engineering went into what sits between its processors and into the failures at the seams.

The second decision was that the text path and the voice path must share one brain. The browser chat hits `POST /api/chat` and the voice agent asks in-process, but both call the same `ask_tutor()` — Neo4j knowledge-graph hops + FAISS hybrid retrieval + Groq. A tutor that gives different answers depending on whether you type or speak is not a tutor, it is two demos.

Key commits on `musetalk`: `cfc1fe1` (Phase 2: Meet-style room, Piper TTS, voice-pipeline fixes), `072e91e` (MuseTalk engine + Record button), `7cb95d5` (voice-pipeline fixes), `90b827f` (README). The `multimodal` branch is unchanged at `cfc1fe1`.

---

## 2. Architecture: how a question actually flows

This is the real path, in order, as the code runs it.

```text
Browser mic ──WebRTC──► LiveKit room "tutor-room" (:7880)
  │
  ▼
transport.input()                    LiveKitInputTransport
  │   UserAudioRawFrame (48 kHz)
  ▼
VADProcessor (Silero)                gates speech vs silence
  │   confidence ≥ 0.6 AND volume ≥ 0.1 → STARTING → SPEAKING
  ▼
WhisperSTTService                    faster-whisper base, int8, CPU
  │   buffers audio between VAD start/stop → TranscriptionFrame
  ▼
TutorProcessor                       RAG: FAISS (528 docs) + KG (Neo4j)
  │   ask_tutor() → Groq openai/gpt-oss-120b → TTSSpeakFrame(answer)
  ▼
TTSProcessor                         Piper en_US-lessac-medium, 22050 Hz PCM
  │   OutputAudioRawFrame
  ▼
AvatarProcessor                      MuseTalkEngine (GPU) or PuppetEngine
  │   OutputImageRawFrame (1024×768 RGB) + paced audio slices @ 25 fps
  ▼
transport.output() ──WebRTC──► browser (teacher video + audio)

Greeting: first participant joins → +1 s → TTSSpeakFrame("Hello! I'm your
tutor. What would you like to learn today?")

Text path (independent): :5173 → POST :8000/api/chat → ask_tutor() → answer
```

Structural decisions worth calling out:

**One pipeline, two entry modes for the answer.** `TutorProcessor` first tries the LLM in-process (fast path, no HTTP hop). If that throws, it falls back to the backend `/api/chat` endpoint. Two paths, one prompt contract — the fallback exists because a dead backend should degrade voice answers, not kill the session.

**The greeting is deferred, not immediate.** It fires one second after the first participant joins, as a task, so a student who joins mid-session never triggers a second greeting (participants are tracked; re-joins do not re-greet).

**Turn-taking is task-cancellation, not a queue.** Each transcription cancels the previous in-flight `_answer` task before starting a new one. If you say "explain recursion" then immediately "actually, binary search", the tutor answers the last question, not both overlapping. This is deliberate: interleaved speech from a tutor is worse than a dropped answer.

**Debuggability is a feature.** Every stage prints a marker (`[vad]`, `[tts]`, `[tutor] Q → A`, `[avatar] emit ... engine=musetalk`, `[audio-out]`). A dead session is diagnosable from the log alone. Section 7 is where those markers paid for themselves.

---

## 3. The two-gate voice-activity problem

The VAD is where the project nearly stayed broken, and the bug was invisible from the browser — the mic was clearly working (the user could use it elsewhere), audio reached the agent, and yet nothing ever happened.

Silero VAD in pipecat has two gates:

```python
speaking = confidence >= params.confidence and volume >= params.min_volume
```

pipecat's stock `VAD_MIN_VOLUME = 0.6`. That threshold is on a BS.1770 loudness measure over a 400 ms window — near-shouting. Real microphone speech through Chrome's processing chain sits around 0.05–0.3. Injected full-scale test audio (a Piper-rendered question WAV fed via `--use-file-for-fake-audio-capture`) sails past 0.6, which is exactly why my automated end-to-end tests passed while every human session failed. The instrument was calibrated to the test, not the user.

I set `VADParams(confidence=0.6, min_volume=0.1)` in `realtime/agent.py` and added a throttled debug line around `_run_analyzer` that prints `[vad] vol=… conf=… state=…` — because the next failure taught me that thresholds without observability are just superstition.

That next failure was mine: the debug line formatted the volume as `:.3f`, but pipecat returns it as a numpy array. `unsupported format string passed to numpy.ndarray.__format__` threw on **every audio chunk** — 4,232 exceptions logged — VAD died completely, and the error flood destabilised the whole pipeline (including the video, section 4.3). The fix was a `_f()` coercion helper. The lesson was older and blunter: instrumentation must never be able to take down the path it observes.

**Verified:** with the lowered gates and the safe debug line, live sessions show `[vad] vol=0.284…0.604 conf=… state=SPEAKING` on real speech, transcriptions reach the tutor, and answers come back. The user confirmed the voice loop working end-to-end after `7cb95d5`.

---

## 4. Speech-to-text: readiness is not "did the download finish"

### 4.1 The stale-marker bug

`_stt_ready()` checked for the `models--Systran--faster-whisper-base` cache directory and the absence of `*.incomplete` markers. One session, the model directory existed, `model.bin` was **complete** (145,217,532 bytes, sha256 verified against the hub), and yet the agent started in "lecture mode" — no STT at all — because a zero-byte `blobs/*.incomplete` marker from an interrupted download was still lying on disk. I deleted the marker; Whisper has loaded in ~1.6 s ever since.

The generalisation I took from this: readiness checks must verify the artefact, not the debris of the process that fetched it.

### 4.2 Configuration is not a Settings dump

My first `WhisperSTTService` construction crashed with `WhisperSTTSettings.__init__() got an unexpected keyword 'device'`. In pipecat 1.11, `device` and `compute_type` are constructor kwargs; only the model name belongs in `WhisperSTTService.Settings(...)`. The line that shipped:

```python
WhisperSTTService(
    device=WHISPER_DEVICE,
    compute_type="int8",
    settings=WhisperSTTService.Settings(model=WHISPER_MODEL),
    audio_passthrough=False,   # see 4.3 — this one matters
)
```

### 4.3 `audio_passthrough` — the bug that blacked out the teacher

This is the bug I would most want reviewed, because the symptom (a totally black teacher screen, no face) pointed nowhere near the cause.

pipecat's `STTService.__init__` defaults to `audio_passthrough=True`: after buffering a `UserAudioRawFrame` for transcription, it **forwards the same frame downstream**. In my pipeline "downstream" is `Tutor → TTS → Avatar`. The avatar's `process_frame` treats any `AudioRawFrame` as a new utterance and does two things: cancels its current render task, then starts a new one. So every ~68 ms mic frame (3300 samples @ 48 kHz) cancelled the teacher's in-progress render. Greeting video: cancelled after one frame. Answer video: cancelled after one frame. What rendered instead was a flood of one-frame micro-emissions — 419 of them in one session, ceasing only when the participant left the room, which was the tell that finally tied the flood to the mic track.

The log made it unambiguous:

```text
[avatar] emit: audio=3300 samples, video_frames=1 engine=musetalk   ×419
```

The fix is one keyword — `audio_passthrough=False` — because nothing downstream of STT legitimately needs raw microphone audio; the avatar must only ever see TTS output. The deeper lesson: **default flags in a framework are part of your architecture whether you chose them or not.** I had silently accepted a default that wired the microphone into the video renderer.

---

## 5. The avatar: MuseTalk, honestly integrated

### 5.1 Two engines behind one interface

`realtime/avatar.py` defines a `LipsyncEngine` contract (`is_ready`, `warmup`, `render`, `render_stream`, `expected_frames`) with two implementations:

- **`MuseTalkEngine`** — GPU neural lip-sync (the real thing), lazy-loaded via `asyncio.to_thread` so agent startup is never blocked.
- **`PuppetEngine`** — CPU fallback: a 2D amplitude-driven mouth overlay on the portrait. Ugly, but a tutor that stops teaching because CUDA failed is worse than one that looks cheap.

`build_engine("auto")` prefers MuseTalk only when all six weight files exist. If MuseTalk init throws (no GPU, missing weights, dtype explosion), the engine caches the error and the pipeline falls back. The avatar is the only GPU-dependent stage; everything else runs on CPU.

### 5.2 Why I vendored the core instead of installing MuseTalk

The upstream repo's requirements pin numpy 1.23, mmpose, face_detection, librosa/numba — a stack that does not install on Python 3.14, and half of which I don't need (their preprocessing.py assumes pre-cropped face landmarks; my portrait bounding box is fixed). So `realtime/musetalk_core.py` is a ~vendored inference path: `AutoencoderKL` (VAE) + `UNet2DConditionModel` + Whisper encoder audio features, wired directly.

Two bugs here were non-obvious:

**UNet construction.** The shipped `musetalk.json` is a diffusers *config dump*, including `_class_name` and `_diffusers_version`. Passing it straight to `UNet2DConditionModel(**json)` is a `TypeError`. Filtering keys that start with `_` is the fix, and it is exactly the kind of thing that looks like a one-liner until you stare at the traceback.

**fp16 promotion.** My positional-encoding buffer was cast to the right *device* but not the right *dtype*: fp16 activations meeting an fp32 buffer promote the whole matmul to fp32 → `expected mat1 and mat2 to have the same dtype, but got: float != c10::Half`. Fixed with `.to(device=x.device, dtype=x.dtype)`. Silent correctness-adjacent failure, loud crash — the best kind of bug, in retrospect.

### 5.3 Audio features and the streaming renderer

MuseTalk wants 50 Hz audio features and produces 25 fps video. `audio_chunks()` ports the upstream recipe: 30 s segments → Whisper encoder hidden states → stack → trim to `floor(seconds × 50)` → pad ±2 rows per edge → reshape `(T, 50, 384)` so every video frame `i` reads rows `floor(i×10) … +10`. The engine's `expected_frames()` uses `floor(dur × fps)` — **it must agree with the audio chunk count exactly**, or video and audio drift apart frame by frame.

Generation runs ~14.9 fps on the RTX 4060 (batched UNet denoise at t=0 + VAE decode, 8 frames per batch). The output is *paced*, not blocked: `render_stream()` yields frames as they complete and `AvatarProcessor` emits each with its audio slice on a 25 fps schedule, so downstream jitter smooths out. Measured offline: **178/178 frames for 7.14 s of speech, frame-motion index 0.70** (i.e. the mouth visibly moves — not a static image), preview MP4 written.

### 5.4 The seam, the vignette, and the mask

Two portrait/blend bugs were found by *looking at the output*, which is why the offline harness saves first/mid/last PNGs:

- **Visible rectangle seam** around the face patch. Fixed by strengthening the feather mask: sides 12% of width, bottom 4% of height, Gaussian σ = h/8.
- **The original `teacher.png` was a black dome.** `make_teacher_avatar.py`'s vignette painted growing discs in L-mode (values >255 wrap/clamp) centred at the top of the head, then *inverted* the mask — painting the entire face region solid black. Rewritten to paint largest-first so smaller discs overwrite the centre, brightness `255 → 0` outside r=360, clamped, composited normally. The regenerated portrait has a proper face, and that face is what the MuseTalk crop box (`MUSETALK_BBOX=317,80,707,585`) is calibrated to.

### 5.5 Streaming emission and the never-cut-the-answer rule

`_emit_utterance` computes video timestamps `pts[i] = round(n_audio × i / n_vid)`, streams engine frames interleaved with `OutputAudioRawFrame(pcm[2·pts[i] : 2·pts[i+1]])`, sleeps `1/fps` between frames, and stops if the speaker is interrupted. The part I deliberately got right: **after the loop, any audio beyond the last video frame is flushed as one final frame.** If the renderer produced one frame fewer than the audio needed, the tail of the answer still plays. Cutting off a student mid-sentence is the one failure mode a tutor cannot have.

---

## 6. The crash that ate the session: `free(): invalid pointer`

The nastiest failure in this project had no Python traceback — the agent process aborted with a glibc `free(): invalid pointer` (SIGABRT), repeatedly, sometimes at startup.

faulthandler (enabled via `PYTHONFAULTHANDLER=1` in the keep-alive script) caught the thread inside `transformers/integrations/deepgemm.py` → `torch/_dynamo` → `torch/utils/_triton.py has_triton_package` → `import triton` → `dlopen("libtriton.so")` → abort. The reproducibility pattern was maddening: standalone `import triton` passed; Whisper + index-load repro passed; deepgemm-in-thread passed — only the full agent context crashed, and only when the RAG index preload ran inside `asyncio.to_thread`.

**Fix:** `import triton` as **line 1 of `realtime/agent.py`**, before torch, transformers, or anything else can touch it. Loading libtriton early, in a clean state, means the later deepgemm chain hits `sys.modules` instead of re-entering the dynamic loader. Agent survived; `Voice tutor RAG index loaded: 528 docs`; zero Fatal errors after.

Related hardening: `_load_index()` is guarded by a module-level `threading.Lock` with a re-check after acquisition — the faulthandler dumps had shown *two* threads inside `importlib._bootstrap` concurrently loading the index, because my early-return only handled the `"loaded"` state, not the in-progress one.

---

## 7. Two "exactly 300 seconds" bugs (and why they look identical)

Both failures killed live sessions at precisely the five-minute mark, and they are easy to confuse:

1. **Pipecat idle timeout.** `PipelineTask` defaults to `idle_timeout_secs=300`, and the idle counter only resets on *speech-related frames* (`BotSpeakingFrame`, `InterimTranscriptionFrame`, `UserStartedSpeakingFrame`, …). A student who joins, listens to the greeting, and thinks for five minutes produces none of them → `Idle timeout detected → cancelling pipeline worker → Disconnected`. Fix: `idle_timeout_secs=None` at the task constructor (the worker guards `if self._idle_timeout_secs:` — `None` cleanly disables it).

2. **LiveKit room empty-timeout.** The agent reconnected and then dropped at *exactly* +300 s again — with no idle log. The LiveKit server log showed `room created with "emptyTimeout": 300` and `closing room reason: IDLE_TIMEOUT`. The cause was a wrong environment variable name: I had set `LIVEKIT_EMPTY_TIMEOUT` (silently ignored), while the real variable is **`LIVEKIT_ROOM_EMPTY_TIMEOUT`**. LiveKit also does not count the agent as an occupant, so an agent-only room is "empty" forever. Fixed by restarting the server with `LIVEKIT_ROOM_EMPTY_TIMEOUT=999999999` and `LIVEKIT_ROOM_DEPARTURE_TIMEOUT=999999999`, verified from the server's own room-update log.

The operational wrinkle that made both worse: the keep-alive loop only restarts the agent **when the process exits**. A zombie (alive, but disconnected or doomed) sits there forever. Revival is `pkill -f "realtime.ag[e]nt"` — brackets included on purpose, because a bare `pkill -f realtime.agent` matches the *issuer's own shell command line* and kills it. I hit that self-match twice before encoding it as a rule.

---

## 8. Frame semantics: one line that silently kills a pipeline

A debugging session that cost hours and now lives as a comment: `FrameProcessor.process_frame` **must call `await super().process_frame(...)` first**. Skipping it means the private `__process_queue` never initialises, and the first non-`SystemFrame` silently kills that processor's input loop — frames arrive, nothing processes, no exception surfaces. My test harness had this bug; the agent did not. When a frame "does nothing", check for the missing `super()` call before doubting the frame's content.

---

## 9. The frontend: Meet layout, quality ramp, and recording

**Layout.** I replaced LiveKit's stock `<VideoConference/>` with a custom `MeetLayout`: one big remote `VideoTrack` (object-fit: contain, 1002×1002 stage), a mirrored local PiP bottom-right with a "You" chip, a bottom control bar (mic/camera/leave only), and a Record pill top-right. A `useEffect` on non-local publications requests `setVideoQuality(VideoQuality.HIGH)` so LiveKit's bandwidth ramp converges fast.

**Quality ramp — measured, not assumed.** At ~9 s post-join the main video decodes at 240×180; it reaches a stable **1024×768** shortly after. That is LiveKit's normal up-leveling behaviour, not a bug — worth stating because "the teacher is blurry" looks like a defect for the first ten seconds of every session.

**Recording.** The Record button captures the screen (`getDisplayMedia({video:{frameRate:25}, audio:false})`), mixes the teacher's remote audio track and the local mic through an `AudioContext → MediaStreamDestination`, and records `vp8/opus` via `MediaRecorder`, auto-downloading `lesson_<timestamp>.webm`. The video path is verified (59 MB file, VP8 800×450, plays fine). **The audio path is not: the mixed track comes out silent** — the source wiring needs work, and this is filed in README's Known Issues rather than quietly claimed as working.

**Known-issues discipline.** When the user asked "why does my recording have no sound", I ffprobed the actual downloaded file instead of guessing: opus stream present, `volumedetect` fails to parse packets, duration `N/A` — the stream exists but carries nothing useful. The honest status (video ✅, audio ❌) is in the README. A write-up that pretends otherwise is worthless.

---

## 10. Operations: running this outside a notebook

Nothing here is glamorous; all of it bit me once:

- **`nohup … &` does not survive tool timeouts** — children die with the process group. Every service starts under `setsid bash -c '… exec CMD > log 2>&1' &`.
- **Streamlit dies silently after the welcome banner** without `--server.headless true` (plus `--server.fileWatcherType none`).
- **Backend:** `python -m uvicorn backend.main:app --port 8000`; **LiveKit:** `./tools/livekit/livekit-server --dev --bind 0.0.0.0` with the two room-timeout env vars; **frontend:** `npm run dev` (:5173, proxies `/api` → :8000).
- **Keep-alive:** `/tmp/opencode/run_agent.sh` loops the agent with `PYTHONFAULTHANDLER=1` and `OMP/MKL/OPENBLAS_NUM_THREADS=2`, appending to one log; `=== agent start/exit ===` timestamps make restart archaeology possible.
- **Model downloads:** the 3.4 GB `unet.pth` stalled repeatedly through the HF CLI (stale `.incomplete` debris, misleading `--include` warnings). The fix was blunt: parallel `curl -C - --retry 30 --retry-all-errors` against `resolve/main` URLs, then **verify exact byte sizes** (the table in README). Resume-capable curl beat the clever tool.

---

## 11. Technical challenges, and how I approached each

- **Works in the test, dead for humans.** Symptom: full E2E passes with an injected WAV; every real session produces no answer. Approach: instrument the actual gate (`[vad] vol/conf`) instead of trusting the passing test. Outcome: `min_volume=0.6` (a near-shout threshold) was rejecting normal speech; tests had passed because Piper output is near full-scale. Calibrate against reality, not fixtures.
- **Black teacher screen while a student is present.** Symptom: avatar emits 419 × one-frame utterances, all correlated with the participant's presence. Approach: trace frame types upstream, read `STTService.__init__`'s defaults. Outcome: `audio_passthrough=True` had wired the mic into the avatar's utterance handler; one keyword disconnects it.
- **Hard native abort, no traceback.** Symptom: `free(): invalid pointer` under faulthandler → `import triton` inside a worker thread. Approach: narrow via standalone repros (which all passed), accept that context matters, force early `dlopen`. Outcome: line-1 `import triton`; stable ever since.
- **Instrumentation that became the outage.** Symptom: 4,232 `numpy.ndarray.__format__` exceptions, one per audio chunk. Approach: the traceback pointed at my own line; fixed by coercion helper. Outcome: debug prints are now defensive by default.
- **Sessions dying at exactly 300 s (twice, different causes).** Approach: read *server* logs, not just agent logs — the LiveKit `emptyTimeout: 300` line named the second bug. Outcome: `idle_timeout_secs=None` + correct `LIVEKIT_ROOM_*` env names, both documented next to the code.
- **MuseTalk would not install on Python 3.14.** Approach: don't fight the dependency tree — vendor the three-model inference core (VAE/UNet/Whisper), skip mmpose/librosa entirely. Outcome: ~1 file of dependency surface instead of a broken environment, plus a puppet fallback if it ever fails at runtime.
- **fp16/fp32 matmul mismatch.** Approach: treat dtype as part of the device contract for every buffer. Outcome: one-line fix, comment preserved.
- **Face-patch seam and a black-dome portrait.** Approach: look at saved frames (the offline harness exists for exactly this), then strengthen the feather and rewrite the vignette paint order. Outcome: soft blend, usable face, bbox calibrated to it.

---

## 12. What I verified vs. what I am reporting

I want to be precise, because it matters when you decide what to trust.

**Verified by running it end-to-end:**
- Full voice loop with injected speech: VAD fires → Whisper transcribes (`[ Hello ]`, `[ What is the hash map… ]`) → tutor answers (e.g. 296- and 903-char RAG answers logged) → Piper pushes audio → MuseTalk emits (`video_frames=92 engine=musetalk`) → browser receives audio (**5,961 inbound packets, 0 lost, jitter 0**).
- MuseTalk offline harness: 178/178 frames for 7.14 s, motion index 0.70, preview MP4, ~14.9 fps generation paced to 25.
- Live screenshot via headless-Chrome CDP: teacher face on the big screen, self PiP, Record pill, control bar; stream settles at 1024×768.
- Weights: all seven files verified against exact expected byte sizes (3.4 GB UNet, etc.).

**Read from the code** (not runtime-verified claims): frame-flow semantics, VAD gate logic, `audio_passthrough` default, engine selection/fallback, pts pacing and the tail-audio flush, prompt/CoT configuration.

**Known broken / not verified:**
- Recording **audio** is silent (video works) — confirmed on the actual output file.
- No automated test suite exists. Verification has been log-markers + offline harnesses + screenshots.
- MuseTalk lower-face quality is hazy (inherent to the model), and generation at ~15 fps means the first answer has 2–4 s of lead-in silence while the stream buffers.

---

## 13. Honest limitations and what I would fix next

- **Recording audio.** The AudioContext mix of remote+local tracks into `MediaRecorder` needs a real fix (likely per-source gain monitoring and/or capturing the tab's audio stream directly). Until then the README says video-only.
- **No tests, no CI.** The pure functions — pts pacing, `audio_chunks` frame math, engine selection, RAG call contract — are exactly where a small pytest suite would lock in behaviour. I learned from VisionTouch that baselines without committed artefacts are guesswork; same truth applies here.
- **Operations are hand-rolled.** The keep-alive shell loop, bracket-pkill rule, and manual env-var prefixes are tribal knowledge encoded in this write-up rather than in systemd units or a compose file.
- **STT/VAD tuning is global.** One `VADParams` for every microphone. Per-device calibration (or an adaptive noise floor) would handle quiet headsets better than my `min_volume=0.1`.
- **Latency budget is not measured end-to-end.** I know stage-level facts (Whisper loads in 1.6 s, MuseTalk renders ~15 fps, answers arrive in 2–4 s) but I have never logged a clean question→first-audio percentile. That number belongs in the README.
- **Subtitles for the teacher's answer** — designed (data-channel push of the `TTSSpeakFrame` text, DOM caption overlay) but not built yet.
- **Puppet fallback is a placeholder**, not an experience. It keeps the tutor teaching when CUDA fails; it does not look like a person.

---

## 14. Closing

The headline is not that an LLM answers questions — that part is a Groq call. The headline is that a microphone, a knowledge graph, a neural lip-sync model, and a browser share one frame pipeline that fails in *defined* ways: fallback to puppet instead of silence, flush trailing audio instead of cutting the student off, lock the index load instead of double-loading, and print a marker at every stage so a dead session is a two-minute diagnosis instead of an evening.

If I were picking this up again with more time, the order would be: fix recording audio and write the test suite first (both are bounded, both protect everything else); then end-to-end latency instrumentation; then systemd/compose to retire the shell keep-alive. The modelling work — RAG, KG, Groq prompting — was the part I understood best going in. The realtime plumbing and the native-crash archaeology were where the real lessons were.
