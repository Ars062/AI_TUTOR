import { useEffect, useRef, useState } from "react";
import {
  LiveKitRoom,
  RoomAudioRenderer,
  ControlBar,
  useTracks,
  VideoTrack,
} from "@livekit/components-react";
import { Track, VideoQuality } from "livekit-client";
import "@livekit/components-styles";

const GREETING = "Hello! What would you like to learn today?";

function cleanForSpeech(text) {
  return text.replace(/[*#`>|_]/g, " ").replace(/\s+/g, " ").trim();
}

function UploadPanel() {
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const inputRef = useRef(null);

  async function handleUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setResult(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("/api/upload", { method: "POST", body: form });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setResult({ error: String(err) });
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <div>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.txt,.md"
        onChange={handleUpload}
        style={{ display: "none" }}
      />
      <button
        className="upload-btn"
        onClick={() => inputRef.current?.click()}
        disabled={uploading}
      >
        {uploading ? "Uploading…" : "📄 Upload PDF / TXT"}
      </button>
      {result && (
        <p className="note">
          {result.error
            ? `Error: ${result.error}`
            : `✅ ${result.chunks} chunks added (${result.total_docs} total)`}
        </p>
      )}
    </div>
  );
}

function MeetLayout() {
  const tracks = useTracks([{ source: Track.Source.Camera, withPlaceholder: true }]);
  const aTracks = useTracks([{ source: Track.Source.Microphone }]);
  const recRef = useRef(null);
  const [recording, setRecording] = useState(false);

  useEffect(() => {
    for (const t of tracks) {
      const pub = t.publication;
      if (
        !t.participant.isLocal &&
        pub &&
        typeof pub.setVideoQuality === "function"
      ) {
        try {
          pub.setVideoQuality(VideoQuality.HIGH);
        } catch {
          /* publication may not be subscribed yet */
        }
      }
    }
  }, [tracks]);

  const teacher = tracks.find(
    (t) => !t.participant.isLocal && !t.placeholder
  );
  const local = tracks.find(
    (t) => t.participant.isLocal && !t.placeholder
  );
  const localVisible = local && !local.publication?.isMuted;

  const teacherAudio = aTracks.find(
    (t) => !t.participant.isLocal && !t.placeholder
  );
  const localMic = aTracks.find(
    (t) => t.participant.isLocal && !t.placeholder
  );

  async function startRecording() {
    if (!teacher || recRef.current) return;
    let display;
    try {
      display = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: 25 },
        audio: false,
      });
    } catch (e) {
      console.error("record: screen share cancelled/failed", e);
      return;
    }
    try {
      const ctx = new AudioContext();
      await ctx.resume();
      const dest = ctx.createMediaStreamDestination();
      const add = (msTrack) => {
        try {
          if (msTrack && msTrack.readyState === "live")
            ctx.createMediaStreamSource(new MediaStream([msTrack])).connect(dest);
        } catch (e) {
          console.error("record: audio source failed", e);
        }
      };
      add(teacherAudio?.publication?.track?.mediaStreamTrack);
      add(localMic?.publication?.track?.mediaStreamTrack);
      const stream = new MediaStream([
        ...display.getVideoTracks(),
        ...dest.stream.getAudioTracks(),
      ]);
      const rec = new MediaRecorder(stream, {
        mimeType: "video/webm;codecs=vp8,opus",
        videoBitsPerSecond: 2500000,
      });
      const chunks = [];
      rec.ondataavailable = (e) => {
        if (e.data && e.data.size) chunks.push(e.data);
      };
      rec.onstop = () => {
        try {
          const blob = new Blob(chunks, { type: "video/webm" });
          const a = document.createElement("a");
          a.href = URL.createObjectURL(blob);
          const ts = new Date()
            .toISOString()
            .slice(0, 19)
            .replace(/[T:]/g, "-");
          a.download = `lesson_${ts}.webm`;
          a.click();
          setTimeout(() => URL.revokeObjectURL(a.href), 15000);
        } catch (e) {
          console.error("record: save failed", e);
        }
        display.getTracks().forEach((t) => t.stop());
        ctx.close().catch(() => {});
        setRecording(false);
      };
      rec.start(1000);
      recRef.current = rec;
      setRecording(true);
    } catch (e) {
      console.error("record: start failed", e);
      display.getTracks().forEach((t) => t.stop());
    }
  }

  function stopRecording() {
    try {
      recRef.current?.stop();
    } catch (e) {
      console.error("record: stop failed", e);
    }
    recRef.current = null;
  }

  useEffect(() => {
    return () => {
      try {
        if (recRef.current && recRef.current.state !== "inactive")
          recRef.current.stop();
      } catch {
        /* page unloading */
      }
    };
  }, []);

  return (
    <div className="meet-stage">
      <div className="meet-main">
        {teacher ? (
          <VideoTrack trackRef={teacher} className="meet-main-video" />
        ) : (
          <div className="meet-waiting">
            <div className="avatar-face big">🧑‍🏫</div>
            <p>AI Tutor is connecting…</p>
          </div>
        )}
      </div>

      <div className="meet-pip">
        {localVisible ? (
          <VideoTrack
            trackRef={local}
            className="meet-pip-video"
            style={{ transform: "scaleX(-1)" }}
          />
        ) : (
          <div className="meet-pip-off">📷</div>
        )}
        <span className="meet-pip-label">You</span>
      </div>

      <button
        type="button"
        className={`meet-record-btn${recording ? " recording" : ""}`}
        title="Record lesson"
        disabled={!recording && !teacher}
        onClick={recording ? stopRecording : startRecording}
      >
        {recording && <span className="meet-rec-dot" />}
        {recording ? "Stop" : "Record"}
      </button>

      <RoomAudioRenderer />

      <ControlBar
        controls={{
          microphone: true,
          camera: true,
          screenShare: false,
          chat: false,
          leave: true,
          settings: false,
        }}
        variation="verbose"
      />
    </div>
  );
}

function LiveSession() {
  const [identity] = useState(
    () => "student-" + Math.random().toString(36).slice(2, 7)
  );
  const [conn, setConn] = useState(null);
  const [error, setError] = useState("");

  async function connect() {
    setError("");
    try {
      const res = await fetch("/api/session/token", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ identity, room: "tutor-room" }),
      });
      if (!res.ok) throw new Error(`token request failed (${res.status})`);
      setConn(await res.json());
    } catch (err) {
      setError(String(err));
    }
  }

  if (!conn) {
    return (
      <main className="live-join">
        <div className="avatar-face big">🧑‍🏫</div>
        <h2>Live tutoring room</h2>
        <p>
          Identity: <strong>{identity}</strong> · Room: <strong>tutor-room</strong>
        </p>
        <button onClick={connect}>🎙️ Join (mic + camera)</button>
        <p className="note">
          The browser will ask for microphone and camera permission. Milestone
          1 verifies the realtime connection; the AI avatar joins this same
          room in a later phase.
        </p>
        {error && <p className="note error">{error}</p>}
      </main>
    );
  }

  return (
    <main className="live-room">
      <LiveKitRoom
        serverUrl={conn.url}
        token={conn.token}
        connect
        audio
        video
        style={{ height: "100%", width: "100%" }}
      >
        <MeetLayout />
      </LiveKitRoom>
    </main>
  );
}

function CotVisualizer({ debug }) {
  const steps = debug?.cot_steps || [];
  const validation = debug?.cot_validation || {};
  if (!steps.length) return null;
  const perStep = {};
  for (const p of validation.per_step || []) perStep[p.label] = p;
  const pct = Math.round((validation.grounded_fraction ?? 0) * 100);
  return (
    <details className="cot">
      <summary>
        CoT Visualizer ({steps.length} steps
        {validation.validated ? `, ${pct}% KG-grounded` : ""})
      </summary>
      <ul>
        {steps.map((s) => {
          const v = perStep[s.label];
          return (
            <li key={s.label}>
              {v ? (v.grounded ? "✅" : "⚠️") : "•"}{" "}
              <strong>{s.label}</strong>
              {v?.matched?.length
                ? ` — grounded in KG (${v.matched.slice(0, 6).join(", ")})`
                : ""}
            </li>
          );
        })}
      </ul>
      {validation.validated && (
        <p className="caption">
          KG grounding: {pct}% of steps reference the knowledge graph.
          {validation.ungrounded_steps?.length
            ? ` Not grounded: ${validation.ungrounded_steps.join(", ")}`
            : ""}
        </p>
      )}
    </details>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    { role: "tutor", text: GREETING, debug: null },
  ]);
  const [input, setInput] = useState("");
  const [useCot, setUseCot] = useState(true);
  const [showCot, setShowCot] = useState(true);
  const [showDebug, setShowDebug] = useState(false);
  const [learnerLevel, setLearnerLevel] = useState("beginner");
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState(null);
  const [view, setView] = useState("chat");
  const [recording, setRecording] = useState(false);
  const [autoSpeak, setAutoSpeak] = useState(false);
  const [visionOn, setVisionOn] = useState(false);
  const [lastVision, setLastVision] = useState("");
  const recorderRef = useRef(null);
  const videoRef = useRef(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    if (!visionOn) return;
    let stream, interval;
    async function start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) videoRef.current.srcObject = stream;
        interval = setInterval(() => {
          if (!videoRef.current) return;
          const canvas = document.createElement("canvas");
          canvas.width = videoRef.current.videoWidth || 320;
          canvas.height = videoRef.current.videoHeight || 240;
          canvas.getContext("2d").drawImage(videoRef.current, 0, 0);
          const b64 = canvas.toDataURL("image/jpeg", 0.6).split(",")[1];
          const desc = `User webcam captured at ${new Date().toLocaleTimeString()}`;
          setLastVision(desc);
          fetch("/api/vision", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ description: desc, confidence: 0.8, frame_b64: b64 }),
          }).catch(() => {});
        }, 3000);
      } catch {}
    }
    start();
    return () => {
      clearInterval(interval);
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [visionOn]);

  async function speak(text) {
    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: cleanForSpeech(text).slice(0, 2000) }),
      });
      if (!res.ok) return;
      const blob = await res.blob();
      new Audio(URL.createObjectURL(blob)).play();
    } catch {
      /* audio unavailable - stay silent */
    }
  }

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "offline" }));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  async function send(e) {
    e.preventDefault();
    const question = input.trim();
    if (!question || busy) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: question }]);
    setBusy(true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          use_cot: useCot,
          learner_level: learnerLevel,
        }),
      });
      const data = await res.json();
      const answer = data.answer || "Error";
      setMessages((m) => [
        ...m,
        { role: "tutor", text: answer, debug: data.debug },
      ]);
      if (autoSpeak && data.answer) speak(answer);
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: "tutor", text: `Backend error: ${err}`, debug: null },
      ]);
    } finally {
      setBusy(false);
    }
  }

  async function toggleRecord() {
    if (recording) {
      recorderRef.current?.stop();
      return;
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      alert("Microphone permission denied.");
      return;
    }
    const chunks = [];
    const rec = new MediaRecorder(stream);
    rec.ondataavailable = (e) => chunks.push(e.data);
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      setRecording(false);
      const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
      const form = new FormData();
      form.append("file", blob, "speech.webm");
      setBusy(true);
      try {
        const res = await fetch("/api/stt", { method: "POST", body: form });
        const data = await res.json();
        if (data.text) {
          setInput((v) => (v ? v + " " : "") + data.text);
        }
      } finally {
        setBusy(false);
      }
    };
    rec.start();
    recorderRef.current = rec;
    setRecording(true);
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>AI Tutor</h1>
        <p className="tagline">Knowledge-Grounded + Chain-of-Thought Tutoring</p>

        <section className="panel">
          <h2>Interface</h2>
          <div className="tabs">
            <button
              className={view === "chat" ? "tab active" : "tab"}
              onClick={() => setView("chat")}
            >
              💬 Text Chat
            </button>
            <button
              className={view === "room" ? "tab active" : "tab"}
              onClick={() => setView("room")}
            >
              🎥 Live Room
            </button>
          </div>
        </section>

        <section className="panel">
          <h2>Settings</h2>
          <label className="row">
            <input
              type="checkbox"
              checked={showCot}
              onChange={(e) => setShowCot(e.target.checked)}
            />
            Show Chain-of-Thought
          </label>
          <label className="row">
            <input
              type="checkbox"
              checked={useCot}
              onChange={(e) => setUseCot(e.target.checked)}
            />
            Chain-of-Thought reasoning
          </label>
          <label className="row">
            <input
              type="checkbox"
              checked={showDebug}
              onChange={(e) => setShowDebug(e.target.checked)}
            />
            Show debug info
          </label>
          <label className="row">
            <input
              type="checkbox"
              checked={autoSpeak}
              onChange={(e) => setAutoSpeak(e.target.checked)}
            />
            Speak answers (TTS)
          </label>
          <label className="row">
            <input
              type="checkbox"
              checked={visionOn}
              onChange={(e) => setVisionOn(e.target.checked)}
            />
            📷 Enable vision (webcam)
          </label>
          {visionOn && (
            <p className="note">
              {lastVision || "Capturing frames every 3s…"}
            </p>
          )}
          <label className="row">
            Learner level
            <select
              value={learnerLevel}
              onChange={(e) => setLearnerLevel(e.target.value)}
            >
              <option value="beginner">Beginner</option>
              <option value="intermediate">Intermediate</option>
              <option value="advanced">Advanced</option>
            </select>
          </label>
        </section>

        <section className="panel">
          <h2>System</h2>
          <div className="kv">
            <span>Status</span>
            <span>{health ? health.status : "…"}</span>
          </div>
          <div className="kv">
            <span>Index size</span>
            <span>{health ? health.index_size ?? 0 : 0}</span>
          </div>
          <div className="kv">
            <span>Learner level</span>
            <span>{health ? health.learner_level : "…"}</span>
          </div>
        </section>

        <section className="panel">
          <h2>Knowledge Upload</h2>
          <UploadPanel />
        </section>

        <section className="panel">
          <h2>Multimodal (coming)</h2>
          <button className="ghost" disabled title="Phase 2: faster-whisper STT">
            🎤 Speak
          </button>
          <button className="ghost" disabled title="Phase 3: vision model">
            📷 Camera
          </button>
          <p className="note">Avatar + voice arrive on the GPU machine.</p>
        </section>
      </aside>

      {view === "chat" ? (
        <main className="chat">
          <div className="avatar-banner">
          <div className="avatar-face">🧑‍🏫</div>
          <p>{GREETING}</p>
        </div>

        <div className="messages">
          {messages.map((msg, i) => (
            <div key={i} className={`bubble ${msg.role}`}>
              {msg.role === "tutor" && msg.text !== GREETING && (
                <div className="bubble-actions">
                  <button
                    className="speak-btn"
                    title="Hear this answer"
                    onClick={() => speak(msg.text)}
                  >
                    🔊
                  </button>
                </div>
              )}
              <div className="text">{msg.text}</div>
              {msg.role === "tutor" && showCot && (
                <CotVisualizer debug={msg.debug} />
              )}
              {msg.role === "tutor" && showDebug && msg.debug && (
                <details className="debug">
                  <summary>Debug Info</summary>
                  <pre>{JSON.stringify(msg.debug, null, 2)}</pre>
                </details>
              )}
            </div>
          ))}
          {busy && <div className="bubble tutor typing">Thinking…</div>}
          <div ref={bottomRef} />
        </div>

        <form className="composer" onSubmit={send}>
          <button
            type="button"
            className={recording ? "mic recording" : "mic"}
            onClick={toggleRecord}
            disabled={busy}
            title={recording ? "Stop recording" : "Speak your question"}
          >
            {recording ? "⏹" : "🎤"}
          </button>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about computer science..."
            autoFocus
          />
          <button type="submit" disabled={busy || !input.trim()}>
            Send
          </button>
        </form>
        <video ref={videoRef} autoPlay muted playsInline style={{ display: "none" }} />
        </main>
      ) : (
        <LiveSession />
      )}
    </div>
  );
}
