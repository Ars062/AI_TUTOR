import { useEffect, useRef, useState } from "react";

const GREETING = "Hello! What would you like to learn today?";

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
  const bottomRef = useRef(null);

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
      setMessages((m) => [
        ...m,
        { role: "tutor", text: data.answer || "Error", debug: data.debug },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: "tutor", text: `Backend error: ${err}`, debug: null },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>AI Tutor</h1>
        <p className="tagline">Knowledge-Grounded + Chain-of-Thought Tutoring</p>

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

      <main className="chat">
        <div className="avatar-banner">
          <div className="avatar-face">🧑‍🏫</div>
          <p>{GREETING}</p>
        </div>

        <div className="messages">
          {messages.map((msg, i) => (
            <div key={i} className={`bubble ${msg.role}`}>
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
      </main>
    </div>
  );
}
