import { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';

/* ═══════════════════════════════════════════
   Canvas Background — animated dot grid
   ═══════════════════════════════════════════ */
function CanvasBackground() {
  const canvasRef = useRef(null);
  const mouse = useRef({ x: -1000, y: -1000 });
  const raf = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    const SPACING = 40;
    const BASE_RADIUS = 1;
    const INTERACT_RADIUS = 120;

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const cols = Math.ceil(canvas.width / SPACING);
      const rows = Math.ceil(canvas.height / SPACING);

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const x = c * SPACING + SPACING / 2;
          const y = r * SPACING + SPACING / 2;
          const dx = mouse.current.x - x;
          const dy = mouse.current.y - y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          let radius = BASE_RADIUS;
          let alpha = 0.08;

          if (dist < INTERACT_RADIUS) {
            const t = 1 - dist / INTERACT_RADIUS;
            radius = BASE_RADIUS + t * 2.5;
            alpha = 0.08 + t * 0.25;
          }

          ctx.beginPath();
          ctx.arc(x, y, radius, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(245, 158, 11, ${alpha})`;
          ctx.fill();
        }
      }
      raf.current = requestAnimationFrame(draw);
    };

    const handleMouse = (e) => {
      mouse.current = { x: e.clientX, y: e.clientY };
    };

    window.addEventListener('mousemove', handleMouse);
    draw();

    return () => {
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', handleMouse);
      cancelAnimationFrame(raf.current);
    };
  }, []);

  return <canvas ref={canvasRef} className="bg-canvas" />;
}

/* ═══════════════════════════════════════════
   Pipeline Flow Visualization
   ═══════════════════════════════════════════ */
const PIPELINE_STAGES = [
  'Preprocess', 'Classify', 'Expand',
  'BM25', 'Embed', 'VectorDB',
  'Fuse', 'Rerank',
  'Context', 'Prompt', 'LLM', 'Format'
];

function PipelineFlow() {
  const [activeIdx, setActiveIdx] = useState(-1);

  useEffect(() => {
    let i = 0;
    const interval = setInterval(() => {
      setActiveIdx(i);
      i++;
      if (i >= PIPELINE_STAGES.length) {
        clearInterval(interval);
        setTimeout(() => setActiveIdx(PIPELINE_STAGES.length), 400);
      }
    }, 200);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="pipeline-flow">
      {PIPELINE_STAGES.map((stage, idx) => (
        <div key={stage} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span className={`pipeline-node ${idx <= activeIdx ? 'active' : ''}`}>
            {stage}
          </span>
          {idx < PIPELINE_STAGES.length - 1 && (
            <span className={`pipeline-arrow ${idx < activeIdx ? 'active' : ''}`}>→</span>
          )}
        </div>
      ))}
    </div>
  );
}

/* ═══════════════════════════════════════════
   Metadata Display (with pipeline)
   ═══════════════════════════════════════════ */
function MetadataDisplay({ meta }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div>
      <button className="metadata-toggle" onClick={() => setIsOpen(!isOpen)}>
        <span className={`chevron ${isOpen ? 'open' : ''}`}>▶</span>
        Processing Pipeline
      </button>

      {isOpen && (
        <div className="metadata-panel">
          <PipelineFlow />
          <div className="metadata-content">
            <h4>Detected Intent</h4>
            <div style={{ color: 'var(--text-primary)', marginBottom: '1rem' }}>{meta.intent}</div>

            <h4>Expanded Queries</h4>
            <ul>
              {meta.expanded.map((q, i) => (
                <li key={i}>{q}</li>
              ))}
            </ul>

            <h4>Retrieved Context</h4>
            <div className="context-list">
              {meta.context_chunks && meta.context_chunks.length > 0
                ? meta.context_chunks.map((chunk, i) => (
                    <div key={i} className="context-chunk-item">
                      <span className="source-badge">Source: {chunk.source}</span>
                      <p className="chunk-text">
                        "{chunk.text.length > 150 ? chunk.text.substring(0, 150) + '…' : chunk.text}"
                      </p>
                    </div>
                  ))
                : <p className="empty-text">No context retrieved</p>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════
   Typing Reveal Hook
   ═══════════════════════════════════════════ */
function useTypingReveal(text, speed = 12) {
  const [displayed, setDisplayed] = useState('');
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!text) return;
    setDisplayed('');
    setDone(false);
    let i = 0;
    const interval = setInterval(() => {
      i++;
      setDisplayed(text.slice(0, i));
      if (i >= text.length) {
        clearInterval(interval);
        setDone(true);
      }
    }, speed);
    return () => clearInterval(interval);
  }, [text, speed]);

  return { displayed, done };
}

/* ═══════════════════════════════════════════
   Message Bubble with typing effect
   ═══════════════════════════════════════════ */
function AssistantBubble({ content, meta }) {
  const { displayed, done } = useTypingReveal(content, 8);

  return (
    <div className="message-wrapper assistant">
      <div className="message-bubble">
        {displayed}
        {!done && <span className="typing-cursor" />}
      </div>
      {done && meta && <MetadataDisplay meta={meta} />}
    </div>
  );
}

/* ═══════════════════════════════════════════
   Welcome Screen
   ═══════════════════════════════════════════ */
const SUGGESTED = [
  { text: 'Summarize my uploaded documents' },
  { text: 'What are the key findings?' },
  { text: 'Compare topics across files' },
  { text: 'Explain the main concepts' },
];

function WelcomeScreen({ onPrompt }) {
  return (
    <div className="welcome-screen">
      <div className="welcome-logo">R</div>
      <div className="welcome-text">
        <h2>What do you want to know?</h2>
        <p>Upload documents and ask questions. The system uses hybrid search, cross-encoder re-ranking, and LLM generation across 23 pipeline components.</p>
      </div>
      <div className="suggested-prompts">
        {SUGGESTED.map((s, i) => (
          <button key={i} className="prompt-chip" onClick={() => onPrompt(s.text)}>
            {s.text}
          </button>
        ))}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════
   Main App
   ═══════════════════════════════════════════ */
function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [dragOver, setDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [copied, setCopied] = useState(false);

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    setSessionId(Math.random().toString(36).substring(2, 12));
  }, []);

  // In a monolithic deployment, the API is served from the same domain, so we use a relative path.
  const apiUrl = import.meta.env.VITE_API_URL || '';

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => { scrollToBottom(); }, [messages, isLoading, scrollToBottom]);

  /* ── Send message ── */
  const sendMessage = async (text) => {
    if (!text.trim()) return;
    const userMessage = { role: 'user', content: text };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch(`${apiUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, session_id: sessionId }),
      });

      if (!response.ok) throw new Error('Backend unreachable — is it running?');
      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.response.answer,
          meta: {
            intent: data.intent_detected,
            expanded: data.expanded_queries,
            citations: data.response.citations,
            context_chunks: data.response.context_chunks,
          },
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${error.message}` },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    sendMessage(input);
  };

  /* ── File upload ── */
  const uploadFile = async (file) => {
    if (!file) return;
    setIsUploading(true);
    setUploadStatus(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${apiUrl}/upload`, {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Upload failed');
      }
      
      setUploadStatus({ type: 'success', msg: `✓ ${data.message} (${data.chunks} chunks)` });
    } catch (error) {
      console.error('Upload error:', error);
      setUploadStatus({ type: 'error', msg: `✗ ${error.message}` });
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleFileChange = (e) => uploadFile(e.target.files[0]);

  /* ── Drag & Drop ── */
  const handleDragOver = (e) => { e.preventDefault(); setDragOver(true); };
  const handleDragLeave = () => setDragOver(false);
  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  };

  /* ── Copy session ID ── */
  const copySession = () => {
    navigator.clipboard.writeText(sessionId);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="app-layout">
      <CanvasBackground />

      {/* Sidebar Toggle */}
      <button
        className={`sidebar-toggle ${sidebarOpen ? 'active' : ''}`}
        onClick={() => setSidebarOpen(!sidebarOpen)}
        aria-label="Toggle sidebar"
      >
        <span className="bar" />
        <span className="bar" />
        <span className="bar" />
      </button>

      {/* ── Sidebar ── */}
      <aside className={`sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
        <div className="sidebar-header">
          <h2>Data Console</h2>
          <p>Manage Context</p>
        </div>

        <div className="upload-container">
          <div
            className={`upload-box ${dragOver ? 'drag-over' : ''}`}
            onClick={() => fileInputRef.current.click()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".txt,.pdf"
              style={{ display: 'none' }}
            />
            <div className="upload-icon">↑</div>
            <p className="upload-text">
              <strong>Drop files here</strong> or click to browse
              <br />
              TXT, PDF supported
            </p>
          </div>

          {isUploading && (
            <div className="upload-status loading">
              <span className="spinner" /> Processing pipeline…
            </div>
          )}

          {uploadStatus && !isUploading && (
            <div className={`upload-status ${uploadStatus.type}`}>
              {uploadStatus.msg}
            </div>
          )}
        </div>

        <div className="sidebar-info">
          <h3>System Status</h3>
          <div className="status-item">
            <span className="indicator green" />
            Vector DB
          </div>
          <div className="status-item">
            <span className="indicator green" />
            BM25 Sparse
          </div>
          <div className="status-item">
            <span className="indicator green" />
            Cross-Encoder
          </div>

          <div className="session-badge" onClick={copySession} title="Click to copy">
            <span className="copy-label">{copied ? 'copied' : 'session'}</span>
            {sessionId}
          </div>
        </div>
      </aside>

      {/* ── Main Chat ── */}
      <main className="main-chat">
        <header className="app-header">
          <div>
            <h1>Enterprise <span className="accent-word">RAG</span> System</h1>
            <p className="subtitle">Hybrid Search · Cross-Encoder · 23 Components</p>
          </div>
        </header>

        <div className="chat-container">
          {messages.length === 0 && !isLoading && (
            <WelcomeScreen onPrompt={(text) => sendMessage(text)} />
          )}

          {messages.map((msg, idx) =>
            msg.role === 'assistant' ? (
              <AssistantBubble key={idx} content={msg.content} meta={msg.meta} />
            ) : (
              <div key={idx} className="message-wrapper user">
                <div className="message-bubble">{msg.content}</div>
              </div>
            )
          )}

          {isLoading && (
            <div className="loading-indicator">
              <div className="loading-dots">
                <span /><span /><span />
              </div>
              Synthesizing…
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <form onSubmit={handleSubmit} className="input-form">
            <input
              type="text"
              className="input-field"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask something about your documents…"
              disabled={isLoading}
              id="query-input"
            />
            <button
              type="submit"
              className="submit-btn"
              disabled={isLoading || !input.trim()}
              id="submit-btn"
            >
              Send
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;
