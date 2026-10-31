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
   Metadata Display (with pipeline, trace, sources)
   ═══════════════════════════════════════════ */
function CitationText({ content, citations = [], onSelect }) {
  const byId = Object.fromEntries(
    citations.filter((c) => c && typeof c === 'object').map((c) => [c.evidence_id, c])
  );
  const parts = [];
  const pattern = /\[((?:E\d+[\s,;]*)+)\]/gi;
  let lastIndex = 0;
  let match;
  let key = 0;

  while ((match = pattern.exec(content)) !== null) {
    if (match.index > lastIndex) {
      parts.push(content.slice(lastIndex, match.index));
    }
    const ids = match[1].match(/E\d+/gi) || [];
    ids.forEach((id) => {
      const evidenceId = id.toUpperCase();
      parts.push(
        <button
          key={`cite-${key++}`}
          className="citation-chip inline"
          onClick={() => byId[evidenceId] && onSelect(byId[evidenceId])}
          title={byId[evidenceId]?.section_title || 'Citation'}
        >
          {evidenceId}
        </button>
      );
    });
    lastIndex = pattern.lastIndex;
  }
  if (lastIndex < content.length) parts.push(content.slice(lastIndex));
  return <>{parts.length ? parts : content}</>;
}

function CitationDetail({ citation, onClose }) {
  if (!citation) return null;
  return (
    <div className="citation-detail">
      <div className="citation-detail-head">
        <span className="source-badge">{citation.evidence_id}</span>
        <button className="citation-close" onClick={onClose} aria-label="Close citation">×</button>
      </div>
      <div className="citation-meta">
        {citation.section_title || citation.section_id || 'Unknown section'}
        {citation.doc ? ` · ${citation.doc}` : ''}
        {citation.page != null ? ` · p.${citation.page}` : ''}
      </div>
      <p className="chunk-text">{citation.text || 'No passage text available.'}</p>
      <div className="citation-ids">
        chunk: {citation.chunk_id || 'n/a'}
        {citation.section_id ? ` · section: ${citation.section_id}` : ''}
      </div>
    </div>
  );
}

function describeRetrieval(trace = {}) {
  const retrieval = trace.retrieval || {};
  const strategy = retrieval.structural_strategy || 'disabled';
  const structuralCount = Number(retrieval.counts?.structural || 0);
  const selectedSections = retrieval.selected_sections || [];
  const sectionCount = selectedSections.length;
  const sectionsLabel = `${sectionCount} section${sectionCount === 1 ? '' : 's'}`;
  const chunksLabel = `${structuralCount} chunk${structuralCount === 1 ? '' : 's'}`;

  if (structuralCount > 0) {
    if (strategy === 'keyword_fallback') {
      return {
        kind: 'warning',
        label: 'Structural retrieval · keyword fallback',
        detail: `${sectionsLabel} · ${chunksLabel} · Ollama selection unavailable`,
      };
    }
    return {
      kind: 'structural',
      label: 'Structural retrieval · STAIR-inspired',
      detail: `${sectionsLabel} · ${chunksLabel} retrieved structurally`,
    };
  }

  if (strategy === 'llm' || strategy === 'keyword_fallback') {
    return {
      kind: 'warning',
      label: 'Structural route ran · no chunks returned',
      detail: `${sectionsLabel} selected · strategy: ${strategy}`,
    };
  }

  if (strategy === 'no_sections') {
    return {
      kind: 'error',
      label: 'Structural route failed · no sections',
      detail: 'Upload a document with extractable structure and try again.',
    };
  }

  if ((trace.mode || 'baseline') === 'sah') {
    return {
      kind: 'neutral',
      label: 'Adaptive hybrid retrieval',
      detail: 'SAH mode was active, but the structural route did not contribute.',
    };
  }

  return {
    kind: 'neutral',
    label: 'Baseline hybrid retrieval',
    detail: 'BM25 + dense retrieval · structural route off',
  };
}

function MetadataDisplay({ meta }) {
  const [openPanel, setOpenPanel] = useState(null);
  const toggle = (panel) => setOpenPanel(openPanel === panel ? null : panel);
  const trace = meta.trace || {};
  const retrieval = trace.retrieval || {};
  const validation = trace.validation || trace.initial_validation;
  const evidence = trace.evidence || {};
  const generation = trace.generation || {};
  const chunks = meta.context_chunks || [];
  const retrievalStatus = describeRetrieval(trace);
  const structuralChunks = Number(retrieval.counts?.structural || 0);
  const selectedSectionCount = retrieval.selected_sections?.length || 0;

  return (
    <div>
      <button
        type="button"
        className={`retrieval-mode ${retrievalStatus.kind}`}
        title="Open the retrieval trace"
        onClick={() => setOpenPanel('trace')}
        aria-expanded={openPanel === 'trace'}
      >
        <span className="retrieval-mode-dot" />
        <span className="retrieval-mode-copy">
          <span className="retrieval-mode-label">{retrievalStatus.label}</span>
          <span className="retrieval-mode-detail">{retrievalStatus.detail}</span>
        </span>
        <span className="retrieval-mode-arrow">
          {openPanel === 'trace' ? '↓' : '→'}
        </span>
      </button>

      <div className="metadata-actions">
        <button className="metadata-toggle" onClick={() => toggle('pipeline')}>
          <span className={`chevron ${openPanel === 'pipeline' ? 'open' : ''}`}>▶</span>
          Processing Pipeline
        </button>
        <button className="metadata-toggle" onClick={() => toggle('trace')}>
          <span className={`chevron ${openPanel === 'trace' ? 'open' : ''}`}>▶</span>
          Retrieval Trace
        </button>
        <button className="metadata-toggle" onClick={() => toggle('sources')}>
          <span className={`chevron ${openPanel === 'sources' ? 'open' : ''}`}>▶</span>
          Sources {chunks.length > 0 ? `(${chunks.length})` : ''}
        </button>
      </div>

      {openPanel === 'pipeline' && (
        <div className="metadata-panel">
          <PipelineFlow />
          <div className="metadata-content">
            <h4>Detected Intent</h4>
            <div style={{ color: 'var(--text-primary)', marginBottom: '1rem' }}>{meta.intent}</div>

            <h4>Expanded Queries</h4>
            <ul>
              {(meta.expanded || []).map((q, i) => (
                <li key={i}>{q}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {openPanel === 'trace' && (
        <div className="metadata-panel">
          <div className="metadata-content trace-grid">
            <div>
              <h4>Mode</h4>
              <p className="trace-value">{trace.mode || 'baseline'}</p>
            </div>
            <div>
              <h4>Retrievers</h4>
              <p className="trace-value">
                {(retrieval.retrievers || trace.controller?.retrievers || ['bm25', 'vector_db']).join(' + ')}
              </p>
            </div>
            <div>
              <h4>Complexity</h4>
              <p className="trace-value">{trace.controller?.complexity || 'n/a'}</p>
            </div>
            <div>
              <h4>Structural Strategy</h4>
              <p className="trace-value">{retrieval.structural_strategy || 'disabled'}</p>
            </div>
            <div>
              <h4>Structural Evidence</h4>
              <p className="trace-value">
                {structuralChunks} chunks · {selectedSectionCount} sections selected
              </p>
            </div>
            <div>
              <h4>Evidence Tokens</h4>
              <p className="trace-value">{evidence.token_count ?? 'n/a'}</p>
            </div>
          </div>

          {trace.controller?.reason && (
            <>
              <h4>Routing Reason</h4>
              <p className="trace-value">{trace.controller.reason}</p>
            </>
          )}

          <h4>Retrieval Counts</h4>
          <p className="trace-value">
            {retrieval.counts
              ? Object.entries(retrieval.counts).map(([k, v]) => `${k}: ${v}`).join(' · ')
              : `candidates: ${retrieval.candidates ?? 'n/a'} · returned: ${retrieval.returned ?? 'n/a'}`}
          </p>

          {retrieval.selected_sections?.length > 0 && (
            <>
              <h4>Selected Sections</h4>
              <ul>
                {retrieval.selected_sections.map((section, i) => (
                  <li key={i}>{section.title || section.section_id}</li>
                ))}
              </ul>
            </>
          )}

          <h4>Validation</h4>
          <p className="trace-value">
            {validation
              ? `${validation.sufficient ? 'sufficient' : 'insufficient'} · max ${Number(validation.max_score).toFixed(2)} · coverage ${(validation.coverage * 100).toFixed(0)}%`
              : 'not enabled'}
            {validation?.missing_aspects?.length
              ? ` · missing: ${validation.missing_aspects.join(', ')}`
              : ''}
          </p>

          {trace.retries?.length > 0 && (
            <>
              <h4>Retries</h4>
              <ul>
                {trace.retries.map((retry, i) => (
                  <li key={i}>
                    {retry.mode} #{retry.attempt}: {retry.reason || retry.query || 're-query'}
                  </li>
                ))}
              </ul>
            </>
          )}

          <h4>Generation</h4>
          <p className="trace-value">
            {generation.provider || 'openrouter'} · grounded: {String(generation.grounded ?? false)}
            {generation.verification
              ? ` · citations ${generation.verification.claims_supported}/${generation.verification.claims_total}`
              : ''}
          </p>
        </div>
      )}

      {openPanel === 'sources' && (
        <div className="metadata-panel">
          <div className="context-list">
            {chunks.length > 0
              ? chunks.map((chunk, i) => (
                  <div key={i} className="context-chunk-item">
                    <div className="context-chunk-head">
                      <span className="source-badge">
                        {chunk.evidence_id || chunk.role || `S${i + 1}`}
                      </span>
                      <span className="context-source">{chunk.source || chunk.doc}</span>
                    </div>
                    {chunk.section_title && (
                      <div className="context-section">{chunk.section_title}</div>
                    )}
                    <p className="chunk-text">
                      {chunk.text
                        ? `"${chunk.text.length > 220 ? chunk.text.substring(0, 220) + '…' : chunk.text}"`
                        : 'No passage text'}
                    </p>
                  </div>
                ))
              : <p className="empty-text">No context retrieved</p>}
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
  const [activeCitation, setActiveCitation] = useState(null);
  const citations = meta?.citations || [];

  return (
    <div className="message-wrapper assistant">
      <div className={`message-bubble ${meta?.abstained ? 'abstained' : ''}`}>
        {done ? (
          <CitationText
            content={content}
            citations={citations}
            onSelect={setActiveCitation}
          />
        ) : (
          displayed
        )}
        {!done && <span className="typing-cursor" />}
      </div>

      {done && meta?.abstained && (
        <div className="abstention-note">Abstained — evidence was insufficient.</div>
      )}

      {done && citations.length > 0 && (
        <div className="citation-row">
          {citations.filter((c) => c && typeof c === 'object').map((citation) => (
            <button
              key={citation.evidence_id}
              className={`citation-pill ${activeCitation?.evidence_id === citation.evidence_id ? 'active' : ''}`}
              onClick={() => setActiveCitation(citation)}
            >
              {citation.evidence_id} · {citation.section_title || citation.doc || 'source'}
            </button>
          ))}
        </div>
      )}

      {done && <CitationDetail citation={activeCitation} onClose={() => setActiveCitation(null)} />}
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
   Source Explorer — documents and section trees
   ═══════════════════════════════════════════ */
function SourceExplorer({ apiUrl, refreshKey }) {
  const [documents, setDocuments] = useState([]);
  const [openDoc, setOpenDoc] = useState(null);
  const [sections, setSections] = useState({});
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    fetch(`${apiUrl}/documents`)
      .then((response) => {
        if (!response.ok) throw new Error('Source store unavailable');
        return response.json();
      })
      .then((data) => {
        if (!cancelled) {
          setDocuments(data.documents || []);
          setError('');
        }
      })
      .catch((err) => !cancelled && setError(err.message));
    return () => { cancelled = true; };
  }, [apiUrl, refreshKey]);

  const toggleDoc = async (doc) => {
    if (openDoc === doc.doc_id) {
      setOpenDoc(null);
      return;
    }
    setOpenDoc(doc.doc_id);
    if (sections[doc.doc_id]) return;
    try {
      const response = await fetch(`${apiUrl}/documents/${doc.doc_id}/structure`);
      if (!response.ok) throw new Error('Structure unavailable');
      const data = await response.json();
      setSections((prev) => ({ ...prev, [doc.doc_id]: data.sections || [] }));
    } catch (err) {
      setSections((prev) => ({ ...prev, [doc.doc_id]: [] }));
      setError(err.message);
    }
  };

  return (
    <div className="source-explorer">
      <div className="explorer-head">
        <h3>Source Explorer</h3>
        <span className="explorer-count">{documents.length}</span>
      </div>

      {error && <p className="explorer-empty">{error}</p>}
      {!error && documents.length === 0 && (
        <p className="explorer-empty">No structured documents yet.</p>
      )}

      {documents.map((doc) => (
        <div className="explorer-doc" key={doc.doc_id}>
          <button className="explorer-doc-row" onClick={() => toggleDoc(doc)}>
            <span className={`chevron ${openDoc === doc.doc_id ? 'open' : ''}`}>▶</span>
            <span className="explorer-doc-name">{doc.filename}</span>
            <span className="explorer-doc-count">{doc.section_count}§</span>
          </button>
          {openDoc === doc.doc_id && (
            <div className="explorer-sections">
              {(sections[doc.doc_id] || []).map((section) => (
                <div
                  className="explorer-section"
                  key={section.section_id}
                  style={{ paddingLeft: `${8 + Math.min(section.level || 1, 4) * 10}px` }}
                  title={section.section_id}
                >
                  <span>{section.title}</span>
                  {section.page_start != null && (
                    <span className="explorer-page">p.{section.page_start}</span>
                  )}
                </div>
              ))}
              {sections[doc.doc_id]?.length === 0 && (
                <p className="explorer-empty">No sections extracted.</p>
              )}
            </div>
          )}
        </div>
      ))}
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
  const [sourceVersion, setSourceVersion] = useState(0);

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
            trace: data.trace,
            abstained: data.response.abstained,
            verification: data.response.verification,
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
      setSourceVersion((version) => version + 1);
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
              accept=".txt,.pdf,.md,.markdown,.docx"
              style={{ display: 'none' }}
            />
            <div className="upload-icon">↑</div>
            <p className="upload-text">
              <strong>Drop files here</strong> or click to browse
              <br />
              TXT, PDF, MD, DOCX supported
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

        <SourceExplorer apiUrl={apiUrl} refreshKey={sourceVersion} />
      </aside>

      {/* ── Main Chat ── */}
      <main className="main-chat">
        <header className="app-header">
          <div>
            <h1>SAH-<span className="accent-word">RAG</span></h1>
            <p className="subtitle">Structure-Aware Adaptive Hybrid RAG · Grounded Citations</p>
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
