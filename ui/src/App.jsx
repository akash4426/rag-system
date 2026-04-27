import { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState('');
  
  // Upload states
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    setSessionId(Math.random().toString(36).substring(2, 15));
  }, []);

  // Determine API URL based on environment
  const apiUrl = import.meta.env.DEV ? 'http://localhost:8000' : '';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch(`${apiUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage.content, session_id: sessionId }),
      });

      if (!response.ok) {
        throw new Error('API Error: Ensure backend is running.');
      }

      const data = await response.json();
      
      const assistantMessage = {
        role: 'assistant',
        content: data.response.answer,
        meta: {
          intent: data.intent_detected,
          expanded: data.expanded_queries,
          citations: data.response.citations,
          context_chunks: data.response.context_chunks
        }
      };
      
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${error.message}` }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
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

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      setUploadStatus({ type: 'success', msg: `Indexed ${file.name} successfully!` });
    } catch (error) {
      setUploadStatus({ type: 'error', msg: error.message });
    } finally {
      setIsUploading(false);
      // Reset input so the same file can be uploaded again if needed
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  return (
    <div className="app-layout">
      {/* Sidebar for Upload */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h2>Data Console</h2>
          <p>Manage Context</p>
        </div>
        
        <div className="upload-container">
          <div className="upload-box" onClick={() => fileInputRef.current.click()}>
            <input 
              type="file" 
              ref={fileInputRef}
              onChange={handleFileUpload} 
              accept=".txt,.pdf" 
              style={{ display: 'none' }} 
            />
            <div className="upload-icon">📄</div>
            <p className="upload-text">Click to upload TXT or PDF</p>
          </div>
          
          {isUploading && (
            <div className="upload-status loading">
              <span className="spinner"></span> Processing 23 Components...
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
            <span className="indicator green"></span>
            Vector DB Online
          </div>
          <div className="status-item">
            <span className="indicator green"></span>
            BM25 Sparse Online
          </div>
          <div className="status-item">
            <span className="indicator green"></span>
            Cross-Encoder Online
          </div>
        </div>
      </aside>

      {/* Main Chat Interface */}
      <main className="main-chat">
        <header className="app-header">
          <h1>Enterprise <span>RAG</span> System</h1>
          <p>Powered by OpenRouter, Hybrid Search, and Cross-Encoder Re-ranking</p>
        </header>

        <div className="chat-container">
          {messages.length === 0 && !isLoading && (
            <div className="empty-state">
              Ask a question based on your indexed documents.
            </div>
          )}
          
          {messages.map((msg, idx) => (
            <div key={idx} className={`message-wrapper ${msg.role}`}>
              <div className="message-bubble">{msg.content}</div>
              {msg.meta && <MetadataDisplay meta={msg.meta} />}
            </div>
          ))}
          
          {isLoading && (
            <div className="loading-indicator">Synthesizing response...</div>
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
              placeholder="Query the system..."
              disabled={isLoading}
            />
            <button type="submit" className="submit-btn" disabled={isLoading || !input.trim()}>
              Send
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}

function MetadataDisplay({ meta }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div>
      <button className="metadata-toggle" onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? '[-]' : '[+]'} View Processing Metadata
      </button>
      
      {isOpen && (
        <div className="metadata-panel">
          <div className="metadata-content">
            <h4>Detected Intent</h4>
            <div style={{ color: 'var(--text-main)', marginBottom: '1rem' }}>{meta.intent}</div>
            
            <h4>Expanded Queries (LLM)</h4>
            <ul>
              {meta.expanded.map((q, i) => (
                <li key={i}>{q}</li>
              ))}
            </ul>
            
            <h4>Retrieved Context Data</h4>
            <div className="context-list">
              {meta.context_chunks && meta.context_chunks.length > 0 
                ? meta.context_chunks.map((chunk, i) => (
                    <div key={i} className="context-chunk-item">
                      <span className="source-badge">DB Source: {chunk.source}</span>
                      <p className="chunk-text">"{chunk.text.length > 150 ? chunk.text.substring(0, 150) + '...' : chunk.text}"</p>
                    </div>
                  ))
                : <p className="empty-text">No data retrieved from DB</p>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
