import React, { useState, useEffect, useRef } from 'react';
import { 
  Terminal, Search, Cpu, Zap, ShieldCheck, Code2, Play, 
  Layers, CheckCircle2, AlertCircle, ArrowRight, RefreshCw, 
  Globe, Database, Server, Copy, Check, Sparkles, Image as ImageIcon,
  MessageSquare, Send, Trash2, ChevronDown, ChevronUp, Paperclip, X,
  Maximize2
} from 'lucide-react';
import './index.css';

const API_BASE = '';

const SAMPLE_QUERIES = [
  "What does AI stand for? 1 of 5 A Automated Information B Applied Interface C Advanced Internet D Artificial Intelligence",
  "A composite B+ tree index exists on (customer_id, order_date). Which query can most directly benefit from the index's leftmost-prefix property? A WHERE customer_id = 42 AND order_date >= '2026-01-01' B WHERE YEAR(order_date) = 2026 C WHERE customer_id + 1 = 42 D WHERE order_date >= '2026-01-01'",
  "A relation has functional dependencies A → B and B → C, with A as a candidate key. Which statement best describes the dependency A → C? A It violates reflexivity B It follows by transitivity C It cannot be inferred from the given dependencies D It follows only if C is a candidate key",
  "Top 10 places to visit in the world",
  "Convert 100 USD to EUR",
  "Write a python solution for Leetcode 3 Longest Substring Without Repeating Characters"
];

export default function App() {
  const [activeTab, setActiveTab] = useState('chat'); // 'chat' | 'pipeline' | 'mcp'
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [copiedIndex, setCopiedIndex] = useState(null);
  const [serverHealth, setServerHealth] = useState(null);

  // Chat conversation state
  const [messages, setMessages] = useState([
    {
      id: 'init-1',
      sender: 'assistant',
      text: "👋 Welcome to **Dynamic Hybrid RAG 2.0 Chatbot**!\n\nYou can ask any question, search real-time data, solve complex database / ML quiz questions, or **paste screenshots / photos directly (`Ctrl+V`)** to analyze equations, code, diagrams, or quiz questions!",
      image: null,
      telemetry: null,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [pastedImage, setPastedImage] = useState(null); // { file, preview, name, size }
  const [isDragging, setIsDragging] = useState(false);
  const [expandedTelemetry, setExpandedTelemetry] = useState({});
  const [modalImage, setModalImage] = useState(null);

  const messagesEndRef = useRef(null);
  const chatInputRef = useRef(null);

  // MCP Tester state
  const [mcpTool, setMcpTool] = useState('search_leetcode_solution');
  const [mcpArg, setMcpArg] = useState('Longest Substring Without Repeating Characters');
  const [mcpLoading, setMcpLoading] = useState(false);
  const [mcpResult, setMcpResult] = useState(null);

  useEffect(() => {
    checkHealth();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      const data = await res.json();
      setServerHealth(data);
    } catch (e) {
      setServerHealth({ status: 'offline' });
    }
  };

  // ── Clipboard Paste Handler (Ctrl+V) ──────────────────────────────────────
  const handlePaste = (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;

    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf('image') !== -1) {
        const blob = items[i].getAsFile();
        if (blob) {
          processImageFile(blob);
          e.preventDefault();
          break;
        }
      }
    }
  };

  // ── File Selection & Drag-Drop ───────────────────────────────────────────
  const processImageFile = (file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onloadend = () => {
      setPastedImage({
        file: file,
        preview: reader.result,
        name: file.name || `pasted_image_${Date.now()}.png`,
        size: `${(file.size / 1024).toFixed(1)} KB`
      });
    };
    reader.readAsDataURL(file);
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      processImageFile(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && (file.type.startsWith('image/') || file.name.endsWith('.pdf'))) {
      processImageFile(file);
    }
  };

  // ── Submit Query (Chat Mode) ─────────────────────────────────────────────
  const handleSendChatMessage = async (overridePrompt) => {
    const promptToSend = overridePrompt !== undefined ? overridePrompt : query;
    if (!promptToSend.trim() && !pastedImage) return;

    const currentImage = pastedImage;
    const userMsgId = `user-${Date.now()}`;
    const userMsg = {
      id: userMsgId,
      sender: 'user',
      text: promptToSend.trim(),
      image: currentImage?.preview || null,
      imageName: currentImage?.name || null,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages((prev) => [...prev, userMsg]);
    setQuery('');
    setPastedImage(null);
    setLoading(true);
    setError(null);

    try {
      let endpoint = `${API_BASE}/api/query`;
      let payload = { query: promptToSend.trim() };

      if (currentImage?.preview) {
        endpoint = `${API_BASE}/api/query/multimodal`;
        payload = {
          query: promptToSend.trim(),
          image_base64: currentImage.preview,
          pdf_path: currentImage.file?.name?.endsWith('.pdf') ? currentImage.file.name : null
        };
      }

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}: Server Error`);
      const data = await res.json();
      setResult(data);

      const assistantMsg = {
        id: `assistant-${Date.now()}`,
        sender: 'assistant',
        text: data.final_answer || 'No response generated.',
        telemetry: data,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const errMsg = {
        id: `assistant-${Date.now()}`,
        sender: 'assistant',
        text: `⚠️ **Execution Error**: ${err.message || 'Failed to connect to backend server.'}`,
        telemetry: null,
        isError: true,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setLoading(false);
      chatInputRef.current?.focus();
    }
  };

  // ── Submit Query (Pipeline Console Mode) ──────────────────────────────────
  const handlePipelineSearch = async (inputQuery) => {
    const targetQ = inputQuery || query;
    if (!targetQ.trim() && !pastedImage) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let endpoint = `${API_BASE}/api/query`;
      let payload = { query: targetQ };

      if (pastedImage?.preview) {
        endpoint = `${API_BASE}/api/query/multimodal`;
        payload = {
          query: targetQ,
          image_base64: pastedImage.preview,
          pdf_path: pastedImage.file?.name?.endsWith('.pdf') ? pastedImage.file.name : null
        };
      }

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}: Server Error`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to connect to Dynamic Hybrid RAG backend server.');
    } finally {
      setLoading(false);
    }
  };

  const handleMcpCall = async () => {
    setMcpLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/mcp/call`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tool_name: mcpTool,
          arguments: { problem: mcpArg, topic: mcpArg, query: mcpArg }
        })
      });
      const data = await res.json();
      setMcpResult(data);
    } catch (e) {
      setMcpResult({ error: e.message });
    } finally {
      setMcpLoading(false);
    }
  };

  const copyToClipboard = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(idx);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <div style={{ maxWidth: '1440px', margin: '0 auto', padding: '20px 16px', minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      
      {/* ── Header Bar ────────────────────────────────────────────────────────── */}
      <header className="glass-panel" style={{ padding: '16px 28px', marginBottom: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{ background: 'linear-gradient(135deg, #38bdf8, #818cf8)', width: '42px', height: '42px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 20px rgba(56, 189, 248, 0.4)' }}>
            <Cpu size={24} color="#090d16" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h1 style={{ fontSize: '20px', fontWeight: '700', letterSpacing: '-0.5px', background: 'linear-gradient(to right, #f8fafc, #38bdf8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                Dynamic Hybrid RAG 2.0
              </h1>
              <span style={{ fontSize: '11px', padding: '2px 8px', borderRadius: '12px', background: 'rgba(56, 189, 248, 0.15)', color: '#38bdf8', fontWeight: '600', border: '1px solid rgba(56, 189, 248, 0.3)' }}>
                DQN Quiz + Vision OCR
              </span>
            </div>
            <p style={{ fontSize: '12px', color: '#94a3b8' }}>
              Paste Images (`Ctrl+V`) • DQN Option Selector • Multi-Stage RAG
            </p>
          </div>
        </div>

        {/* Server Status Badge & Navigation Tabs */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 12px', borderRadius: '20px', background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(255,255,255,0.08)' }}>
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: serverHealth?.status === 'online' ? '#34d399' : '#f43f5e', boxShadow: serverHealth?.status === 'online' ? '0 0 10px #34d399' : 'none' }} />
            <span style={{ fontSize: '12px', color: '#cbd5e1', textTransform: 'capitalize' }}>
              {serverHealth?.status || 'Connecting...'}
            </span>
          </div>

          <div style={{ display: 'flex', background: 'rgba(15, 23, 42, 0.8)', padding: '4px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)' }}>
            <button 
              onClick={() => setActiveTab('chat')}
              style={{ padding: '8px 16px', borderRadius: '8px', border: 'none', background: activeTab === 'chat' ? '#38bdf8' : 'transparent', color: activeTab === 'chat' ? '#090d16' : '#94a3b8', fontWeight: '600', fontSize: '13px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', transition: 'all 0.2s' }}>
              <MessageSquare size={15} /> AI Chatbot
            </button>
            <button 
              onClick={() => setActiveTab('pipeline')}
              style={{ padding: '8px 16px', borderRadius: '8px', border: 'none', background: activeTab === 'pipeline' ? '#38bdf8' : 'transparent', color: activeTab === 'pipeline' ? '#090d16' : '#94a3b8', fontWeight: '600', fontSize: '13px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', transition: 'all 0.2s' }}>
              <Zap size={15} /> Pipeline Console
            </button>
            <button 
              onClick={() => setActiveTab('mcp')}
              style={{ padding: '8px 16px', borderRadius: '8px', border: 'none', background: activeTab === 'mcp' ? '#38bdf8' : 'transparent', color: activeTab === 'mcp' ? '#090d16' : '#94a3b8', fontWeight: '600', fontSize: '13px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', transition: 'all 0.2s' }}>
              <Code2 size={15} /> MCP Tools
            </button>
          </div>
        </div>
      </header>

      {/* ── TAB 1: AI CHATBOT (PASTE PHOTO & CHAT) ─────────────────────────────── */}
      {activeTab === 'chat' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', flex: 1, gap: '16px' }}>
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 140px)', position: 'relative', overflow: 'hidden' }}>
            
            {/* Chat Messages Scroll Area */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
              {messages.map((msg, idx) => {
                const isUser = msg.sender === 'user';
                const hasTelemetry = !!msg.telemetry;
                const isExpanded = !!expandedTelemetry[msg.id];

                return (
                  <div 
                    key={msg.id || idx}
                    className="fade-in"
                    style={{ 
                      display: 'flex', 
                      flexDirection: isUser ? 'row-reverse' : 'row', 
                      gap: '12px',
                      maxWidth: isUser ? '80%' : '85%',
                      alignSelf: isUser ? 'flex-end' : 'flex-start'
                    }}>
                    
                    {/* Avatar */}
                    <div style={{ 
                      width: '36px', 
                      height: '36px', 
                      borderRadius: '10px', 
                      background: isUser ? 'linear-gradient(135deg, #38bdf8, #818cf8)' : 'rgba(30, 41, 59, 0.9)', 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center', 
                      flexShrink: 0,
                      border: isUser ? 'none' : '1px solid rgba(56, 189, 248, 0.3)',
                      color: isUser ? '#090d16' : '#38bdf8',
                      fontWeight: '700',
                      fontSize: '13px'
                    }}>
                      {isUser ? 'YOU' : 'AI'}
                    </div>

                    {/* Message Card */}
                    <div className={isUser ? 'chat-bubble-user' : 'chat-bubble-assistant'} style={{ padding: '16px 20px', position: 'relative' }}>
                      
                      {/* Attached Image Thumbnail (if user attached photo) */}
                      {msg.image && (
                        <div style={{ marginBottom: '12px', position: 'relative' }}>
                          <img 
                            src={msg.image} 
                            alt="Pasted upload" 
                            onClick={() => setModalImage(msg.image)}
                            style={{ 
                              maxWidth: '280px', 
                              maxHeight: '200px', 
                              borderRadius: '10px', 
                              border: '1px solid rgba(56, 189, 248, 0.4)',
                              cursor: 'pointer',
                              display: 'block',
                              transition: 'transform 0.2s',
                              objectFit: 'cover'
                            }} 
                          />
                          <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <ImageIcon size={12} /> {msg.imageName || 'Pasted Image'} (Click to expand)
                          </div>
                        </div>
                      )}

                      {/* Text Content */}
                      <div style={{ fontSize: '14.5px', lineHeight: '1.6', color: '#f8fafc', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                        {msg.text}
                      </div>

                      {/* Telemetry / Quiz Details Footer (For Assistant) */}
                      {hasTelemetry && (
                        <div style={{ marginTop: '14px', paddingTop: '12px', borderTop: '1px solid rgba(255,255,255,0.08)' }}>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                              <span style={{ fontSize: '11px', padding: '3px 8px', borderRadius: '6px', background: 'rgba(56, 189, 248, 0.12)', color: '#38bdf8', fontWeight: '600' }}>
                                Route: {msg.telemetry.routing}
                              </span>
                              {msg.telemetry.domain && (
                                <span style={{ fontSize: '11px', padding: '3px 8px', borderRadius: '6px', background: 'rgba(129, 140, 248, 0.12)', color: '#818cf8', fontWeight: '600' }}>
                                  Domain: {msg.telemetry.domain}
                                </span>
                              )}
                              {msg.telemetry.verification?.passed && (
                                <span style={{ fontSize: '11px', padding: '3px 8px', borderRadius: '6px', background: 'rgba(52, 211, 153, 0.12)', color: '#34d399', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '4px' }}>
                                  <ShieldCheck size={12} /> 100% Verified
                                </span>
                              )}
                            </div>

                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                              <button
                                onClick={() => copyToClipboard(msg.text, idx)}
                                style={{ background: 'transparent', border: 'none', color: copiedIndex === idx ? '#34d399' : '#94a3b8', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px', fontSize: '12px' }}>
                                {copiedIndex === idx ? <Check size={13} /> : <Copy size={13} />} {copiedIndex === idx ? 'Copied' : 'Copy'}
                              </button>

                              <button
                                onClick={() => setExpandedTelemetry(prev => ({ ...prev, [msg.id]: !prev[msg.id] }))}
                                style={{ background: 'transparent', border: 'none', color: '#38bdf8', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '2px', fontSize: '12px', fontWeight: '600' }}>
                                Telemetry {isExpanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                              </button>
                            </div>
                          </div>

                          {/* Expanded Telemetry Box */}
                          {isExpanded && (
                            <div style={{ marginTop: '12px', padding: '12px', borderRadius: '8px', background: 'rgba(10, 15, 29, 0.9)', border: '1px solid rgba(56, 189, 248, 0.2)', fontSize: '12px', color: '#cbd5e1' }}>
                              {msg.telemetry.funnel_meta?.probability_distribution && (
                                <div style={{ marginBottom: '10px' }}>
                                  <div style={{ fontWeight: '600', color: '#38bdf8', marginBottom: '4px' }}>DQN Probability Distribution:</div>
                                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '6px' }}>
                                    {Object.entries(msg.telemetry.funnel_meta.probability_distribution).map(([letter, prob]) => (
                                      <div key={letter} style={{ padding: '4px 8px', borderRadius: '4px', background: letter === msg.telemetry.funnel_meta?.selected_letter ? 'rgba(52, 211, 153, 0.2)' : 'rgba(255,255,255,0.04)', border: letter === msg.telemetry.funnel_meta?.selected_letter ? '1px solid #34d399' : '1px solid transparent', textAlign: 'center' }}>
                                        <strong>{letter}:</strong> {(prob * 100).toFixed(1)}%
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {msg.telemetry.funnel_meta?.evidence && (
                                <div style={{ marginBottom: '6px' }}>
                                  <span style={{ color: '#818cf8', fontWeight: '600' }}>Retrieved Evidence: </span>
                                  <span style={{ color: '#94a3b8' }}>"{msg.telemetry.funnel_meta.evidence}"</span>
                                </div>
                              )}

                              {msg.telemetry.funnel_meta?.ocr_text && (
                                <div>
                                  <span style={{ color: '#c084fc', fontWeight: '600' }}>OCR Extracted Text: </span>
                                  <span style={{ color: '#94a3b8' }}>"{msg.telemetry.funnel_meta.ocr_text}"</span>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}

                      {/* Timestamp */}
                      <div style={{ fontSize: '10px', color: '#64748b', textAlign: 'right', marginTop: '6px' }}>
                        {msg.timestamp}
                      </div>
                    </div>
                  </div>
                );
              })}

              {/* Loading Chat Bubble */}
              {loading && (
                <div className="fade-in" style={{ display: 'flex', gap: '12px', maxWidth: '80%' }}>
                  <div style={{ width: '36px', height: '36px', borderRadius: '10px', background: 'rgba(30, 41, 59, 0.9)', border: '1px solid rgba(56, 189, 248, 0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#38bdf8' }}>
                    <RefreshCw className="animate-spin" size={16} />
                  </div>
                  <div className="chat-bubble-assistant" style={{ padding: '16px 20px', display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <span style={{ fontSize: '14px', color: '#cbd5e1' }}>Analyzing query & evidence via Dueling DQN & RAG...</span>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Pasted Image Preview Pill (Above Input) */}
            {pastedImage && (
              <div style={{ padding: '8px 20px', background: 'rgba(56, 189, 248, 0.1)', borderTop: '1px solid rgba(56, 189, 248, 0.25)', display: 'flex', alignItems: 'center', gap: '12px' }}>
                <img 
                  src={pastedImage.preview} 
                  alt="Attached preview" 
                  style={{ width: '36px', height: '36px', borderRadius: '6px', objectFit: 'cover', border: '1px solid #38bdf8' }} 
                />
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '13px', color: '#f8fafc', fontWeight: '600' }}>
                    📷 {pastedImage.name}
                  </div>
                  <div style={{ fontSize: '11px', color: '#38bdf8' }}>
                    {pastedImage.size} • Ready to send with OCR & Vision Analysis
                  </div>
                </div>
                <button
                  onClick={() => setPastedImage(null)}
                  style={{ background: 'rgba(244, 63, 94, 0.15)', border: '1px solid rgba(244, 63, 94, 0.3)', color: '#f43f5e', padding: '4px 10px', borderRadius: '6px', cursor: 'pointer', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <X size={13} /> Remove
                </button>
              </div>
            )}

            {/* Chat Input & Drag-Drop Bar */}
            <div 
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={isDragging ? 'dropzone-active' : ''}
              style={{ 
                padding: '16px 20px', 
                background: 'rgba(10, 15, 29, 0.95)', 
                borderTop: '1px solid rgba(255, 255, 255, 0.08)',
                transition: 'all 0.2s'
              }}>
              
              <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                
                {/* Paste / Attach Button */}
                <label 
                  title="Attach or Paste Image (Ctrl+V)"
                  style={{ 
                    padding: '12px 14px', 
                    borderRadius: '10px', 
                    background: pastedImage ? 'rgba(56, 189, 248, 0.2)' : 'rgba(255, 255, 255, 0.05)', 
                    border: '1px solid rgba(56, 189, 248, 0.3)', 
                    color: '#38bdf8', 
                    cursor: 'pointer', 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '6px',
                    fontSize: '13px',
                    fontWeight: '600'
                  }}>
                  <Paperclip size={16} />
                  <span style={{ display: 'inline' }}>Attach/Paste</span>
                  <input type="file" accept="image/*,.pdf" onChange={handleFileInputChange} style={{ display: 'none' }} />
                </label>

                {/* Main Query Input (with onPaste listener) */}
                <input
                  ref={chatInputRef}
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onPaste={handlePaste}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSendChatMessage();
                    }
                  }}
                  placeholder="Type a message or press Ctrl+V to paste a photo/screenshot..."
                  style={{ 
                    flex: 1, 
                    padding: '14px 18px', 
                    borderRadius: '10px', 
                    background: 'rgba(15, 23, 42, 0.8)', 
                    border: '1px solid var(--border-glow)', 
                    color: '#f8fafc', 
                    fontSize: '14.5px', 
                    outline: 'none' 
                  }}
                />

                {/* Send Button */}
                <button
                  onClick={() => handleSendChatMessage()}
                  disabled={loading || (!query.trim() && !pastedImage)}
                  style={{ 
                    padding: '0 24px', 
                    height: '48px', 
                    borderRadius: '10px', 
                    border: 'none', 
                    background: 'linear-gradient(135deg, #38bdf8, #818cf8)', 
                    color: '#090d16', 
                    fontWeight: '700', 
                    fontSize: '14px', 
                    cursor: 'pointer', 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '8px', 
                    opacity: (loading || (!query.trim() && !pastedImage)) ? 0.5 : 1 
                  }}>
                  {loading ? <RefreshCw className="animate-spin" size={16} /> : <Send size={16} />} Send
                </button>

                {/* Clear Chat Button */}
                <button
                  title="Clear conversation"
                  onClick={() => setMessages([])}
                  style={{ 
                    padding: '12px', 
                    borderRadius: '10px', 
                    background: 'rgba(255,255,255,0.04)', 
                    border: '1px solid rgba(255,255,255,0.08)', 
                    color: '#94a3b8', 
                    cursor: 'pointer' 
                  }}>
                  <Trash2 size={16} />
                </button>
              </div>

              {/* Sample Prompt Chips */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', alignItems: 'center', marginTop: '12px' }}>
                <span style={{ fontSize: '11px', color: '#64748b', fontWeight: '600' }}>Quick Samples:</span>
                {SAMPLE_QUERIES.map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => { setQuery(q); handleSendChatMessage(q); }}
                    style={{ padding: '4px 10px', borderRadius: '16px', background: 'rgba(56, 189, 248, 0.06)', border: '1px solid rgba(56, 189, 248, 0.15)', color: '#38bdf8', fontSize: '11px', cursor: 'pointer' }}>
                    {q.length > 35 ? q.slice(0, 35) + '...' : q}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: PIPELINE CONSOLE ───────────────────────────────────────────── */}
      {activeTab === 'pipeline' && (
        <main>
          <section className="glass-panel" style={{ padding: '28px', marginBottom: '24px' }}>
            <h2 style={{ fontSize: '17px', fontWeight: '600', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Search size={18} color="#38bdf8" /> Multi-Stage RAG Execution Console
            </h2>

            <form onSubmit={(e) => { e.preventDefault(); handlePipelineSearch(); }} style={{ display: 'flex', gap: '12px', marginBottom: '16px', alignItems: 'center' }}>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onPaste={handlePaste}
                placeholder="Ask any text query or paste an image (`Ctrl+V`)..."
                style={{ flex: 1, padding: '14px 18px', borderRadius: '10px', background: 'rgba(15, 23, 42, 0.9)', border: '1px solid var(--border-glow)', color: '#f8fafc', fontSize: '14.5px', outline: 'none' }}
              />

              <label style={{ padding: '14px 18px', borderRadius: '10px', background: 'rgba(56, 189, 248, 0.1)', border: '1px dashed #38bdf8', color: '#38bdf8', fontWeight: '600', fontSize: '13px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', whiteSpace: 'nowrap' }}>
                📷 {pastedImage ? pastedImage.name.slice(0, 15) + '...' : 'Upload/Paste'}
                <input type="file" accept="image/*,.pdf" onChange={handleFileInputChange} style={{ display: 'none' }} />
              </label>

              <button
                type="submit"
                disabled={loading}
                style={{ padding: '0 28px', height: '48px', borderRadius: '10px', border: 'none', background: 'linear-gradient(135deg, #38bdf8, #818cf8)', color: '#090d16', fontWeight: '700', fontSize: '14px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', opacity: loading ? 0.7 : 1 }}>
                {loading ? <RefreshCw className="animate-spin" size={16} /> : <Play size={16} />} Run Pipeline
              </button>
            </form>

            {pastedImage && (
              <div style={{ marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '10px', background: 'rgba(56, 189, 248, 0.08)', padding: '8px 14px', borderRadius: '8px', border: '1px solid rgba(56, 189, 248, 0.2)' }}>
                <img src={pastedImage.preview} alt="upload preview" style={{ height: '36px', borderRadius: '4px', border: '1px solid #38bdf8' }} />
                <span style={{ fontSize: '12px', color: '#f8fafc' }}>{pastedImage.name} ({pastedImage.size})</span>
                <button onClick={() => setPastedImage(null)} style={{ marginLeft: 'auto', background: 'transparent', border: 'none', color: '#f43f5e', cursor: 'pointer', fontSize: '12px', fontWeight: '700' }}>Remove ✕</button>
              </div>
            )}
          </section>

          {/* Results Display */}
          {result && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '24px' }}>
              <div>
                <section className="glass-panel" style={{ padding: '24px', marginBottom: '24px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                    <h3 style={{ fontSize: '16px', fontWeight: '600', color: '#38bdf8', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Sparkles size={18} /> Verified Generated Answer
                    </h3>
                    <button
                      onClick={() => copyToClipboard(result.final_answer, 'pipeline')}
                      style={{ padding: '6px 12px', borderRadius: '6px', background: 'rgba(56, 189, 248, 0.1)', border: '1px solid rgba(56, 189, 248, 0.3)', color: '#38bdf8', fontSize: '12px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      {copiedIndex === 'pipeline' ? <Check size={12} /> : <Copy size={12} />} {copiedIndex === 'pipeline' ? 'Copied' : 'Copy'}
                    </button>
                  </div>

                  <div style={{ fontSize: '14.5px', lineHeight: '1.7', whiteSpace: 'pre-wrap', color: '#f8fafc', background: 'rgba(10, 15, 29, 0.6)', padding: '18px', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.06)' }}>
                    {result.final_answer}
                  </div>
                </section>
              </div>

              {/* Sidebar Telemetry */}
              <div>
                <section className="glass-panel" style={{ padding: '20px' }}>
                  <h4 style={{ fontSize: '14px', fontWeight: '600', color: '#38bdf8', marginBottom: '12px' }}>
                    Pipeline Telemetry
                  </h4>
                  <div style={{ fontSize: '13px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <div><strong>Route:</strong> <span style={{ color: '#38bdf8' }}>{result.routing}</span></div>
                    <div><strong>Domain:</strong> <span style={{ color: '#818cf8' }}>{result.domain || 'N/A'}</span></div>
                    <div><strong>Intent:</strong> <span style={{ color: '#cbd5e1' }}>{result.intent?.type} ({((result.intent?.confidence || 1) * 100).toFixed(0)}%)</span></div>
                    <div><strong>Verification:</strong> <span style={{ color: '#34d399' }}>{result.verification?.passed ? 'PASSED (100%)' : 'CHECKING'}</span></div>
                  </div>
                </section>
              </div>
            </div>
          )}
        </main>
      )}

      {/* ── TAB 3: MCP TOOLS STUDIO ───────────────────────────────────────────── */}
      {activeTab === 'mcp' && (
        <main>
          <section className="glass-panel" style={{ padding: '28px' }}>
            <h2 style={{ fontSize: '17px', fontWeight: '600', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Code2 size={18} color="#38bdf8" /> Model Context Protocol (MCP) JSON-RPC Tester
            </h2>
            <div style={{ display: 'flex', gap: '12px', marginBottom: '16px' }}>
              <select
                value={mcpTool}
                onChange={(e) => setMcpTool(e.target.value)}
                style={{ padding: '12px', borderRadius: '8px', background: 'rgba(15, 23, 42, 0.9)', border: '1px solid var(--border-glow)', color: '#f8fafc', outline: 'none' }}>
                <option value="search_leetcode_solution">LeetCode Solution Tool</option>
                <option value="travel_route_finder">Travel Route Finder</option>
                <option value="finance_market_stats">Finance Market Stats</option>
                <option value="live_weather_lookup">Live Weather Lookup</option>
              </select>

              <input
                type="text"
                value={mcpArg}
                onChange={(e) => setMcpArg(e.target.value)}
                placeholder="Argument value..."
                style={{ flex: 1, padding: '12px 16px', borderRadius: '8px', background: 'rgba(15, 23, 42, 0.9)', border: '1px solid var(--border-glow)', color: '#f8fafc', outline: 'none' }}
              />

              <button
                onClick={handleMcpCall}
                disabled={mcpLoading}
                style={{ padding: '0 24px', borderRadius: '8px', border: 'none', background: 'linear-gradient(135deg, #38bdf8, #818cf8)', color: '#090d16', fontWeight: '700', cursor: 'pointer' }}>
                {mcpLoading ? 'Executing...' : 'Call MCP Tool'}
              </button>
            </div>

            {mcpResult && (
              <pre style={{ padding: '16px', borderRadius: '8px', background: 'rgba(10, 15, 29, 0.8)', border: '1px solid rgba(56, 189, 248, 0.2)', color: '#38bdf8', overflowX: 'auto' }}>
                {JSON.stringify(mcpResult, null, 2)}
              </pre>
            )}
          </section>
        </main>
      )}

      {/* ── Image Modal (Enlarge Pasted Image) ────────────────────────────────── */}
      {modalImage && (
        <div 
          onClick={() => setModalImage(null)}
          style={{ position: 'fixed', inset: 0, background: 'rgba(0, 0, 0, 0.85)', backdropFilter: 'blur(8px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9999, padding: '20px' }}>
          <div style={{ position: 'relative', maxWidth: '90vw', maxHeight: '90vh' }}>
            <img src={modalImage} alt="Enlarged view" style={{ maxWidth: '100%', maxHeight: '85vh', borderRadius: '12px', border: '1px solid rgba(56, 189, 248, 0.5)' }} />
            <button
              onClick={() => setModalImage(null)}
              style={{ position: 'absolute', top: '-14px', right: '-14px', width: '32px', height: '32px', borderRadius: '50%', background: '#f43f5e', color: '#fff', border: 'none', fontWeight: '700', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              ✕
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
