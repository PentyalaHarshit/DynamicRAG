import React, { useState, useEffect } from 'react';
import { 
  Terminal, Search, Cpu, Zap, ShieldCheck, Code2, Play, 
  Layers, CheckCircle2, AlertCircle, ArrowRight, RefreshCw, 
  Globe, Database, Server, Copy, Check, Sparkles
} from 'lucide-react';
import './index.css';

const API_BASE = 'http://localhost:8000';

const SAMPLE_QUERIES = [
  "Write a python solution for Leetcode 3 Longest Substring Without Repeating Characters",
  "Write a python solution for Leetcode 1 Two Sum",
  "Convert 100 USD to EUR",
  "What is the weather in London right now?",
  "Explain the difference between REST API and GraphQL"
];

export default function App() {
  const [activeTab, setActiveTab] = useState('pipeline');
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);
  const [serverHealth, setServerHealth] = useState(null);

  // MCP Tester state
  const [mcpTool, setMcpTool] = useState('search_leetcode_solution');
  const [mcpArg, setMcpArg] = useState('Longest Substring Without Repeating Characters');
  const [mcpLoading, setMcpLoading] = useState(false);
  const [mcpResult, setMcpResult] = useState(null);

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      const data = await res.json();
      setServerHealth(data);
    } catch (e) {
      setServerHealth({ status: 'offline' });
    }
  };

  const handleSearch = async (inputQuery) => {
    const targetQ = inputQuery || query;
    if (!targetQ.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: targetQ })
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

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '24px 16px' }}>
      
      {/* ── Header Bar ────────────────────────────────────────────────────────── */}
      <header className="glass-panel" style={{ padding: '20px 32px', marginBottom: '28px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ background: 'linear-gradient(135deg, #38bdf8, #818cf8)', width: '48px', height: '48px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 20px rgba(56, 189, 248, 0.4)' }}>
            <Cpu size={28} color="#090d16" />
          </div>
          <div>
            <h1 style={{ fontSize: '24px', fontWeight: '700', letterSpacing: '-0.5px', background: 'linear-gradient(to right, #f8fafc, #38bdf8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              Dynamic Hybrid RAG 2.0
            </h1>
            <p style={{ fontSize: '13px', color: '#94a3b8' }}>
              Hierarchical Multi-Stage RAG • MCP Web Tools • SAC Policy Learning
            </p>
          </div>
        </div>

        {/* Server Status Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 14px', borderRadius: '20px', background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(255,255,255,0.08)' }}>
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: serverHealth?.status === 'online' ? '#34d399' : '#f43f5e', boxShadow: serverHealth?.status === 'online' ? '0 0 10px #34d399' : 'none' }} />
            <span style={{ fontSize: '13px', color: '#cbd5e1', textTransform: 'capitalize' }}>
              Server: {serverHealth?.status || 'Connecting...'}
            </span>
          </div>

          {/* Navigation Tabs */}
          <div style={{ display: 'flex', background: 'rgba(15, 23, 42, 0.8)', padding: '4px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)' }}>
            <button 
              onClick={() => setActiveTab('pipeline')}
              style={{ padding: '8px 16px', borderRadius: '8px', border: 'none', background: activeTab === 'pipeline' ? '#38bdf8' : 'transparent', color: activeTab === 'pipeline' ? '#090d16' : '#94a3b8', fontWeight: '600', fontSize: '13px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', transition: 'all 0.2s' }}>
              <Zap size={15} /> RAG Pipeline
            </button>
            <button 
              onClick={() => setActiveTab('mcp')}
              style={{ padding: '8px 16px', borderRadius: '8px', border: 'none', background: activeTab === 'mcp' ? '#38bdf8' : 'transparent', color: activeTab === 'mcp' ? '#090d16' : '#94a3b8', fontWeight: '600', fontSize: '13px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', transition: 'all 0.2s' }}>
              <Code2 size={15} /> MCP Tools Studio
            </button>
          </div>
        </div>
      </header>

      {/* ── MAIN TAB 1: RAG PIPELINE DASHBOARD ───────────────────────────────────────── */}
      {activeTab === 'pipeline' && (
        <main>
          {/* Search Console Card */}
          <section className="glass-panel" style={{ padding: '32px', marginBottom: '32px' }}>
            <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Search size={20} color="#38bdf8" /> Query Execution Console
            </h2>

            <form onSubmit={(e) => { e.preventDefault(); handleSearch(); }} style={{ display: 'flex', gap: '12px', marginBottom: '20px' }}>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask any question, LeetCode problem, currency rate, or weather..."
                style={{ flex: 1, padding: '16px 20px', borderRadius: '12px', background: 'rgba(15, 23, 42, 0.9)', border: '1px solid var(--border-glow)', color: '#f8fafc', fontSize: '15px', outline: 'none' }}
              />
              <button
                type="submit"
                disabled={loading}
                style={{ padding: '0 32px', borderRadius: '12px', border: 'none', background: 'linear-gradient(135deg, #38bdf8, #818cf8)', color: '#090d16', fontWeight: '700', fontSize: '15px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', opacity: loading ? 0.7 : 1 }}>
                {loading ? <RefreshCw className="animate-spin" size={18} /> : <Play size={18} />} Run Query
              </button>
            </form>

            {/* Quick Sample Chips */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center' }}>
              <span style={{ fontSize: '12px', color: '#64748b', fontWeight: '600' }}>Sample Prompts:</span>
              {SAMPLE_QUERIES.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => { setQuery(q); handleSearch(q); }}
                  style={{ padding: '6px 12px', borderRadius: '20px', background: 'rgba(56, 189, 248, 0.08)', border: '1px solid rgba(56, 189, 248, 0.2)', color: '#38bdf8', fontSize: '12px', cursor: 'pointer', transition: 'all 0.2s' }}>
                  {q.length > 40 ? q.slice(0, 40) + '...' : q}
                </button>
              ))}
            </div>
          </section>

          {/* Loading Indicator */}
          {loading && (
            <div className="glass-panel" style={{ padding: '48px', textAlign: 'center', marginBottom: '32px' }}>
              <RefreshCw className="animate-spin" size={36} color="#38bdf8" style={{ margin: '0 auto 16px' }} />
              <h3 style={{ fontSize: '18px', fontWeight: '600' }}>Executing Multi-Stage RAG Pipeline</h3>
              <p style={{ fontSize: '14px', color: '#94a3b8', marginTop: '8px' }}>
                Routing Intent ➔ Filtering $10 \rightarrow 5 \rightarrow 3$ Chunks ➔ Verifying Answer...
              </p>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="glass-panel" style={{ padding: '24px', borderColor: 'rgba(244, 63, 94, 0.4)', background: 'rgba(244, 63, 94, 0.05)', marginBottom: '32px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', color: '#f43f5e' }}>
                <AlertCircle size={24} />
                <div>
                  <h4 style={{ fontWeight: '600' }}>Pipeline Execution Error</h4>
                  <p style={{ fontSize: '14px', color: '#fca5a5' }}>{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* Pipeline Results Dashboard */}
          {result && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '28px' }}>
              
              {/* Left Column: Final Answer & Agentic Code Studio */}
              <div>
                <section className="glass-panel" style={{ padding: '32px', marginBottom: '28px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                    <h3 style={{ fontSize: '18px', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <Sparkles size={20} color="#34d399" /> Pipeline Synthesis Answer
                    </h3>
                    <button
                      onClick={() => copyToClipboard(result.final_answer)}
                      style={{ padding: '6px 14px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.1)', background: 'rgba(15, 23, 42, 0.6)', color: '#cbd5e1', fontSize: '13px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px' }}>
                      {copied ? <Check size={14} color="#34d399" /> : <Copy size={14} />} {copied ? 'Copied' : 'Copy Answer'}
                    </button>
                  </div>

                  <div style={{ background: '#090d16', padding: '24px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)', whiteSpace: 'pre-wrap', lineHeight: '1.6', fontSize: '14px', color: '#e2e8f0' }}>
                    {result.final_answer}
                  </div>
                </section>
              </div>

              {/* Right Column: Pipeline Telemetry & Verifier Radar */}
              <div>
                {/* Intent & Router Telemetry */}
                <section className="glass-panel" style={{ padding: '24px', marginBottom: '24px' }}>
                  <h4 style={{ fontSize: '15px', fontWeight: '600', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px', color: '#38bdf8' }}>
                    <Layers size={18} /> Domain Router Telemetry
                  </h4>
                  
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
                      <span style={{ color: '#94a3b8' }}>Detected Intent:</span>
                      <span style={{ fontWeight: '700', color: '#38bdf8' }}>{result.intent.type}</span>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
                      <span style={{ color: '#94a3b8' }}>Confidence:</span>
                      <span style={{ fontWeight: '600', color: '#34d399' }}>{(result.intent.confidence * 100).toFixed(0)}%</span>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
                      <span style={{ color: '#94a3b8' }}>Route Target:</span>
                      <span style={{ fontWeight: '600', color: '#c084fc' }}>{result.routing}</span>
                    </div>
                  </div>
                </section>

                {/* Multi-Stage RAG Funnel Progress */}
                <section className="glass-panel" style={{ padding: '24px', marginBottom: '24px' }}>
                  <h4 style={{ fontSize: '15px', fontWeight: '600', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px', color: '#818cf8' }}>
                    <Database size={18} /> Multi-Stage RAG Funnel
                  </h4>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '13px' }}>
                    <div style={{ padding: '10px 14px', borderRadius: '8px', background: 'rgba(56, 189, 248, 0.08)', border: '1px solid rgba(56, 189, 248, 0.2)' }}>
                      <strong>1. Retrieval Pool:</strong> 10+ Chunks Extracted
                    </div>
                    <div style={{ padding: '10px 14px', borderRadius: '8px', background: 'rgba(129, 140, 248, 0.08)', border: '1px solid rgba(129, 140, 248, 0.2)' }}>
                      <strong>2. Embedding Filter:</strong> Top-5 Chunks Filtered
                    </div>
                    <div style={{ padding: '10px 14px', borderRadius: '8px', background: 'rgba(192, 132, 252, 0.08)', border: '1px solid rgba(192, 132, 252, 0.2)' }}>
                      <strong>3. Cross-Encoder Rerank:</strong> Top-3 Chunks Distilled
                    </div>
                  </div>
                </section>

                {/* 4D Verification & SAC Reward */}
                <section className="glass-panel" style={{ padding: '24px' }}>
                  <h4 style={{ fontSize: '15px', fontWeight: '600', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px', color: '#34d399' }}>
                    <ShieldCheck size={18} /> 4D Verifier & SAC Policy
                  </h4>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '13px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#94a3b8' }}>Verifier Score:</span>
                      <span style={{ fontWeight: '700', color: '#34d399' }}>{result.verification.score.toFixed(2)}</span>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#94a3b8' }}>SAC Policy Reward R(s,a):</span>
                      <span style={{ fontWeight: '700', color: '#fbbf24' }}>+{result.sac_reward.toFixed(2)}</span>
                    </div>
                  </div>
                </section>
              </div>

            </div>
          )}
        </main>
      )}

      {/* ── TAB 2: MCP TOOLS STUDIO ─────────────────────────────────────────── */}
      {activeTab === 'mcp' && (
        <main className="glass-panel" style={{ padding: '32px' }}>
          <h2 style={{ fontSize: '20px', fontWeight: '700', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Code2 size={24} color="#38bdf8" /> Model Context Protocol (MCP) Tools Playground
          </h2>
          <p style={{ fontSize: '14px', color: '#94a3b8', marginBottom: '24px' }}>
            Execute standard JSON-RPC 2.0 MCP tool calls targeting GeeksforGeeks, LeetCode, and Codeforces.
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '24px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '13px', color: '#cbd5e1', marginBottom: '8px', fontWeight: '600' }}>
                Select MCP Tool:
              </label>
              <select
                value={mcpTool}
                onChange={(e) => setMcpTool(e.target.value)}
                style={{ width: '100%', padding: '12px 16px', borderRadius: '10px', background: 'rgba(15, 23, 42, 0.9)', border: '1px solid var(--border-glow)', color: '#f8fafc', outline: 'none' }}>
                <option value="search_leetcode_solution">search_leetcode_solution (LeetCode)</option>
                <option value="search_geeksforgeeks_solution">search_geeksforgeeks_solution (GeeksforGeeks)</option>
                <option value="search_codeforces_solution">search_codeforces_solution (Codeforces)</option>
                <option value="mcp_web_rag_coding_search">mcp_web_rag_coding_search (Unified Platform Search)</option>
              </select>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '13px', color: '#cbd5e1', marginBottom: '8px', fontWeight: '600' }}>
                Query / Problem Parameter:
              </label>
              <input
                type="text"
                value={mcpArg}
                onChange={(e) => setMcpArg(e.target.value)}
                placeholder="e.g. Longest Substring, Two Sum, Quicksort"
                style={{ width: '100%', padding: '12px 16px', borderRadius: '10px', background: 'rgba(15, 23, 42, 0.9)', border: '1px solid var(--border-glow)', color: '#f8fafc', outline: 'none' }}
              />
            </div>
          </div>

          <button
            onClick={handleMcpCall}
            disabled={mcpLoading}
            style={{ padding: '12px 28px', borderRadius: '10px', border: 'none', background: 'linear-gradient(135deg, #38bdf8, #818cf8)', color: '#090d16', fontWeight: '700', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '28px' }}>
            {mcpLoading ? <RefreshCw className="animate-spin" size={16} /> : <Play size={16} />} Execute MCP JSON-RPC Request
          </button>

          {mcpResult && (
            <div>
              <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '12px' }}>JSON-RPC Response Payload:</h3>
              <pre style={{ background: '#090d16', padding: '20px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)', fontSize: '13px', color: '#34d399', overflowX: 'auto' }}>
                {JSON.stringify(mcpResult, null, 2)}
              </pre>
            </div>
          )}
        </main>
      )}

    </div>
  );
}
