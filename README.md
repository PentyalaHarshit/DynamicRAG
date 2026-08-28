# Hierarchical Multi-Stage Hybrid RAG Pipeline & Reinforcement Learning Architecture

A production-grade, state-of-the-art **Hierarchical Multi-Stage RAG Pipeline** with **Soft Actor-Critic (SAC)** policy learning, **Rich Deep Q-Networks (DQN)**, **Model Context Protocol (MCP)** Web RAG tools, and **Agentic AI Code Generation**.

---

## 🏛️ System Architecture Blueprint

```
                         USER QUERY
                              │
                              ▼
                    ┌──────────────────┐
                    │ Problem Analyzer │
                    │                  │
                    │ Domain           │
                    │ Topic            │
                    │ Complexity       │
                    │ Problem Pattern  │
                    │ Variables        │
                    │ Equations        │
                    └────────┬─────────┘
                             │
                             ▼
                       INTENT DETECTOR
     (TRAVEL | WEATHER | FINANCE | CURRENCY | CODING | MATH | FACTOID)
                             │
                             ▼
                        ROUTER AGENT
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
      Dedicated APIs     Direct LLM       Web RAG
    (Exchange / Weather)               (MCP Coding Tools)
             │               │               │
             └───────────────┼───────────────┘
                             ▼
                    Document Parsing & HTML
                  (Preserves <pre> & <code>)
                             │
                             ▼
                    Chunking + Embedding
                             │
                             ▼
                     QA Cross-Encoder
                             │
                             ▼
                       PATTERN ENGINE
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
        Mathematical     Conceptual      Coding / Platform
          Patterns        Patterns          Patterns
             │               │               │
             └───────────────┼───────────────┘
                             ▼
                  Strategy Selection (RL)
                    ┌────────┴────────┐
                    ▼                 ▼
             DQN (Discrete)     SAC (Continuous)
                    │                 │
                    └────────┬────────┘
                             ▼
                 Agentic Code Synthesizer
                             │
                             ▼
                     4D Verification
                    ┌────────┴────────┐
                    ▼                 ▼
                  PASS              FAIL
                    │                 │
                    ▼                 ▼
              Final Answer       Reward + Retry
                                      │
                                      ▼
                                 SAC Update
```

---

## 🚀 Key Modules & Capabilities

### 1. 🛠️ Model Context Protocol (MCP) Web RAG (`mcp_coding_rag.py`)
Standardized **JSON-RPC 2.0 MCP tools** designed to query live coding platforms via Web RAG:
- `search_leetcode_solution`: Target `site:leetcode.com` for LeetCode problems & Python/C++ code.
- `search_geeksforgeeks_solution`: Target `site:geeksforgeeks.org` for DSA tutorials and algorithm solutions.
- `search_codeforces_solution`: Target `site:codeforces.com` for competitive programming problems.
- `mcp_web_rag_coding_search`: Unified MCP Web RAG search across all platforms.

#### MCP JSON-RPC 2.0 Example:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search_leetcode_solution",
    "arguments": {
      "problem": "Longest Substring Without Repeating Characters",
      "language": "python"
    }
  }
}
```

---

### 2. ⚡ Agentic AI Code Generation Engine (`llm_client.py` & `agents/search_tool.py`)
- **HTML Code Block Preservation**: HTML extraction in `agents/search_tool.py` preserves `<pre>` and `<code>` elements from web pages, enabling direct code snippet extraction from GeeksforGeeks, LeetCode, and Codeforces.
- **Dynamic Code Synthesis**: Converts retrieved Web RAG context into production-ready executable Python code (` ```python ... ``` `), complete with problem technique breakdowns, time/space complexity analysis ($O(N)$), and source link citations.

---

### 3. 🎯 Multi-Domain Intent Detector & Router (`intent_detector.py` & `graph.py`)
Automatically detects domain intents and routes queries to optimal execution paths:
- **`CODING`**: Triggers Web RAG targeted at LeetCode, GeeksforGeeks, and Codeforces.
- **`CURRENCY`**: Executes live exchange rate calculations via Open Exchange Rates API.
- **`WEATHER`**: Fetches real-time weather metrics via Open-Meteo API.
- **`FINANCE`**: Fetches stock quotes via Yahoo Finance API.
- **`TRAVEL`**: Parses origin, destination, and dates for flight/travel queries.
- **`BIOGRAPHY` / `FACTOID`**: Triggers Wikipedia REST API retrieval.

---

### 4. 🔬 4D Verification Agent & SAC Policy Learning (`verifier.py` & `sac_learning.py`)
Evaluates answer quality across 4 explicit dimensions:
1. `retrieved_context_has_answer`: Context relevance check.
2. `answer_contains_entity`: Entity presence check.
3. `user_question_answered`: Answer completeness check.
4. `hallucination`: Verification against grounded evidence.

Calculates continuous reward $R(s, a)$ for Soft Actor-Critic (SAC) reinforcement learning:
$$R(s, a) = w_1 \cdot \text{Score} + w_2 \cdot \text{Verification} + w_3 \cdot \text{Efficiency} - \text{Penalty}$$

---

## 🧪 Automated Testing & Verification

Run the full automated unit test suite:
```powershell
python -u .\test_suite.py
```
```text
Ran 21 tests in 27.129s
OK (100% Pass Mark, 0 Failures, 0 Errors)
```

Run the 100-query benchmark regression suite:
```powershell
python .\run_regression_tests.py
```

---

## 💻 Terminal Execution Examples

### 1. LeetCode Coding Query
```powershell
python .\main.py "Write a python solution for Leetcode 3 Longest Substring Without Repeating Characters "
```

### 2. Live Currency Conversion Query
```powershell
python .\main.py "Convert 100 USD to EUR"
```

### 3. Live Weather Query
```powershell
python .\main.py "What is the weather in London right now?"
```

### 4. MCP Tools Self-Test
```powershell
python .\mcp_coding_rag.py
```
