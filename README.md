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
                             │
                             ▼
                 Multi-Stage RAG Funnel
                (10 → Top-5 → Top-3 Chunks)
                             │
                             ▼
                    Document Parsing & HTML
                  (Preserves <pre> & <code>)
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

## 🔄 Multi-Stage Hierarchical RAG Funnel (10 → Top-5 → Top-3 Chunks)

The system employs a multi-stage progressive funnel to filter and rank retrieved evidence for **General Knowledge**, **Factoids**, **Science**, and **Coding** queries:

```
       ┌──────────────────────────────────────────────┐
       │     Candidate Retrieval Pool (10+ Chunks)     │
       │ Vector Store / BM25 Hybrid / Live Web Search │
       └──────────────────────┬───────────────────────┘
                              │
                              ▼
       ┌──────────────────────────────────────────────┐
       │  Stage 1: Embedding Similarity Filter        │
       │  Filters initial 10+ pool to Top-5 Chunks    │
       └──────────────────────┬───────────────────────┘
                              │
                              ▼
       ┌──────────────────────────────────────────────┐
       │  Stage 2: QA Cross-Encoder Reranker          │
       │  Reranks Top-5 down to Top-3 Best Chunks     │
       └──────────────────────┬───────────────────────┘
                              │
                              ▼
       ┌──────────────────────────────────────────────┐
       │  Stage 3: Rich DQN & Answerability Agent    │
       │  Selects optimal evidence chunk              │
       │  Passes Hard Evidence Gate & 4D Verifier     │
       └──────────────────────────────────────────────┘
```

### Funnel Stages Breakdown:
1. **Initial Candidate Retrieval Pool (10+ Chunks)**: Fetches an initial candidate pool of 10+ evidence chunks from ChromaDB vector store, BM25 keyword index, or live Web search fallback.
2. **Stage 1 — Embedding Similarity Filter (Top-5 Chunks)**: Calculates cosine similarity using SentenceTransformers (`all-MiniLM-L6-v2`) to prune noisy documents and isolate the **Top-5** candidate chunks.
3. **Stage 2 — QA Cross-Encoder Reranker (Top-3 Chunks)**: Evaluates deep question-context interactions using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`), reranking the Top-5 down to the **Top-3** highest confidence chunks.
4. **Stage 3 — Rich DQN Selector & Hard Evidence Gate**: Evaluates neural state vectors (similarity score, cross-encoder score, position, entity density) and selects the optimal chunk while enforcing the **Hard Evidence Gate** (`Evidence Gate Passed = True`).

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

### 2. General Knowledge Query
```powershell
python .\main.py "Explain the difference between REST API and GraphQL"
```

### 3. Live Currency Conversion Query
```powershell
python .\main.py "Convert 100 USD to EUR"
```

### 4. Live Weather Query
```powershell
python .\main.py "What is the weather in London right now?"
```

### 5. MCP Tools Self-Test
```powershell
python .\mcp_coding_rag.py
```
