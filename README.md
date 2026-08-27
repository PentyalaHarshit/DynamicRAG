# End-to-End Hierarchical Hybrid RAG Pipeline (OmniAgentAI Architecture)

## Current architecture split

The implementation now follows this division of responsibility:

- Router / graph orchestration: `graph.py`
- Web RAG execution: `web_rag.py`
- Vector RAG execution: `vector_store.py`
- Embedding + cross-encoder retrieval: `reranker.py`
- Evidence selection: `answerability_agent.py`, `dqn_selector.py`, `sentence_retriever.py`
- Generation and verification: `generator.py`, `verifier.py`, `llm_client.py`
- Optimization logging: `sac_learning.py`

The router uses a direct LLM branch for `MATH`, `CODING`, and `REASONING`
queries, so simple non-retrieval tasks do not pay the web/vector/reranker
cost. Plain named-person questions such as "Who is Neeraj Chopra?" are treated
as `BIOGRAPHY`, not `CURRENT_FACT`, so they use local RAG first and web only as
fallback. `CURRENT_FACT` remains reserved for current role-holder, price, rate,
latest, today, and similar live-data queries.

## Architecture Flow

```
                               User Query
                                    │
                                    ▼
                        Intent Detection Agent
                                    │
                                    ▼
                              Router Agent
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
            Traditional RAG                    Web RAG
                    │                               │
             Vector Database                  ReAct Planner
                    │                               │
                    │                         MCP Tool Calling
                    │                               │
                    │                    Google / Search APIs
                    │                               │
                    │                         Web Retrieval
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                      Initial Top-10 Candidate Chunks
                                    │
                                    ▼
                     Embedding Similarity Filtering
                                    │
                                    ▼
                               Top-5 Chunks
                                    │
                                    ▼
                       Cross-Encoder QA Reranker
                                    │
                                    ▼
                               Top-3 Chunks
                                    │
                                    ▼
                         Answerability Agent
                                    │
               ┌────────────────────┴────────────────────┐
               │                                         │
          Answer Found                             Answer Missing
               │                                         │
               ▼                                         ▼
      Rich DQN Chunk Selector                  Query Expansion
               │                                         │
               ▼                                         ▼
    Fine-Grained Sentence Retrieval             Web Search Again
               │                                         │
               └────────────────────┬────────────────────┘
                                    │
                                    ▼
                           Evidence Gate
                     ┌────────────┴────────────┐
                     │                         │
                  PASS                       FAIL
                     │                         │
                     ▼                         ▼
              LLM Generation           Retrieve Again /
                     │                 Return "No Evidence"
                     ▼
          Verification & Self-Correction
                     │
                     ▼
              Reward Computation
                     │
             ┌───────┴────────┐
             │                │
             ▼                ▼
      Positive Reward   Negative Reward
             │                │
             ▼                ▼
        SAC Policy Update   SAC Policy Update
                     │
                     ▼
          Memory / Traditional RAG Update
                     │
                     ▼
               Final Response
```

## Answer memory

After an answer passes verification, `main.py` writes the (question, answer)
pair back into the same Chroma collection used for traditional RAG
(`TraditionalRAG.add_qa_memory`, in `vector_store.py`), tagged with
`source: "memory:web_rag"` or `"memory:traditional_rag"`. The next time
someone asks that question - or anything close enough in embedding space to
clear `TRADITIONAL_RAG_CONFIDENCE_THRESHOLD` - the router's normal
`has_confident_answer` check matches it and answers straight from memory,
skipping web search entirely. `main.py` labels this route `"memory"` in the
response so you can tell it apart from a real KB hit or a fresh web lookup.

Two things worth knowing:
- Only *verified* answers get stored (`verified=result["passed"]`) - an
  answer that failed the verifier's hallucination/relevance check is not
  written back, so bad answers can't get served with false confidence later.
- There's no expiry on memory entries yet. For fast-changing facts (prices,
  current office-holders, scores) you'll want to either skip storing those
  categories of question, or add a TTL check against the `stored_at`
  timestamp already in the metadata before trusting a memory hit.

## Setup

```bash
pip install -r requirements.txt
ollama pull phi3        # or any local model, update OLLAMA_MODEL in config.py
export GOOGLE_API_KEY=your_key
export GOOGLE_CSE_ID=your_cse_id
python main.py
```

You'll need a Google Programmable Search Engine (CSE) ID and API key from the
Google Cloud console - the free tier gives 100 queries/day.

## What changed from the original design, and why

**Dropped: SAC/DQN with up to 1000 retries as the correction mechanism.**
SAC and DQN are reinforcement learning algorithms trained over thousands of
episodes against a stable environment/reward function, run *offline*. They
are not built to converge synchronously inside a single user request - a
person is not going to wait through 1000 regeneration attempts. What replaced
it is a **bounded reflection loop** (`verifier.py`): generate an answer,
verify it against the source chunk for hallucination and relevance, and if it
fails, regenerate once or twice with the verifier's specific complaint fed
back into the prompt, capped at `config.MAX_SELF_CORRECTION_RETRIES` (default
3). This is the same self-correction *behavior* you wanted, without needing
an RL training loop live in the request path.

**Kept, for later: episode logging.** Every attempt (question, context,
answer, verifier score) is appended to `data/episodes.jsonl`. That log is the
right raw material if you want to add real RL later - e.g. periodically
running an offline DPO/RLHF-style fine-tune of the generator or reranker
using verifier pass/fail as the reward signal. That would be a separate
scheduled training job, not something invoked per-query.

**Dropped: DQN to pick 1 of 5 chunks.** Picking the best of a fixed set of
5 candidates against a scoring function is exactly what a cross-encoder
reranker does deterministically (`reranker.py`) - no training loop needed
for this step, since there's no sequential decision-making, just a scoring
function applied 5 times.

**Kept as designed:** router (traditional RAG confidence check before
falling back to web), ReAct before search, MCP-style search tool, top-5
retrieval, CrewAI analyzer+validator stage, verification agent.

## Known gaps / next steps
- `vector_store.py` assumes you've already populated the Chroma collection;
  add your own ingestion script for your knowledge base.
- Error handling on the Google API and page-fetch calls is minimal (empty
  results and fetch failures degrade gracefully, but rate-limit handling and
  retries with backoff are not implemented).
- `crew_validator.py` calls a local Ollama model for both agents - swap `llm=`
  to any CrewAI-supported provider if you want a bigger model for this step.
- No caching layer - repeat questions re-run the full pipeline. Worth adding
  a cache keyed on the question once you have real traffic.
