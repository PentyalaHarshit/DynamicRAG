# End-to-End Hierarchical Hybrid RAG Pipeline & Reinforcement Learning Architecture

## Architectural Blueprint

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
                             │
                             ▼
                        ROUTER AGENT
                             │
                             ▼
                      RESEARCH AGENT
                             │
                             ▼
                         Web RAG
                             │
                         MCP Tools
                             │
             ┌───────────────┼────────────────┐
             ▼               ▼                ▼
        Research papers   Publishers      Textbooks/PDFs
             │               │                │
             └───────────────┼────────────────┘
                             ▼
                    Relevant Documents
                             │
                             ▼
                      Document Parsing
                             │
                             ▼
                    Chunking + Embedding
                             │
                             ▼
                       Top-K Evidence
                             │
                             ▼
                    Evidence → Agent
                             │
                             ▼
                    PATTERN ENGINE
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
       Mathematical      Conceptual       Structural
         patterns          patterns         patterns
            │                │                │
            └────────────────┼────────────────┘
                             ▼
                    Pattern Representation
                             │
                             ▼
                 Neural Model / Learning
                       PyTorch / TF
                             │
                             ▼
                 Probability / Uncertainty
                   P(S_i | x, E)
                             │
                             ▼
                    Strategy Selection
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
             DQN (Discrete)     SAC (Continuous)
                    │                 │
                    └────────┬────────┘
                             ▼
                  Solver / Reasoning LLM
                             │
                             ▼
                         Solution
                             │
                             ▼
                       Verification
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
                  PASS              FAIL
                    │                 │
                    ▼                 ▼
              Final Answer       Reward + Retry
                                      │
                                      ▼
                                 RL Update
```

---

## Key System Principles

### 1. Hard Evidence Gate
$$\text{Zero Evidence / Filtered Chunks == 0} \implies \text{Evidence Gate FAIL} \implies \text{Negative Reward (-1.0)} \implies \text{Retry / Refusal}$$

The system never proceeds to generate a "verified answer" when zero chunks pass retrieval and filtering.

---

### 2. Pattern Representation & Uncertainty Estimation
Rather than passing raw text directly to neural models, the **Pattern Engine** converts retrieved evidence into structured representations:

$$\text{Retrieved Evidence} \rightarrow \text{Equations / Variables} \rightarrow \text{Assumptions} \rightarrow \text{Problem Structure} \rightarrow \text{Neural Representation} \rightarrow P(S_i \mid x, E)$$

where:
- $x = \text{User query / problem}$
- $E = \text{Retrieved evidence}$
- $S_i = \text{Candidate strategy}$

$$\text{Selected Strategy } S^* = \arg\max_i P(S_i \mid x, E)$$

---

### 3. Dual RL Framework (DQN + SAC)

- **DQN (Discrete Actions)**:
  $$a \in \{\text{Research Agent}, \text{Direct LLM}, \text{Symbolic Solver}, \text{Web RAG}, \text{Traditional RAG}, \text{Query Expansion}, \text{Retry}\}$$

- **SAC (Continuous Parameter Control)**:
  $$\theta = \{\text{retrieval\_depth}, \text{research\_depth}, \text{verification\_threshold}, \text{temperature}, \text{attempt\_budget}\}$$

---

### 4. Scientific Derivation Verification
For mathematical and physics derivation queries (e.g. *Derive the Schwarzschild solution*), the **Verification Agent** checks 5 explicit derivation milestones:
1. Metric ansatz / initial setup ($ds^2 = -e^{2\nu} c^2 dt^2 + e^{2\lambda} dr^2 + r^2 d\Omega^2$)
2. Vacuum field equations ($G_{\mu\nu} = 0$ or $R_{\mu\nu} = 0$)
3. Differential equations & integration steps ($\nu' + \lambda' = 0$)
4. Boundary conditions & Newtonian limit ($r \to \infty$, $g_{tt} \approx -(1 + 2\Phi/c^2)$)
5. Final explicit metric formula

Fewer than 2 milestones met $\implies \text{user\_question\_answered = False}$, $\text{incomplete\_derivation = True}$, Score $\le 0.20$, and Negative Reward ($-0.7$).

---

## Quantitative Evaluation Metrics

$$\text{Pattern Accuracy} \quad \Big| \quad \text{Strategy Selection Accuracy} \quad \Big| \quad \text{Evidence Retrieval Recall@K}$$
$$\text{Solution Correctness} \quad \Big| \quad \text{Verification Accuracy} \quad \Big| \quad \text{Retry Success Rate}$$
