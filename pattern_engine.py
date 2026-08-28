"""
Hierarchical Document and Problem Pattern Learning Engine

Implements:
1. Document Pattern Graph: Models structural transitions across equations & sections
   (Ansatz -> Vacuum Field Eq -> Differential Eqs -> Boundary Conditions -> Physical Solution).
2. Document Type Schemas: Textbook, Research Paper, Mathematical Proof, Physics Derivation.
3. PyTorch Neural Section Encoder f_theta(x_i) -> h_i computing section embeddings and
   transition probabilities P(x_{i+1} | x_i).
4. Pattern Matcher: Aligns problem pattern x with document pattern E to compute strategy
   probabilities P(S_i | x, E) and uncertainty metrics.
"""
import math
import re
from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Document Type Schemas & Pattern Chains
# ---------------------------------------------------------------------------

DOCUMENT_PATTERNS = {
    "PHYSICS_DERIVATION": [
        "assumptions_symmetry",     # Static, spherically symmetric
        "metric_ansatz",           # ds^2 = -e^(2nu) dt^2 + e^(2lambda) dr^2 + r^2 dOmega^2
        "vacuum_field_equations",   # G_uv = 0 or R_uv = 0
        "differential_equations",   # ODE integration step
        "boundary_conditions",      # r -> inf (Minkowski) & Newtonian limit (2GM/c^2)
        "physical_solution",        # Final metric tensor
    ],
    "MATHEMATICAL_PROOF": [
        "definition",
        "lemma",
        "theorem_statement",
        "proof_steps",
        "corollary",
    ],
    "RESEARCH_PAPER": [
        "problem_formulation",
        "related_work",
        "methodology",
        "experimental_results",
        "conclusion",
    ],
    "TEXTBOOK_EXPLANATION": [
        "concept_introduction",
        "qualitative_explanation",
        "governing_equations",
        "worked_example",
        "physical_interpretation",
    ],
}


# ---------------------------------------------------------------------------
# 2. PyTorch Neural Section Encoder & Relational Predictor
# ---------------------------------------------------------------------------

class DocumentSectionEncoder(nn.Module):
    """
    PyTorch Neural Section Encoder f_theta(x_i) -> h_i
    Maps document chunks & equation structures to latent pattern embeddings.
    """
    def __init__(self, input_dim: int = 16, embed_dim: int = 32):
        super(DocumentSectionEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.transition_head = nn.Linear(embed_dim, embed_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def predict_next_step(self, h_current: torch.Tensor) -> torch.Tensor:
        return self.transition_head(h_current)


class StrategyProbabilityNet(nn.Module):
    """
    Computes P(S_i | x, E) by combining problem pattern vectors with document pattern embeddings.
    """
    def __init__(self, pattern_dim: int = 32, num_strategies: int = 4):
        super(StrategyProbabilityNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(pattern_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_strategies)
        )

    def forward(self, pattern_embed: torch.Tensor) -> torch.Tensor:
        return self.classifier(pattern_embed)


# Global instantiated neural models
_SECTION_ENCODER = DocumentSectionEncoder()
_STRATEGY_NET = StrategyProbabilityNet()
_STRATEGIES = ["symbolic_derivation", "step_by_step_physics", "research_then_llm", "symbolic_solver"]


# ---------------------------------------------------------------------------
# 3. Document Pattern Graph & Feature Extractor
# ---------------------------------------------------------------------------

class DocumentPatternGraph:
    """
    Constructs a directed pattern graph representing section/equation transitions
    retrieved from documents.
    """
    def __init__(self, evidence_chunks: List[str]):
        self.chunks = evidence_chunks
        self.nodes: List[str] = []
        self.edges: List[Tuple[str, str]] = []
        self._build_graph()

    def _build_graph(self):
        text = " ".join(self.chunks).lower()
        if re.search(r'\b(static|spherical|symmetry)\b', text):
            self.nodes.append("assumptions_symmetry")
        if re.search(r'\b(ds\^?2|dt\^?2|dr\^?2|metric ansatz|line element)\b', text):
            self.nodes.append("metric_ansatz")
        if re.search(r'\b(einstein|g_\{?\\mu\\nu\}?|r_\{?\\mu\\nu\}?|vacuum|g_{\\mu\\nu}\s*=\s*0)\b', text):
            self.nodes.append("vacuum_field_equations")
        if re.search(r'\b(differential|integrate|derivative|d/dr|ode)\b', text):
            self.nodes.append("differential_equations")
        if re.search(r'\b(boundary|infinity|minkowski|newtonian|2gm)\b', text):
            self.nodes.append("boundary_conditions")
        if re.search(r'\b(schwarzschild metric|final solution|boxed)\b', text):
            self.nodes.append("physical_solution")

        # Build directed transition edges
        for i in range(len(self.nodes) - 1):
            self.edges.append((self.nodes[i], self.nodes[i+1]))

    def get_pattern_completeness(self) -> float:
        total_steps = len(DOCUMENT_PATTERNS["PHYSICS_DERIVATION"])
        return len(self.nodes) / float(total_steps)


# ---------------------------------------------------------------------------
# 4. Neural Strategy Prediction P(S_i | x, E)
# ---------------------------------------------------------------------------

def extract_evidence_features(question: str, evidence_text: str, domain: str, detected_patterns: List[str]) -> torch.Tensor:
    """
    Converts query, evidence text, domain, and detected patterns into a 16-dim neural input vector.
    """
    e_text = evidence_text.lower()
    q_text = question.lower()

    feat = [
        1.0 if "derivation" in q_text or "derive" in q_text else 0.0,
        1.0 if "spherical_symmetry" in detected_patterns or "spherical" in e_text else 0.0,
        1.0 if "vacuum_solution" in detected_patterns or "vacuum" in e_text else 0.0,
        1.0 if "field_equations" in detected_patterns or "einstein" in e_text else 0.0,
        1.0 if "ds^2" in e_text or "metric" in e_text else 0.0,
        1.0 if "ricci" in e_text or "g_mu" in e_text or "g_uv" in e_text else 0.0,
        1.0 if "boundary" in e_text or "minkowski" in e_text else 0.0,
        1.0 if "newtonian" in e_text or "2gm" in e_text else 0.0,
        1.0 if domain == "physics" else 0.0,
        1.0 if domain == "mathematics" else 0.0,
        min(1.0, len(evidence_text) / 2000.0),
        1.0 if "perturbation" in e_text else 0.0,
        1.0 if "conservation" in e_text else 0.0,
        1.0 if "integral" in e_text or "ode" in e_text else 0.0,
        1.0 if "schwarzschild" in q_text else 0.0,
        1.0 if "proven" in e_text or "proof" in e_text else 0.0,
    ]
    return torch.tensor([feat], dtype=torch.float32)


def predict_strategy_probabilities(
    question: str,
    evidence_text: str,
    domain: str,
    detected_patterns: List[str]
) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
    """
    Passes evidence representation through PyTorch neural network to compute:
    1. Strategy probability distribution P(S_i | x, E)
    2. Shannon Uncertainty Entropy H(P)
    3. Document Pattern Graph metadata
    """
    _SECTION_ENCODER.eval()
    _STRATEGY_NET.eval()

    graph = DocumentPatternGraph([evidence_text])

    with torch.no_grad():
        x_vector = extract_evidence_features(question, evidence_text, domain, detected_patterns)
        h_pattern = _SECTION_ENCODER.encode(x_vector)
        logits = _STRATEGY_NET(h_pattern)
        probs = F.softmax(logits, dim=-1).squeeze(0).tolist()

    prob_dict = {strat: round(p, 4) for strat, p in zip(_STRATEGIES, probs)}

    # Prior adjustments based on document pattern graph completeness
    if graph.get_pattern_completeness() > 0.5:
        prob_dict["symbolic_derivation"] = round(prob_dict.get("symbolic_derivation", 0.25) * 1.5, 4)

    # Normalize probability distribution P(S_i | x, E)
    total_p = sum(prob_dict.values())
    prob_dict = {k: round(v / total_p, 4) for k, v in prob_dict.items()}

    # Calculate uncertainty entropy H(P)
    entropy = -sum(p * math.log(p + 1e-9) for p in prob_dict.values())

    graph_meta = {
        "pattern_type": "PHYSICS_DERIVATION" if domain == "physics" else "GENERAL_EXPLANATION",
        "nodes": graph.nodes,
        "edges": graph.edges,
        "pattern_completeness": graph.get_pattern_completeness(),
    }

    return prob_dict, round(entropy, 4), graph_meta
