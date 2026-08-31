"""
OmniKnowledge 2.0 - Entity-Relation-Entity Knowledge Graph Engine
================================================================
Provides graph-based relational modeling, multi-hop path traversal, and
hybrid Graph-RAG fusion to complement dense vector retrieval.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import re
import json


@dataclass
class KnowledgeTriple:
    subject: str
    predicate: str  # e.g., 'permits', 'uses', 'differs_from', 'violates', 'implements', 'causes', 'optimizes'
    object_: str
    confidence: float = 1.0
    source_chunk_id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityNode:
    entity_id: str
    entity_type: str  # 'Protocol', 'Algorithm', 'Anomaly', 'Concept', 'System', 'Metric'
    name: str
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)


class OmniKnowledgeGraph:
    """
    In-memory / Persistent Entity-Relation-Entity Graph Engine.
    Enables graph traversal, relational path discovery, and fusion with text retrieval.
    """

    def __init__(self):
        self.nodes: Dict[str, EntityNode] = {}
        self.adjacency: Dict[str, List[KnowledgeTriple]] = {}
        self.reverse_adjacency: Dict[str, List[KnowledgeTriple]] = {}
        self._seed_foundational_domain_knowledge()

    def add_node(self, node: EntityNode):
        self.nodes[node.entity_id.lower()] = node
        if node.entity_id.lower() not in self.adjacency:
            self.adjacency[node.entity_id.lower()] = []
        if node.entity_id.lower() not in self.reverse_adjacency:
            self.reverse_adjacency[node.entity_id.lower()] = []

    def add_triple(self, triple: KnowledgeTriple):
        sub_id = triple.subject.lower()
        obj_id = triple.object_.lower()

        if sub_id not in self.adjacency:
            self.adjacency[sub_id] = []
        if obj_id not in self.reverse_adjacency:
            self.reverse_adjacency[obj_id] = []

        # Avoid duplicate edges
        for existing in self.adjacency[sub_id]:
            if existing.predicate == triple.predicate and existing.object_.lower() == obj_id:
                return

        self.adjacency[sub_id].append(triple)
        self.reverse_adjacency[obj_id].append(triple)

    def _seed_foundational_domain_knowledge(self):
        """Seeds foundational distributed systems, database theory, and CS knowledge triples."""
        triples = [
            # Snapshot Isolation & Concurrency
            KnowledgeTriple("Snapshot Isolation", "uses", "MVCC", 1.0),
            KnowledgeTriple("Snapshot Isolation", "permits", "Write Skew", 1.0),
            KnowledgeTriple("Snapshot Isolation", "differs_from", "Serializable Isolation", 1.0),
            KnowledgeTriple("Serializable Snapshot Isolation", "detects", "rw-Antidependency Cycle", 1.0),
            KnowledgeTriple("Serializable Snapshot Isolation", "aborts_to_prevent", "Non-Serializable Execution", 1.0),
            KnowledgeTriple("Serializable Snapshot Isolation", "guarantees", "Full Serializability", 1.0),
            KnowledgeTriple("Write Skew", "is_anomaly_of", "Snapshot Isolation", 1.0),
            KnowledgeTriple("Write Skew", "caused_by", "Concurrent Overlapping Predicate Reads and Writes", 1.0),
            
            # Distributed Protocols
            KnowledgeTriple("Two-Phase Commit (2PC)", "coordinates", "Distributed Atomic Transactions", 1.0),
            KnowledgeTriple("Two-Phase Commit (2PC)", "has_vulnerability", "Blocking on Coordinator Failure", 1.0),
            KnowledgeTriple("Three-Phase Commit (3PC)", "mitigates", "Blocking via Pre-Commit State", 0.95),
            KnowledgeTriple("Paxos", "provides", "Consensus with Majority Quorum", 1.0),
            KnowledgeTriple("Raft", "provides", "Understandable Leader-Based Consensus", 1.0),
            KnowledgeTriple("Replication Factor", "improves", "Fault Tolerance", 1.0),
            KnowledgeTriple("Replication Factor", "increases", "Write Coordination Overhead", 1.0),
            
            # Database Normalization
            KnowledgeTriple("1NF", "requires", "Atomic Column Values", 1.0),
            KnowledgeTriple("2NF", "prohibits", "Partial Dependency on Candidate Key", 1.0),
            KnowledgeTriple("3NF", "prohibits", "Transitive Dependency on Non-Key Attribute", 1.0),
            KnowledgeTriple("BCNF", "requires", "Determinant is Superkey", 1.0),
            
            # Indexing & Query Optimization
            KnowledgeTriple("B-Tree Index", "efficiently_supports", "Range Predicates and Prefix LIKE", 1.0),
            KnowledgeTriple("B-Tree Index", "invalidated_by", "Function Call on Indexed Column", 1.0),
            KnowledgeTriple("Cost-Based Optimizer", "selects", "Single Selective Index over Index Merge when Cheaper", 0.95),
            KnowledgeTriple("NOT EXISTS", "implements", "Anti-Join Filtering for Non-Matching Rows", 1.0),

            # History & Numeral Systems
            KnowledgeTriple("Ancient Indian Mathematics", "developed", "Decimal Place-Value System with Zero Symbol", 1.0),
            KnowledgeTriple("Babylonian Numeral System", "used", "Sexagesimal Base-60 Positional System", 1.0),
            KnowledgeTriple("Egyptian Numerals", "used", "Hieroglyphic Non-Positional Base-10 Additive System", 1.0)
        ]

        for t in triples:
            self.add_triple(t)
            # Ensure entity nodes exist
            if t.subject.lower() not in self.nodes:
                self.add_node(EntityNode(entity_id=t.subject, entity_type="Concept", name=t.subject))
            if t.object_.lower() not in self.nodes:
                self.add_node(EntityNode(entity_id=t.object_, entity_type="Concept", name=t.object_))

    def extract_entities_from_text(self, text: str) -> List[str]:
        """Extracts recognized graph entities from arbitrary input text."""
        text_low = text.lower()
        matched: Set[str] = set()

        for entity_id, node in self.nodes.items():
            if re.search(r'\b' + re.escape(entity_id) + r'\b', text_low):
                matched.add(node.name)
            for alias in node.aliases:
                if re.search(r'\b' + re.escape(alias.lower()) + r'\b', text_low):
                    matched.add(node.name)

        return list(matched)

    def traverse_subgraph(self, entity_name: str, max_depth: int = 2) -> List[KnowledgeTriple]:
        """Traverses multi-hop relations starting from an entity."""
        entity_key = entity_name.lower()
        visited_nodes: Set[str] = {entity_key}
        queue: List[Tuple[str, int]] = [(entity_key, 0)]
        collected_triples: List[KnowledgeTriple] = []

        while queue:
            curr, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            for triple in self.adjacency.get(curr, []):
                collected_triples.append(triple)
                obj_key = triple.object_.lower()
                if obj_key not in visited_nodes:
                    visited_nodes.add(obj_key)
                    queue.append((obj_key, depth + 1))

            for triple in self.reverse_adjacency.get(curr, []):
                collected_triples.append(triple)
                sub_key = triple.subject.lower()
                if sub_key not in visited_nodes:
                    visited_nodes.add(sub_key)
                    queue.append((sub_key, depth + 1))

        return collected_triples

    def query_graph_context(self, text: str, max_triples: int = 8) -> Dict[str, Any]:
        """
        Extracts relevant graph triples and relational facts given an input query or option text.
        """
        entities = self.extract_entities_from_text(text)
        all_triples: List[KnowledgeTriple] = []

        for entity in entities:
            subgraph = self.traverse_subgraph(entity, max_depth=2)
            all_triples.extend(subgraph)

        # Deduplicate
        unique_triples: List[KnowledgeTriple] = []
        seen = set()
        for t in all_triples:
            key = (t.subject.lower(), t.predicate.lower(), t.object_.lower())
            if key not in seen:
                seen.add(key)
                unique_triples.append(t)

        formatted_facts = [
            f"{t.subject} --[{t.predicate}]--> {t.object_}"
            for t in unique_triples[:max_triples]
        ]

        return {
            "matched_entities": entities,
            "triples_count": len(unique_triples),
            "relational_facts": formatted_facts,
            "raw_triples": [
                {"subject": t.subject, "predicate": t.predicate, "object": t.object_, "confidence": t.confidence}
                for t in unique_triples[:max_triples]
            ]
        }


# Global singleton instance
_GLOBAL_KG = OmniKnowledgeGraph()


def get_knowledge_graph() -> OmniKnowledgeGraph:
    return _GLOBAL_KG
