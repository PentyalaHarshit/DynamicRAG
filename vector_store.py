"""
Three-Tier Memory Architecture
================================

Knowledge Memory  (knowledge_base collection)
  Stable ingested documents — textbooks, articles, curated KB docs.
  Written at ingest time.  Never mixed with ephemeral QA pairs.

QA Memory  (qa_memory collection)
  Verified (question, answer) pairs written back after a successful
  pipeline run.  Queried by the router as the fastest answer path:
  if an identical or near-identical question was answered before and
  verified, serve it from here without re-running retrieval.

Conversation Memory  (conversation_memory collection)
  Per-session turn history — (user_turn, assistant_turn) pairs stored
  in insertion order.  Not yet used for retrieval; reserved for future
  context window reconstruction, multi-turn QA, and RL replay.

RL Experience Memory  (data/sac_episodes.jsonl)
  State → Action → Reward → Next-State transitions.
  Written by sac_learning.py as a flat JSONL file.  Kept out of Chroma
  because it is append-only and read in batches for offline training.

                        ┌──────────────────────────────┐
                        │      MEMORY SYSTEM           │
                        │                              │
                        │  knowledge_base  (Chroma)    │
                        │  qa_memory       (Chroma)    │
                        │  conversation    (Chroma)    │
                        │  rl_experience   (JSONL)     │
                        └──────────────┬───────────────┘
                                       │ relevant context
                                       ▼
                                     LLM
"""
import time
import uuid
import re
from dataclasses import dataclass
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions

import config


@dataclass
class RetrievedChunk:
    text: str
    score: float
    source: str


class TraditionalRAG:
    """
    Manages the three Chroma memory collections.

    Public query interface combines knowledge_base + qa_memory results so
    the router only needs to call one object.  add_qa_memory writes only
    to qa_memory.  add_conversation_turn writes only to conversation_memory.
    """

    def __init__(self):
        self.client   = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )

        # ── Three separate collections ──────────────────────────────────
        self.kb_collection = self.client.get_or_create_collection(
            name=config.CHROMA_KB_COLLECTION,
            embedding_function=self.embed_fn,
        )
        self.qa_collection = self.client.get_or_create_collection(
            name=config.CHROMA_QA_MEMORY_COLLECTION,
            embedding_function=self.embed_fn,
        )
        self.conv_collection = self.client.get_or_create_collection(
            name=config.CHROMA_CONV_MEMORY_COLLECTION,
            embedding_function=self.embed_fn,
        )

        # Legacy alias so existing ingestion code that calls
        # trad_rag.collection.add(...) keeps working unchanged.
        self.collection = self.kb_collection

    # ------------------------------------------------------------------
    # Ingestion (knowledge base)
    # ------------------------------------------------------------------

    def add_documents(self, texts: List[str], sources: List[str], ids: List[str]):
        """Add documents to the knowledge base collection."""
        self.kb_collection.add(
            documents=texts,
            metadatas=[{"source": s} for s in sources],
            ids=ids,
        )

    # ------------------------------------------------------------------
    # Retrieval — searches knowledge_base then qa_memory, merges results
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 20) -> List[RetrievedChunk]:
        """
        Query both knowledge_base and qa_memory collections.
        Returns merged results sorted by similarity score (highest first).
        """
        chunks: List[RetrievedChunk] = []

        for coll, source_prefix in [
            (self.kb_collection,  "kb"),
            (self.qa_collection,  "memory"),
        ]:
            try:
                n = min(top_k, max(1, coll.count()))
                results = coll.query(query_texts=[question], n_results=n)
            except Exception:
                continue

            if not results.get("documents") or not results["documents"][0]:
                continue

            for doc, dist, meta in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            ):
                similarity = 1.0 / (1.0 + dist)
                src = meta.get("source", source_prefix)
                if coll == self.qa_collection and not _memory_matches_question(question, doc):
                    continue
                chunks.append(RetrievedChunk(text=doc, score=similarity, source=src))

        # Sort by score descending, return top_k
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:top_k]

    def has_confident_answer(self, question: str):
        """
        Returns (bool, best_chunk_or_None).
        Checks qa_memory first (fastest path), then kb.
        """
        # 1. Check qa_memory for a verified prior answer
        try:
            n = min(3, max(1, self.qa_collection.count()))
            qa_results = self.qa_collection.query(
                query_texts=[question], n_results=n
            )
            if qa_results.get("documents") and qa_results["documents"][0]:
                for doc, dist, meta in zip(
                    qa_results["documents"][0],
                    qa_results["distances"][0],
                    qa_results["metadatas"][0],
                ):
                    similarity = 1.0 / (1.0 + dist)
                    chunk = RetrievedChunk(
                        text=doc, score=similarity,
                        source=meta.get("source", "memory:")
                    )
                    if (
                        similarity >= config.TRADITIONAL_RAG_CONFIDENCE_THRESHOLD
                        and not _is_low_quality_memory_answer(doc)
                        and _memory_matches_question(question, doc)
                    ):
                        return True, chunk
        except Exception:
            pass

        # 2. Fall back to knowledge base
        kb_chunks = self._query_kb(question, top_k=3)
        if not kb_chunks:
            return False, None
        best = max(kb_chunks, key=lambda c: c.score)
        is_confident = (
            best.score >= config.TRADITIONAL_RAG_CONFIDENCE_THRESHOLD
            and _memory_matches_question(question, best.text)
        )
        return is_confident, (best if is_confident else None)

    def _query_kb(self, question: str, top_k: int = 20) -> List[RetrievedChunk]:
        """Query the knowledge base collection only."""
        chunks = []
        try:
            n = min(top_k, max(1, self.kb_collection.count()))
            results = self.kb_collection.query(query_texts=[question], n_results=n)
            if results.get("documents") and results["documents"][0]:
                for doc, dist, meta in zip(
                    results["documents"][0],
                    results["distances"][0],
                    results["metadatas"][0],
                ):
                    similarity = 1.0 / (1.0 + dist)
                    chunks.append(RetrievedChunk(
                        text=doc, score=similarity,
                        source=meta.get("source", "kb"),
                    ))
        except Exception:
            pass
        return chunks

    # ------------------------------------------------------------------
    # QA Memory write-back (verified answers only)
    # ------------------------------------------------------------------

    def add_qa_memory(
        self,
        question: str,
        answer: str,
        route: str,
        verified: bool = True,
    ):
        """
        Writes a verified (question, answer) pair to the qa_memory collection.
        Never touches the knowledge_base collection.

        Only stores if:
          - verified=True
          - answer is not a low-quality/sentinel string
        """
        if not verified:
            return
        if _is_low_quality_memory_answer(answer):
            return

        # Remove stale entries for the same question
        try:
            n = min(5, max(1, self.qa_collection.count()))
            existing = self.qa_collection.query(
                query_texts=[question], n_results=n
            )
            if existing and existing.get("ids") and existing["ids"][0]:
                to_delete = [
                    doc_id
                    for doc_id, dist in zip(
                        existing["ids"][0],
                        existing.get("distances", [[]])[0],
                    )
                    if dist < 0.35
                ]
                if to_delete:
                    self.qa_collection.delete(ids=to_delete)
        except Exception:
            pass

        doc_text = f"Q: {question}\nA: {answer}"
        self.qa_collection.add(
            documents=[doc_text],
            metadatas=[{
                "source":     f"memory:{route}",
                "question":   question,
                "stored_at":  time.time(),
            }],
            ids=[f"memory-{uuid.uuid4().hex}"],
        )
        print(f"[Memory] QA pair stored in qa_memory collection (route={route}).")

    # ------------------------------------------------------------------
    # Conversation Memory (turn history)
    # ------------------------------------------------------------------

    def add_conversation_turn(
        self,
        session_id: str,
        user_turn: str,
        assistant_turn: str,
    ):
        """
        Stores one (user, assistant) exchange in the conversation_memory collection.
        Keyed by session_id so multi-session contexts stay separate.
        Not currently used for retrieval — reserved for future context
        window reconstruction and RL replay.
        """
        doc_text = f"User: {user_turn}\nAssistant: {assistant_turn}"
        self.conv_collection.add(
            documents=[doc_text],
            metadatas=[{
                "session_id":  session_id,
                "stored_at":   time.time(),
                "user_turn":   user_turn[:200],
            }],
            ids=[f"conv-{session_id}-{uuid.uuid4().hex}"],
        )

    def get_conversation_history(
        self,
        session_id: str,
        last_n: int = 5,
    ) -> List[str]:
        """
        Retrieves the last_n conversation turns for a given session.
        Returns a list of "User: ...\nAssistant: ..." strings, oldest first.
        """
        try:
            results = self.conv_collection.get(
                where={"session_id": session_id},
                include=["documents", "metadatas"],
            )
            if not results or not results.get("documents"):
                return []
            pairs = list(zip(results["documents"], results["metadatas"]))
            pairs.sort(key=lambda x: x[1].get("stored_at", 0))
            return [doc for doc, _ in pairs[-last_n:]]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_low_quality_memory_answer(text: str) -> bool:
    """
    Reject sentinel / fallback / bare-entity answers before storing them.
    """
    answer = text
    if "\nA:" in text or " A:" in text:
        answer = text.split("A:", 1)[1].strip()

    # Reject LLM-unavailable sentinel strings
    if re.search(r'\[LLM unavailable|no response generated', answer, re.IGNORECASE):
        return True

    weak_patterns = (
        r'^Based on the available information,?\s+the answer is\s+.+\.?$',
        r'^Based on the available information[:.].+$',
        r'^.+\s+is the person identified by the retrieved (context|evidence)\.?$',
        r'^The USA declared independence on .+\.?$',
        r'^The event occurred on .+\.?$',
        r'^The answer is .+\.?$',
        r'^\[LLM unavailable',
    )
    if any(re.match(p, answer, flags=re.IGNORECASE) for p in weak_patterns):
        return True

    # Reject very short biography answers (< 80 words) so they are not served
    # as the authoritative answer for "Who is X?" queries.  A short cached
    # string like "Nikola Tesla was an inventor." forces the pipeline to re-run
    # full retrieval and produce a proper multi-paragraph response.
    word_count = len(answer.split())
    if word_count < 80:
        return True

    return False


def _parse_memory_qa(text: str):
    match = re.search(r'Q:\s*(.*?)\s+A:\s*(.*)$', text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return "", text.strip()
    return match.group(1).strip(), match.group(2).strip()


def _memory_matches_question(question: str, memory_text: str) -> bool:
    stored_question, stored_answer = _parse_memory_qa(memory_text)
    query_terms = _important_terms(question)
    if not query_terms:
        return True
    memory_terms = _important_terms(f"{stored_question} {stored_answer}")
    coverage = len(query_terms & memory_terms) / max(1, len(query_terms))
    return coverage >= 0.80


def _important_terms(text: str) -> set:
    stopwords = {
        "what", "who", "when", "where", "why", "how", "which", "is", "are",
        "was", "were", "the", "a", "an", "of", "to", "in", "on", "for",
        "and", "or", "by", "with", "about", "tell", "me", "does", "do",
        "did", "law", "laws",
    }
    terms = set()
    for raw in re.findall(r"[a-zA-Z][a-zA-Z']+", text.lower()):
        token = raw.replace("'s", "")
        if token in stopwords or len(token) < 3:
            continue
        if token.endswith("s") and len(token) > 4:
            token = token[:-1]
        terms.add(token)
    return terms


class TraditionalRAG:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            embedding_function=self.embed_fn,
        )

    def add_documents(self, texts: List[str], sources: List[str], ids: List[str]):
        self.collection.add(documents=texts, metadatas=[{"source": s} for s in sources], ids=ids)

    def query(self, question: str, top_k: int = 20) -> List[RetrievedChunk]:
        results = self.collection.query(query_texts=[question], n_results=top_k)
        chunks = []
        if not results["documents"] or not results["documents"][0]:
            return chunks
        for doc, dist, meta in zip(
            results["documents"][0], results["distances"][0], results["metadatas"][0]
        ):
            # chroma returns distance (lower = closer); convert to a 0-1 similarity score
            similarity = 1 / (1 + dist)
            chunks.append(RetrievedChunk(text=doc, score=similarity, source=meta.get("source", "kb")))
        return chunks

    def has_confident_answer(self, question: str):
        """
        Returns (bool, best_chunk_or_None). This is the router's decision point:
        confident -> stay on traditional RAG. Not confident -> fall back to web RAG.
        """
        chunks = self.query(question, top_k=3)
        if not chunks:
            return False, None
        best = max(chunks, key=lambda c: c.score)
        if best.source.startswith("memory:"):
            if _is_low_quality_memory_answer(best.text):
                return False, best
            if not _memory_matches_question(question, best.text):
                return False, best
        return best.score >= config.TRADITIONAL_RAG_CONFIDENCE_THRESHOLD, best

    def add_qa_memory(self, question: str, answer: str, route: str, verified: bool = True):
        """
        Writes a (question, answer) pair back into the store after it's been
        generated and verified, so a future close-paraphrase of this question
        is answered from memory instead of re-running web RAG.

        Only call this with answers that passed verification - writing a
        hallucinated or unverified answer into the KB would let a bad answer
        get served with false confidence next time it's matched.
        """
        if not verified:
            return
        if _is_low_quality_memory_answer(answer):
            return

        # Delete existing stale memory entries for the same or close paraphrase question
        try:
            existing = self.collection.query(query_texts=[question], n_results=5)
            if existing and existing.get("ids") and existing["ids"][0]:
                to_delete = []
                for doc_id, dist in zip(existing["ids"][0], existing.get("distances", [[]])[0]):
                    if dist < 0.35:  # Close match / same question
                        to_delete.append(doc_id)
                if to_delete:
                    self.collection.delete(ids=to_delete)
        except Exception:
            pass

        # Embed on the question text (what future queries will look like),
        # but store the full Q+A so it can be used directly as context.
        doc_text = f"Q: {question}\nA: {answer}"
        self.collection.add(
            documents=[doc_text],
            metadatas=[{"source": f"memory:{route}", "question": question, "stored_at": time.time()}],
            ids=[f"memory-{uuid.uuid4().hex}"],
        )


def _is_low_quality_memory_answer(text: str) -> bool:
    """
    Reject old memory entries that contain only a bare entity answer. Those
    entries pass entity checks but produce poor final answers for biography
    questions, so the router should retrieve fresh evidence instead.
    """
    answer = text
    if "\nA:" in text or " A:" in text:
        answer = text.split("A:", 1)[1].strip()

    weak_patterns = (
        r'^Based on the available information,?\s+the answer is\s+.+\.?$',
        r'^Based on the available information[:.].+$',
        r'^.+\s+is the person identified by the retrieved (context|evidence)\.?$',
        r'^The USA declared independence on .+\.?$',
        r'^The event occurred on .+\.?$',
        r'^The answer is .+\.?$',
    )
    return any(re.match(pattern, answer, flags=re.IGNORECASE) for pattern in weak_patterns)


def _parse_memory_qa(text: str) -> tuple[str, str]:
    """Extract stored question and answer from the memory document format."""
    match = re.search(r'Q:\s*(.*?)\s+A:\s*(.*)$', text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return "", text.strip()
    return match.group(1).strip(), match.group(2).strip()


def _memory_matches_question(question: str, memory_text: str) -> bool:
    """
    Prevent near-neighbor memory from answering a different question.
    Example: "Newton's law" memory should not satisfy "Newton's laws of motion"
    when the cached answer is actually about universal gravitation.
    """
    stored_question, stored_answer = _parse_memory_qa(memory_text)
    query_terms = _important_terms(question)
    if not query_terms:
        return True

    memory_terms = _important_terms(f"{stored_question} {stored_answer}")
    coverage = len(query_terms & memory_terms) / max(1, len(query_terms))
    return coverage >= 0.80


def _important_terms(text: str) -> set[str]:
    stopwords = {
        "what", "who", "when", "where", "why", "how", "which", "is", "are",
        "was", "were", "the", "a", "an", "of", "to", "in", "on", "for",
        "and", "or", "by", "with", "about", "tell", "me", "does", "do",
        "did", "law", "laws",
    }
    terms = set()
    for raw in re.findall(r"[a-zA-Z][a-zA-Z']+", text.lower()):
        token = raw.replace("'s", "")
        if token in stopwords or len(token) < 3:
            continue
        if token.endswith("s") and len(token) > 4:
            token = token[:-1]
        terms.add(token)
    return terms
