"""
Traditional RAG: local vector store (Chroma). This is the "fast path" -
if the KB already has a confident answer, we never touch the web.

This also doubles as answer memory: once a question has been answered
(whether the answer came from the KB or from web RAG), the (question, answer)
pair gets written back in here. The next time someone asks the same or a
close paraphrase of that question, it's answered straight from this store
instead of re-running web search.
"""
import time
import uuid
import re
from dataclasses import dataclass
from typing import List

import chromadb
from chromadb.utils import embedding_functions

import config


@dataclass
class RetrievedChunk:
    text: str
    score: float          # similarity score, higher = more relevant
    source: str


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
        r'^.+\s+is the person identified by the retrieved (context|evidence)\.?$',
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
