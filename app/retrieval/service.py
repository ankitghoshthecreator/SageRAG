"""
Hybrid Retrieval Service
Combines:
  1. Dense vector search  (Qdrant / in-memory)
  2. Keyword/BM25 search  (in-memory index built from ingested chunks)
  3. Reciprocal Rank Fusion (RRF) to merge both ranked lists
"""
import logging
import math
from collections import defaultdict
from typing import List, Dict, Any, Optional

from app.utils.config import settings
from app.embeddings.service import embed_query
from app.retrieval.qdrant_store import vector_store

logger = logging.getLogger("sagerag.retrieval")

# ── Simple in-memory BM25-like keyword index ───────────────────────────────────
# Populated when documents are ingested via `index_chunks()`.

class _KeywordIndex:
    """
    Minimal BM25-inspired keyword index stored in RAM.
    Sufficient for < 100 k chunks; can be replaced with Elasticsearch later.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List[Dict[str, Any]] = []        # raw chunks
        self._tf: List[Dict[str, int]] = []          # term frequencies per doc
        self._df: Dict[str, int] = defaultdict(int)  # document frequencies
        self._avgdl: float = 1.0

    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        for chunk in chunks:
            tokens = self._tokenize(chunk["text"])
            tf: Dict[str, int] = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            self._docs.append(chunk)
            self._tf.append(dict(tf))
            for term in set(tf.keys()):
                self._df[term] += 1

        # Recompute average doc length
        if self._docs:
            self._avgdl = sum(
                sum(tf.values()) for tf in self._tf
            ) / len(self._tf)

    def remove_by_document_id(self, document_id: str):
        indices = [
            i for i, d in enumerate(self._docs)
            if d["document_id"] == document_id
        ]
        for i in reversed(indices):
            chunk = self._docs.pop(i)
            tf = self._tf.pop(i)
            for term in tf:
                self._df[term] = max(0, self._df[term] - 1)
        if self._docs:
            self._avgdl = sum(sum(tf.values()) for tf in self._tf) / len(self._tf)
        logger.info(f"Keyword index: removed {len(indices)} chunks for doc {document_id}")

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        if not self._docs:
            return []
        tokens = self._tokenize(query)
        N = len(self._docs)
        scores: List[float] = []

        for idx, (doc, tf) in enumerate(zip(self._docs, self._tf)):
            score = 0.0
            dl = sum(tf.values())
            for term in set(tokens):
                if term not in tf:
                    continue
                df = self._df.get(term, 0)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                tf_val = tf[term]
                tf_norm = (tf_val * (self.k1 + 1)) / (
                    tf_val + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                )
                score += idf * tf_norm
            scores.append(score)

        ranked = sorted(
            zip(scores, self._docs), key=lambda x: x[0], reverse=True
        )
        return [
            {**doc, "score": sc}
            for sc, doc in ranked[:top_k]
            if sc > 0
        ]


# Module-level keyword index singleton
keyword_index = _KeywordIndex()


# ── RRF fusion ─────────────────────────────────────────────────────────────────

def _rrf_fusion(
    dense_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion.
    Each result gets a score of 1 / (rank + k).
    Results from both lists are merged by chunk_id.
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    chunks_by_id: Dict[str, Dict[str, Any]] = {}

    for rank, result in enumerate(dense_results):
        cid = result["chunk_id"]
        rrf_scores[cid] += 1.0 / (rank + 1 + k)
        chunks_by_id[cid] = result

    for rank, result in enumerate(keyword_results):
        cid = result["chunk_id"]
        rrf_scores[cid] += 1.0 / (rank + 1 + k)
        chunks_by_id[cid] = result

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {**chunks_by_id[cid], "score": score}
        for cid, score in merged
    ]


# ── Public API ─────────────────────────────────────────────────────────────────

def index_chunks(chunks: List[Dict[str, Any]], vectors: List[List[float]]):
    """Index chunks into both Qdrant (dense) and keyword index (sparse)."""
    vector_store.upsert(chunks, vectors)
    keyword_index.add_chunks(chunks)
    logger.info(f"Indexed {len(chunks)} chunks in both dense and keyword stores.")


def retrieve(
    query: str,
    top_k_retrieval: Optional[int] = None,
    filter_doc_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: dense + keyword → RRF fusion.
    Returns `top_k_retrieval` ranked results (before reranking).
    """
    top_k = top_k_retrieval or settings.TOP_K_RETRIEVAL

    query_vector = embed_query(query)

    # Dense search
    dense_results = vector_store.search(
        query_vector=query_vector,
        top_k=top_k,
        filter_doc_id=filter_doc_id,
    )

    # Keyword search
    keyword_results = keyword_index.search(query=query, top_k=top_k)

    # Fuse
    fused = _rrf_fusion(dense_results, keyword_results)

    logger.info(
        f"Hybrid retrieval → dense={len(dense_results)}, "
        f"keyword={len(keyword_results)}, fused={len(fused)}"
    )
    return fused[:top_k]


def remove_document(document_id: str):
    """Remove all chunks for a document from both stores."""
    vector_store.delete_by_document_id(document_id)
    keyword_index.remove_by_document_id(document_id)
