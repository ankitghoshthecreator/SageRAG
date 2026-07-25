"""
Cross-Encoder Reranker
Uses sentence-transformers CrossEncoder to reorder retrieved chunks
by relevance to the query before passing them to the LLM.
Falls back to returning chunks sorted by their existing score if the
model cannot be loaded (e.g., no internet or restricted environment).
"""
import logging
from typing import List, Dict, Any, Optional

from app.utils.config import settings

logger = logging.getLogger("sagerag.reranker")

_reranker = None  # lazy-loaded CrossEncoder


def _get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker
    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading reranker model: {settings.RERANK_MODEL_NAME}")
        _reranker = CrossEncoder(settings.RERANK_MODEL_NAME, max_length=512)
        logger.info("Reranker model loaded.")
    except Exception as e:
        logger.warning(f"Could not load reranker model ({e}). Falling back to score-only ordering.")
        _reranker = None
    return _reranker


def rerank(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Rerank chunks against the query using the cross-encoder.

    Returns the top_k most relevant chunks with a `rerank_score` field added.
    If the cross-encoder is unavailable, returns chunks sorted by their
    existing retrieval score.
    """
    if not chunks:
        return []

    final_k = top_k or settings.TOP_K_RERANK
    reranker = _get_reranker()

    if reranker is None:
        # Fallback: return top_k by existing retrieval score
        sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)
        for i, c in enumerate(sorted_chunks):
            c["rerank_score"] = c.get("score", 0.0)
        return sorted_chunks[:final_k]

    # Build (query, passage) pairs
    pairs = [[query, c["text"]] for c in chunks]
    scores = reranker.predict(pairs, show_progress_bar=False)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    logger.info(
        f"Reranker: {len(chunks)} → {final_k} chunks  "
        f"(top score={ranked[0]['rerank_score']:.3f})"
    )
    return ranked[:final_k]
