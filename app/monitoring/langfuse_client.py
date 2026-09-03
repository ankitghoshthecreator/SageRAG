"""
Langfuse Observability Client
==============================
Wraps the Langfuse Python SDK to trace RAG queries.
Safe no-op when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set.

Usage
-----
    from app.monitoring.langfuse_client import trace_rag_query

    trace_rag_query(
        trace_id="session-uuid",
        user_id="42",
        query="What is our refund policy?",
        retrieved_chunks=[...],
        answer="Based on the documents...",
        latency_ms=1234,
    )
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from app.utils.config import settings

logger = logging.getLogger("sagerag.monitoring")

_langfuse_instance = None
_langfuse_disabled = False  # set True on first failed init


def get_langfuse():
    """
    Return a Langfuse singleton, or None if credentials are not configured.
    Thread-safe enough for single-process FastAPI.
    """
    global _langfuse_instance, _langfuse_disabled

    if _langfuse_disabled:
        return None

    if _langfuse_instance is not None:
        return _langfuse_instance

    if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
        logger.info("Langfuse not configured (LANGFUSE_PUBLIC_KEY / SECRET_KEY missing). Tracing disabled.")
        _langfuse_disabled = True
        return None

    try:
        from langfuse import Langfuse

        _langfuse_instance = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_BASE_URL,
        )
        logger.info(f"Langfuse initialized (host={settings.LANGFUSE_BASE_URL})")
    except Exception as e:
        logger.warning(f"Langfuse init failed ({e}). Tracing disabled.")
        _langfuse_disabled = True
        return None

    return _langfuse_instance


def trace_rag_query(
    *,
    trace_id: str,
    user_id: str,
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    answer: str,
    latency_ms: float,
    session_id: Optional[str] = None,
) -> None:
    """
    Record a single RAG query trace in Langfuse.
    This call is fire-and-forget; any exception is swallowed so it never
    blocks the chat response.
    """
    lf = get_langfuse()
    if lf is None:
        return

    try:
        trace = lf.trace(
            id=trace_id,
            name="rag_query",
            user_id=str(user_id),
            session_id=session_id or trace_id,
            input={"query": query},
            output={"answer": answer},
            metadata={
                "retrieved_chunks": len(retrieved_chunks),
                "latency_ms": round(latency_ms, 2),
                "top_sources": [
                    {
                        "filename": c.get("filename", ""),
                        "page": c.get("page_number", ""),
                        "score": round(c.get("rerank_score", c.get("score", 0.0)), 4),
                    }
                    for c in retrieved_chunks[:5]
                ],
            },
        )

        # Retrieval span
        span = trace.span(
            name="hybrid_retrieval",
            input={"query": query},
            output={"num_chunks": len(retrieved_chunks)},
        )
        span.end()

        lf.flush()
        logger.debug(f"Langfuse trace recorded: {trace_id}")
    except Exception as e:
        logger.warning(f"Langfuse trace failed (non-fatal): {e}")
