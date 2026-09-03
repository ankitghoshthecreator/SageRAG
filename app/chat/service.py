"""
Chat Service
============
Orchestrates:
  1. Session and message management (DB CRUD)
  2. Full RAG pipeline: retrieve → rerank → build prompt → LLM
  3. Langfuse tracing
  4. Streaming support

All database writes are synchronous (SQLAlchemy sync session).
The async `chat_stream` helper is used only for SSE streaming.
"""
from __future__ import annotations

import datetime
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.database.models import ChatSession, Message
from app.retrieval.service import retrieve
from app.reranker.service import rerank
from app.llm.service import llm_service
from app.chat.prompt_builder import build_rag_prompt, extract_citations
from app.monitoring.langfuse_client import trace_rag_query
from app.utils.config import settings

logger = logging.getLogger("sagerag.chat")


# ── Session management ─────────────────────────────────────────────────────────

def create_session(db: Session, user_id: int, title: str = "New Chat") -> ChatSession:
    """Create a new chat session for the user."""
    session = ChatSession(
        id=str(uuid.uuid4()),
        title=title,
        user_id=user_id,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    logger.info(f"Created chat session {session.id} for user {user_id}")
    return session


def list_sessions(db: Session, user_id: int) -> List[ChatSession]:
    """Return all sessions for the user, newest first."""
    return (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )


def get_session(db: Session, session_id: str, user_id: int) -> Optional[ChatSession]:
    """Fetch a single session, ensuring ownership."""
    return (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .first()
    )


def delete_session(db: Session, session_id: str, user_id: int) -> bool:
    """Delete a session and all its messages. Returns True on success."""
    session = get_session(db, session_id, user_id)
    if not session:
        return False
    db.delete(session)
    db.commit()
    logger.info(f"Deleted session {session_id}")
    return True


def update_session_title(
    db: Session, session_id: str, user_id: int, title: str
) -> Optional[ChatSession]:
    """Rename a chat session."""
    session = get_session(db, session_id, user_id)
    if not session:
        return None
    session.title = title
    session.updated_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(session)
    return session


# ── Message management ─────────────────────────────────────────────────────────

def get_messages(db: Session, session_id: str, user_id: int) -> List[Message]:
    """Return all messages in a session, chronologically."""
    session = get_session(db, session_id, user_id)
    if not session:
        return []
    return (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .all()
    )


def _save_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    citations: Optional[List[Dict]] = None,
) -> Message:
    msg = Message(
        session_id=session_id,
        role=role,
        content=content,
        citations=citations,
    )
    db.add(msg)
    # Touch session updated_at
    db.query(ChatSession).filter(ChatSession.id == session_id).update(
        {"updated_at": datetime.datetime.utcnow()}
    )
    db.commit()
    db.refresh(msg)
    return msg


def _get_history(db: Session, session_id: str) -> List[Dict[str, str]]:
    """Return recent message history as plain dicts for the prompt builder."""
    messages = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    return [{"role": m.role, "content": m.content} for m in messages]


# ── RAG pipeline ───────────────────────────────────────────────────────────────

def _run_rag_pipeline(
    query: str,
    history: List[Dict[str, str]],
    filter_doc_id: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run the full pipeline: retrieve → rerank → build prompt → generate.

    Returns (answer, reranked_chunks, citations)
    """
    # 1. Hybrid retrieval
    candidates = retrieve(query=query, filter_doc_id=filter_doc_id)

    # 2. Cross-encoder reranking
    reranked = rerank(query=query, chunks=candidates)

    # 3. Build prompt
    messages = build_rag_prompt(query=query, chunks=reranked, history=history)

    # 4. Generate
    answer = llm_service.generate(messages)

    # 5. Build citations
    citations = extract_citations(reranked)

    return answer, reranked, citations


# ── Public chat API ────────────────────────────────────────────────────────────

def chat(
    db: Session,
    session_id: str,
    user_id: int,
    query: str,
    filter_doc_id: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Process a user message synchronously.

    Returns (answer_text, citations_list).
    """
    t0 = time.perf_counter()

    # Persist user message
    _save_message(db, session_id, role="user", content=query)

    # Retrieve conversation history (excluding the message just saved)
    history = _get_history(db, session_id)[:-1]  # exclude the user msg we just saved

    # Run pipeline
    answer, reranked, citations = _run_rag_pipeline(
        query=query, history=history, filter_doc_id=filter_doc_id
    )

    # Persist assistant message
    _save_message(db, session_id, role="assistant", content=answer, citations=citations)

    latency_ms = (time.perf_counter() - t0) * 1000

    # Langfuse trace (fire-and-forget)
    trace_rag_query(
        trace_id=str(uuid.uuid4()),
        user_id=str(user_id),
        query=query,
        retrieved_chunks=reranked,
        answer=answer,
        latency_ms=latency_ms,
        session_id=session_id,
    )

    logger.info(
        f"Chat: session={session_id} query_len={len(query)} "
        f"chunks={len(reranked)} latency={latency_ms:.0f}ms"
    )
    return answer, citations


async def chat_stream(
    db: Session,
    session_id: str,
    user_id: int,
    query: str,
    filter_doc_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Streaming version of chat().
    Yields SSE-formatted strings: 'data: <token>\n\n'
    The final event contains citation JSON: 'data: [CITATIONS]<json>\n\n'

    Note: The user message is saved before streaming starts; the assistant
    message is saved after the stream completes.
    """
    import json

    t0 = time.perf_counter()

    # Persist user message
    _save_message(db, session_id, role="user", content=query)
    history = _get_history(db, session_id)[:-1]

    # Retrieve + rerank (synchronous part)
    candidates = retrieve(query=query, filter_doc_id=filter_doc_id)
    reranked = rerank(query=query, chunks=candidates)
    messages = build_rag_prompt(query=query, chunks=reranked, history=history)
    citations = extract_citations(reranked)

    # Stream tokens
    full_answer_parts: List[str] = []
    async for token in llm_service.stream(messages):
        full_answer_parts.append(token)
        yield f"data: {token}\n\n"

    full_answer = "".join(full_answer_parts)

    # Persist assistant message after stream completes
    _save_message(db, session_id, role="assistant", content=full_answer, citations=citations)

    latency_ms = (time.perf_counter() - t0) * 1000

    # Send citations as final event
    yield f"data: [CITATIONS]{json.dumps(citations)}\n\n"
    yield "data: [DONE]\n\n"

    # Langfuse trace
    trace_rag_query(
        trace_id=str(uuid.uuid4()),
        user_id=str(user_id),
        query=query,
        retrieved_chunks=reranked,
        answer=full_answer,
        latency_ms=latency_ms,
        session_id=session_id,
    )

    logger.info(
        f"Stream: session={session_id} tokens={len(full_answer_parts)} "
        f"latency={latency_ms:.0f}ms"
    )
