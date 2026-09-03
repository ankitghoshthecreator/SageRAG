"""
Chat API Router
===============
Endpoints
---------
POST   /sessions                       Create a new chat session
GET    /sessions                       List user's sessions
PATCH  /sessions/{id}                  Rename a session
DELETE /sessions/{id}                  Delete session + messages
GET    /sessions/{id}/messages         Get full message history
POST   /sessions/{id}/chat             Send message, get RAG answer
POST   /sessions/{id}/chat/stream      Streaming SSE version
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.database.models import User
from app.auth.security import get_current_active_user
from app.chat import service as chat_service

router = APIRouter()


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    title: str = Field(default="New Chat", max_length=200)


class SessionUpdate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class SessionResponse(BaseModel):
    id: str
    title: str
    user_id: int
    created_at: datetime.datetime
    updated_at: datetime.datetime

    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    id: int
    session_id: str
    role: str
    content: str
    citations: Optional[List[Dict[str, Any]]] = None
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096, description="The user's question.")
    filter_doc_id: Optional[str] = Field(
        default=None,
        description="Optional document UUID to restrict retrieval to a single document.",
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: List[Dict[str, Any]]


# ── Session endpoints ──────────────────────────────────────────────────────────

@router.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new chat session",
)
def create_session(
    body: SessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create a new named chat session for the authenticated user."""
    session = chat_service.create_session(
        db=db, user_id=current_user.id, title=body.title
    )
    return session


@router.get(
    "/sessions",
    response_model=List[SessionResponse],
    summary="List all chat sessions",
)
def list_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Return all sessions for the authenticated user, newest first."""
    return chat_service.list_sessions(db=db, user_id=current_user.id)


@router.patch(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    summary="Rename a chat session",
)
def rename_session(
    session_id: str,
    body: SessionUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Rename an existing chat session."""
    updated = chat_service.update_session_title(
        db=db, session_id=session_id, user_id=current_user.id, title=body.title
    )
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return updated


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a chat session",
)
def delete_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a session and all its messages."""
    success = chat_service.delete_session(
        db=db, session_id=session_id, user_id=current_user.id
    )
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")


# ── Message history ────────────────────────────────────────────────────────────

@router.get(
    "/sessions/{session_id}/messages",
    response_model=List[MessageResponse],
    summary="Get message history for a session",
)
def get_messages(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Return the full message history for the given session."""
    # Verify ownership
    session = chat_service.get_session(db=db, session_id=session_id, user_id=current_user.id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return chat_service.get_messages(db=db, session_id=session_id, user_id=current_user.id)


# ── Chat endpoints ─────────────────────────────────────────────────────────────

@router.post(
    "/sessions/{session_id}/chat",
    response_model=ChatResponse,
    summary="Send a message and get a RAG answer",
)
def send_message(
    session_id: str,
    body: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Run the full RAG pipeline for the given query and return the answer
    with source citations.

    The user message and assistant response are persisted to the session
    history automatically.
    """
    # Verify ownership
    session = chat_service.get_session(db=db, session_id=session_id, user_id=current_user.id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    try:
        answer, citations = chat_service.chat(
            db=db,
            session_id=session_id,
            user_id=current_user.id,
            query=body.query,
            filter_doc_id=body.filter_doc_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat pipeline error: {str(e)}",
        )

    return ChatResponse(session_id=session_id, answer=answer, citations=citations)


@router.post(
    "/sessions/{session_id}/chat/stream",
    summary="Stream a RAG answer via Server-Sent Events",
    response_description="text/event-stream SSE stream",
)
async def stream_message(
    session_id: str,
    body: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Stream the RAG answer token-by-token using Server-Sent Events.

    Event format:
    - `data: <token>\\n\\n`  — each token fragment
    - `data: [CITATIONS]<json_array>\\n\\n` — citation list at end
    - `data: [DONE]\\n\\n`  — stream complete sentinel

    The user message and final assistant message are persisted automatically.
    """
    # Verify ownership
    session = chat_service.get_session(db=db, session_id=session_id, user_id=current_user.id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    return StreamingResponse(
        chat_service.chat_stream(
            db=db,
            session_id=session_id,
            user_id=current_user.id,
            query=body.query,
            filter_doc_id=body.filter_doc_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
