"""
tests/test_chat.py
==================
Integration tests for the Chat API (Part 3).

These tests run against the actual FastAPI app with a SQLite test database
(the same pattern used by test_auth.py and test_ingestion.py).

Tests cover:
  - Session CRUD (create, list, rename, delete)
  - Message history retrieval
  - Chat endpoint (mocked LLM + retrieval to avoid network calls)
  - Streaming chat endpoint
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base
from app.database.connection import get_db
from app.main import app

# ── Test database setup ────────────────────────────────────────────────────────

TEST_DB_URL = "sqlite:///./test_sage_rag.db"

engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

# ── Helpers ────────────────────────────────────────────────────────────────────

_registered = False
_token: str = ""
_session_id: str = ""


def _auth_headers() -> dict:
    global _registered, _token
    if not _registered:
        client.post(
            "/api/v1/auth/register",
            json={"username": "chatuser", "email": "chat@test.com", "password": "testpass1"},
        )
        _registered = True
    if not _token:
        resp = client.post(
            "/api/v1/auth/login",
            data={"username": "chatuser", "password": "testpass1"},
        )
        assert resp.status_code == 200
        _token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {_token}"}


# ── Session tests ──────────────────────────────────────────────────────────────

class TestSessions:
    def test_create_session(self):
        global _session_id
        resp = client.post(
            "/api/v1/chat/sessions",
            json={"title": "Test Session"},
            headers=_auth_headers(),
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "Test Session"
        assert "id" in data
        _session_id = data["id"]

    def test_list_sessions(self):
        resp = client.get("/api/v1/chat/sessions", headers=_auth_headers())
        assert resp.status_code == 200
        sessions = resp.json()
        assert isinstance(sessions, list)
        assert len(sessions) >= 1

    def test_rename_session(self):
        resp = client.patch(
            f"/api/v1/chat/sessions/{_session_id}",
            json={"title": "Renamed Session"},
            headers=_auth_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["title"] == "Renamed Session"

    def test_rename_nonexistent_session_returns_404(self):
        resp = client.patch(
            "/api/v1/chat/sessions/nonexistent-id",
            json={"title": "Ghost"},
            headers=_auth_headers(),
        )
        assert resp.status_code == 404

    def test_get_messages_empty(self):
        resp = client.get(
            f"/api/v1/chat/sessions/{_session_id}/messages",
            headers=_auth_headers(),
        )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_messages_nonexistent_session_returns_404(self):
        resp = client.get(
            "/api/v1/chat/sessions/ghost-session/messages",
            headers=_auth_headers(),
        )
        assert resp.status_code == 404


# ── Chat endpoint tests ────────────────────────────────────────────────────────

MOCK_CHUNKS = [
    {
        "chunk_id": "c1",
        "document_id": "d1",
        "filename": "policy.pdf",
        "page_number": 1,
        "chunk_index": 0,
        "text": "Our refund policy allows returns within 30 days.",
        "score": 0.92,
        "rerank_score": 0.95,
    }
]


class TestChat:
    def test_chat_returns_answer_and_citations(self):
        with (
            patch("app.chat.service.retrieve", return_value=MOCK_CHUNKS),
            patch("app.chat.service.rerank", return_value=MOCK_CHUNKS),
            patch("app.chat.service.llm_service") as mock_llm,
            patch("app.chat.service.trace_rag_query"),
        ):
            mock_llm.generate.return_value = "You can return items within 30 days. [Source 1]"

            resp = client.post(
                f"/api/v1/chat/sessions/{_session_id}/chat",
                json={"query": "What is the refund policy?"},
                headers=_auth_headers(),
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == _session_id
        assert "30 days" in data["answer"]
        assert isinstance(data["citations"], list)
        assert len(data["citations"]) == 1
        assert data["citations"][0]["filename"] == "policy.pdf"

    def test_chat_message_persisted_in_history(self):
        resp = client.get(
            f"/api/v1/chat/sessions/{_session_id}/messages",
            headers=_auth_headers(),
        )
        assert resp.status_code == 200
        msgs = resp.json()
        # Should have user + assistant messages from the chat above
        roles = [m["role"] for m in msgs]
        assert "user" in roles
        assert "assistant" in roles

    def test_chat_nonexistent_session_returns_404(self):
        resp = client.post(
            "/api/v1/chat/sessions/ghost-session/chat",
            json={"query": "Hello?"},
            headers=_auth_headers(),
        )
        assert resp.status_code == 404

    def test_chat_stream_returns_event_stream(self):
        async def _fake_stream(*args, **kwargs):
            yield "data: Hello\n\n"
            yield "data: World\n\n"
            yield f"data: [CITATIONS]{json.dumps([])}\n\n"
            yield "data: [DONE]\n\n"

        with patch("app.chat.router.chat_service.chat_stream", side_effect=_fake_stream):
            resp = client.post(
                f"/api/v1/chat/sessions/{_session_id}/chat/stream",
                json={"query": "Stream test"},
                headers=_auth_headers(),
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_chat_empty_query_rejected(self):
        resp = client.post(
            f"/api/v1/chat/sessions/{_session_id}/chat",
            json={"query": ""},
            headers=_auth_headers(),
        )
        assert resp.status_code == 422  # Pydantic validation error


# ── Session delete (last — keeps ID valid for other tests) ─────────────────────

class TestSessionDelete:
    def test_delete_session(self):
        # Create a throwaway session to delete
        create_resp = client.post(
            "/api/v1/chat/sessions",
            json={"title": "To Delete"},
            headers=_auth_headers(),
        )
        sid = create_resp.json()["id"]

        del_resp = client.delete(f"/api/v1/chat/sessions/{sid}", headers=_auth_headers())
        assert del_resp.status_code == 204

        # Verify gone
        msgs_resp = client.get(f"/api/v1/chat/sessions/{sid}/messages", headers=_auth_headers())
        assert msgs_resp.status_code == 404

    def test_delete_nonexistent_session_returns_404(self):
        resp = client.delete(
            "/api/v1/chat/sessions/nonexistent",
            headers=_auth_headers(),
        )
        assert resp.status_code == 404
