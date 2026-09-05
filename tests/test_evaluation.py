"""
tests/test_evaluation.py
========================
Tests for the Evaluation API (Part 4).
"""
import uuid
import datetime
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base, ChatSession, Message
from app.database.connection import get_db
from app.utils.config import settings
from app.evaluation.router import router as evaluation_router

settings.EVALUATION_ENABLED = True

from app.main import app

# Ensure router is included (in case main.py was imported by other tests or before settings were changed)
if not any(r.prefix == f"{settings.API_V1_STR}/evaluation" for r in app.router.routes if hasattr(r, "prefix")):
    app.include_router(
        evaluation_router,
        prefix=f"{settings.API_V1_STR}/evaluation",
        tags=["Evaluation"],
    )

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

_registered = False
_token: str = ""
_user_id: int = 1

def _auth_headers() -> dict:
    global _registered, _token
    if not _registered:
        client.post(
            "/api/v1/auth/register",
            json={"username": "evaluser", "email": "eval@test.com", "password": "testpass1"},
        )
        _registered = True
    if not _token:
        resp = client.post(
            "/api/v1/auth/login",
            data={"username": "evaluser", "password": "testpass1"},
        )
        _token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {_token}"}

# Enable evaluation for tests
settings.EVALUATION_ENABLED = True

class TestEvaluation:
    @patch("app.evaluation.service.run_ragas")
    def test_run_manual_evaluation(self, mock_run_ragas):
        mock_run_ragas.return_value = {"faithfulness": 0.95, "answer_relevancy": 0.88}
        
        resp = client.post(
            "/api/v1/evaluation/run",
            json={
                "samples": [
                    {
                        "question": "What is the capital of France?",
                        "answer": "Paris",
                        "contexts": ["Paris is the capital of France."]
                    }
                ]
            },
            headers=_auth_headers(),
        )
        
        assert resp.status_code == 200
        data = resp.json()
        assert data["faithfulness"] == 0.95
        assert data["answer_relevancy"] == 0.88

    @patch("app.evaluation.service.run_ragas")
    def test_evaluate_session(self, mock_run_ragas):
        mock_run_ragas.return_value = {"faithfulness": 0.90}
        
        user_resp = client.get("/api/v1/auth/me", headers=_auth_headers())
        real_user_id = user_resp.json()["id"]

        # 1. Create a dummy session
        db = TestingSessionLocal()
        session_id = str(uuid.uuid4())
        session = ChatSession(id=session_id, title="Eval Test", user_id=real_user_id)
        db.add(session)
        
        # 2. Add some messages with context
        msg1 = Message(session_id=session_id, role="user", content="How do I get a refund?")
        msg2 = Message(
            session_id=session_id, 
            role="assistant", 
            content="You have 30 days.",
            citations=[{"text_preview": "Refunds take 30 days"}]
        )
        db.add(msg1)
        db.add(msg2)
        db.commit()
        db.close()
        
        # 3. Call evaluate endpoint
        resp = client.post(
            f"/api/v1/evaluation/session/{session_id}",
            headers=_auth_headers(),
        )
        
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert data["sample_count"] == 1
        assert "faithfulness" in data["scores"]

    def test_get_evaluation_results(self):
        resp = client.get(
            "/api/v1/evaluation/results",
            headers=_auth_headers(),
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
