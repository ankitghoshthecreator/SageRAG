"""
Evaluation API Router
=====================
Endpoints
---------
POST /evaluation/run                Run evaluation on manual samples
POST /evaluation/session/{id}       Run auto-evaluation on a chat session
GET  /evaluation/results            List historical evaluation runs
"""
import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.database.models import User, EvaluationRun
from app.auth.security import get_current_active_user
from app.evaluation import service as eval_service
from app.utils.config import settings

router = APIRouter()

# ── Schemas ────────────────────────────────────────────────────────────────────

class SampleRequest(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None

class ManualRunRequest(BaseModel):
    samples: List[SampleRequest]

class EvaluationRunResponse(BaseModel):
    id: str
    session_id: str
    user_id: int
    scores: Dict[str, float]
    sample_count: int
    created_at: datetime.datetime

    class Config:
        from_attributes = True

# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/run", response_model=Dict[str, float])
def run_manual_evaluation(
    body: ManualRunRequest,
    current_user: User = Depends(get_current_active_user),
):
    """Run RAGAS on manually provided samples (doesn't save to DB)."""
    if not settings.EVALUATION_ENABLED:
        raise HTTPException(status_code=403, detail="Evaluation is disabled in settings.")
        
    samples = [eval_service.EvaluationSample(**s.dict()) for s in body.samples]
    scores = eval_service.run_ragas(samples)
    if not scores:
        raise HTTPException(status_code=500, detail="Evaluation failed or RAGAS not installed.")
    return scores

@router.post("/session/{session_id}", response_model=EvaluationRunResponse)
def evaluate_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Extract Q&A pairs from a session, run RAGAS, and save results."""
    if not settings.EVALUATION_ENABLED:
        raise HTTPException(status_code=403, detail="Evaluation is disabled in settings.")
        
    run = eval_service.evaluate_from_session(db, session_id, current_user.id)
    if not run:
        raise HTTPException(
            status_code=400, 
            detail="Session not found, no evaluable Q&A pairs, or evaluation failed."
        )
    return run

@router.get("/results", response_model=List[EvaluationRunResponse])
def get_evaluation_results(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List historical evaluation runs for the user."""
    runs = (
        db.query(EvaluationRun)
        .filter(EvaluationRun.user_id == current_user.id)
        .order_by(EvaluationRun.created_at.desc())
        .all()
    )
    return runs
