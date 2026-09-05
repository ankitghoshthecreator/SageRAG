"""
Evaluation Service
==================
Runs RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall)
on a set of Q&A samples or a full chat session.
"""
import logging
import uuid
import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.database.models import ChatSession, Message, EvaluationRun
from app.utils.config import settings

logger = logging.getLogger("sagerag.evaluation")


@dataclass
class EvaluationSample:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


def run_ragas(samples: List[EvaluationSample]) -> Dict[str, float]:
    """
    Run RAGAS evaluation on a list of samples.
    Requires `ragas` and `datasets` to be installed.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
    except ImportError:
        logger.error("RAGAS or datasets is not installed. Returning empty scores.")
        return {}

    if not samples:
        return {}

    # Convert to HuggingFace Dataset
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for sample in samples:
        data["question"].append(sample.question)
        data["answer"].append(sample.answer)
        data["contexts"].append(sample.contexts)
        data["ground_truth"].append(sample.ground_truth or "")

    dataset = Dataset.from_dict(data)

    # Note: RAGAS uses Langchain + OpenAI under the hood by default,
    # expecting OPENAI_API_KEY to be set in the environment.
    metrics = [faithfulness, answer_relevancy, context_precision]
    
    # Only use context_recall if ground truth is provided for all samples
    if all(sample.ground_truth for sample in samples):
        metrics.append(context_recall)

    logger.info(f"Running RAGAS evaluation on {len(samples)} samples...")
    try:
        result = evaluate(
            dataset,
            metrics=metrics,
        )
        scores = {k: round(v, 4) for k, v in result.items()}
        logger.info(f"RAGAS evaluation complete: {scores}")
        return scores
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return {}


def evaluate_from_session(db: Session, session_id: str, user_id: int) -> Optional[EvaluationRun]:
    """
    Extracts Q&A pairs from a chat session, runs RAGAS, and saves the result to DB.
    """
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user_id).first()
    if not session:
        return None

    messages = db.query(Message).filter(Message.session_id == session_id).order_by(Message.created_at.asc()).all()

    samples = []
    # Iterate through messages to find user -> assistant pairs
    for i in range(len(messages) - 1):
        if messages[i].role == "user" and messages[i+1].role == "assistant":
            user_msg = messages[i]
            asst_msg = messages[i+1]
            
            # Extract contexts from citations if available
            contexts = []
            if asst_msg.citations:
                for cit in asst_msg.citations:
                    contexts.append(cit.get("text_preview", ""))
            
            # Skip if no context was used (can't evaluate RAG metrics without it)
            if not contexts:
                continue

            samples.append(EvaluationSample(
                question=user_msg.content,
                answer=asst_msg.content,
                contexts=contexts
            ))

    if not samples:
        logger.warning(f"No valid Q&A pairs with context found in session {session_id}")
        return None

    scores = run_ragas(samples)
    if not scores:
        return None

    eval_run = EvaluationRun(
        id=str(uuid.uuid4()),
        session_id=session_id,
        user_id=user_id,
        scores=scores,
        sample_count=len(samples)
    )
    db.add(eval_run)
    db.commit()
    db.refresh(eval_run)
    
    return eval_run
