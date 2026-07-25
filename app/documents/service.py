"""
Document Ingestion Service
Coordinates: save → chunk → embed → index → DB record update
"""
import os
import uuid
import logging
import datetime
from typing import Tuple

from sqlalchemy.orm import Session

from app.utils.config import settings
from app.database.models import Document
from app.ingestion.chunker import load_and_chunk
from app.embeddings.service import embed_texts
from app.retrieval.service import index_chunks, remove_document

logger = logging.getLogger("sagerag.documents")

UPLOAD_DIR = settings.UPLOAD_DIR


def _ensure_upload_dir():
    os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Save uploaded file ─────────────────────────────────────────────────────────

def save_upload(file_bytes: bytes, filename: str) -> Tuple[str, str]:
    """
    Persist the raw file to disk.
    Returns (document_id, file_path).
    """
    _ensure_upload_dir()
    doc_id = str(uuid.uuid4())
    ext = os.path.splitext(filename)[1].lower()
    stored_name = f"{doc_id}{ext}"
    file_path = os.path.join(UPLOAD_DIR, stored_name)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    logger.info(f"Saved uploaded file: {file_path}")
    return doc_id, file_path


# ── Create DB record ───────────────────────────────────────────────────────────

def create_document_record(
    db: Session,
    doc_id: str,
    filename: str,
    file_path: str,
    file_size: int,
    user_id: int,
) -> Document:
    file_ext = os.path.splitext(filename)[1].lower().lstrip(".")
    doc = Document(
        id=doc_id,
        filename=filename,
        storage_path=file_path,
        file_type=file_ext,
        size_bytes=file_size,
        status="pending",
        user_id=user_id,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


# ── Full ingestion pipeline ────────────────────────────────────────────────────

def ingest_document(db: Session, doc_id: str, file_path: str, filename: str):
    """
    Run the full ingestion pipeline for a saved document:
    chunk → embed → index → update DB status.
    Called synchronously; wrap in BackgroundTasks for async execution.
    """
    doc: Document = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        logger.error(f"Document {doc_id} not found in DB.")
        return

    try:
        # Mark as processing
        doc.status = "processing"
        doc.updated_at = datetime.datetime.utcnow()
        db.commit()

        # 1. Chunk
        chunks = load_and_chunk(
            file_path=file_path,
            document_id=doc_id,
            filename=filename,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        if not chunks:
            raise ValueError("No text could be extracted from the document.")

        # 2. Embed
        texts = [c["text"] for c in chunks]
        vectors = embed_texts(texts)

        # 3. Index (Qdrant + keyword)
        index_chunks(chunks, vectors)

        # 4. Mark completed
        doc.status = "completed"
        doc.updated_at = datetime.datetime.utcnow()
        db.commit()
        logger.info(f"Ingestion complete: doc={doc_id}, chunks={len(chunks)}")

    except Exception as e:
        logger.error(f"Ingestion failed for doc {doc_id}: {e}")
        doc.status = "failed"
        doc.error_message = str(e)
        doc.updated_at = datetime.datetime.utcnow()
        db.commit()
        raise


def delete_document(db: Session, doc_id: str, user_id: int) -> bool:
    """
    Delete a document: remove vectors + DB record + disk file.
    Returns True on success.
    """
    doc: Document = (
        db.query(Document)
        .filter(Document.id == doc_id, Document.user_id == user_id)
        .first()
    )
    if not doc:
        return False

    # Remove from vector stores
    remove_document(doc_id)

    # Remove file from disk
    if os.path.exists(doc.storage_path):
        try:
            os.remove(doc.storage_path)
        except Exception as e:
            logger.warning(f"Could not delete file {doc.storage_path}: {e}")

    db.delete(doc)
    db.commit()
    logger.info(f"Deleted document {doc_id}")
    return True
