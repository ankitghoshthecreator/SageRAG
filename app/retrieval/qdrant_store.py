"""
Qdrant Vector Store
Manages the Qdrant collection: create, upsert, search.
Falls back to a simple in-memory list when Qdrant is not reachable,
so the rest of the pipeline always works during development.
"""
import logging
import uuid
from typing import List, Dict, Any, Optional

from app.utils.config import settings

logger = logging.getLogger("sagerag.qdrant")

# ── Fallback in-memory store ────────────────────────────────────────────────────

class _InMemoryStore:
    """Minimal fallback when Qdrant is not running."""

    def __init__(self):
        self._docs: List[Dict] = []
        logger.warning(
            "Qdrant not available – using in-memory vector store. "
            "Data will be lost on restart."
        )

    def upsert(self, chunks: List[Dict[str, Any]], vectors: List[List[float]]):
        for chunk, vec in zip(chunks, vectors):
            self._docs.append({"chunk": chunk, "vector": vec})
        logger.info(f"In-memory store: upserted {len(chunks)} chunks.")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filter_doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        import numpy as np

        results = []
        for item in self._docs:
            if filter_doc_id and item["chunk"]["document_id"] != filter_doc_id:
                continue
            score = float(
                np.dot(query_vector, item["vector"])
                / (
                    np.linalg.norm(query_vector) * np.linalg.norm(item["vector"])
                    + 1e-9
                )
            )
            results.append({**item["chunk"], "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete_by_document_id(self, document_id: str):
        before = len(self._docs)
        self._docs = [d for d in self._docs if d["chunk"]["document_id"] != document_id]
        logger.info(f"Deleted {before - len(self._docs)} chunks for doc {document_id}")

    def collection_exists(self) -> bool:
        return True


# ── Qdrant store ────────────────────────────────────────────────────────────────

class QdrantStore:
    """
    Wraps the official qdrant-client.
    All public methods have the same signature as _InMemoryStore.
    """

    def __init__(self):
        self._client = None
        self._dim: Optional[int] = None

    def _connect(self):
        if self._client is not None:
            return
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct

            self._client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY or None,
                timeout=5,
            )
            # lightweight connectivity check
            self._client.get_collections()
            logger.info(
                f"Connected to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
            )
        except Exception as e:
            logger.error(f"Cannot connect to Qdrant: {e}")
            raise

    def _ensure_collection(self, dim: int):
        from qdrant_client.models import Distance, VectorParams

        name = settings.QDRANT_COLLECTION_NAME
        existing = [c.name for c in self._client.get_collections().collections]
        if name not in existing:
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection '{name}' (dim={dim})")

    def upsert(self, chunks: List[Dict[str, Any]], vectors: List[List[float]]):
        from qdrant_client.models import PointStruct

        self._connect()
        if not chunks:
            return
        self._ensure_collection(len(vectors[0]))

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=chunk,
            )
            for chunk, vec in zip(chunks, vectors)
        ]
        self._client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=points,
        )
        logger.info(f"Qdrant upserted {len(points)} vectors.")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filter_doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        self._connect()
        query_filter = None
        if filter_doc_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=filter_doc_id),
                    )
                ]
            )
        hits = self._client.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            {**hit.payload, "score": hit.score}
            for hit in hits
        ]

    def delete_by_document_id(self, document_id: str):
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        self._connect()
        self._client.delete(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )
        logger.info(f"Deleted vectors for document {document_id} from Qdrant.")


# ── Factory: return Qdrant or in-memory fallback ───────────────────────────────

def get_vector_store():
    """Return a connected QdrantStore, or an in-memory fallback."""
    store = QdrantStore()
    try:
        store._connect()
        return store
    except Exception:
        return _InMemoryStore()


# module-level singleton
vector_store = get_vector_store()
