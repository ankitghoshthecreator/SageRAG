"""
Embedding Service
Uses sentence-transformers (BAAI/bge-small-en-v1.5 by default).
The model is loaded once and cached at module level (lazy init on first call).
"""
import logging
from typing import List, Optional

from app.utils.config import settings

logger = logging.getLogger("sagerag.embeddings")

_model = None  # lazy-loaded SentenceTransformer instance


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
            _model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            logger.info("Embedding model loaded successfully.")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
    return _model


def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Embed a list of strings and return a list of float vectors.
    Normalises embeddings for cosine-similarity search.
    """
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    # BGE models benefit from a short instruction prefix for queries
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    return embed_texts([prefixed])[0]


def get_embedding_dimension() -> int:
    """Return the vector dimension of the current model."""
    model = _get_model()
    return model.get_sentence_embedding_dimension()
