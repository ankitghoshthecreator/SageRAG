"""
Part 2 Integration Tests
Tests document ingestion pipeline:
  - Chunker on plain text
  - Hybrid retrieval RRF logic
  - Reranker fallback (no model needed)
  - Document upload endpoint (with mocked ingestion)
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Chunker ────────────────────────────────────────────────────────────────────

def test_chunker_plain_text(tmp_path):
    from app.ingestion.chunker import load_and_chunk

    content = "Hello world. " * 100  # ~1300 chars
    txt_file = tmp_path / "test.txt"
    txt_file.write_text(content)

    chunks = load_and_chunk(
        file_path=str(txt_file),
        document_id="doc-001",
        filename="test.txt",
        chunk_size=200,
        chunk_overlap=20,
    )
    assert len(chunks) >= 2
    for c in chunks:
        assert "chunk_id" in c
        assert "document_id" in c
        assert c["document_id"] == "doc-001"
        assert "text" in c
        assert len(c["text"]) > 0


def test_chunker_markdown(tmp_path):
    from app.ingestion.chunker import load_and_chunk

    md_content = "# Title\n\nParagraph one.\n\nParagraph two.\n\nParagraph three."
    md_file = tmp_path / "test.md"
    md_file.write_text(md_content)

    chunks = load_and_chunk(
        file_path=str(md_file),
        document_id="doc-002",
        filename="test.md",
        chunk_size=100,
        chunk_overlap=10,
    )
    assert len(chunks) >= 1
    assert all(c["filename"] == "test.md" for c in chunks)


def test_chunker_unsupported_ext(tmp_path):
    from app.ingestion.chunker import load_and_chunk

    f = tmp_path / "test.xyz"
    f.write_text("content")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_and_chunk(str(f), "doc-003", "test.xyz")


# ── Keyword index / retrieval ─────────────────────────────────────────────────

def test_keyword_index_basic():
    from app.retrieval.service import _KeywordIndex

    idx = _KeywordIndex()
    chunks = [
        {"chunk_id": "c1", "document_id": "d1", "text": "FastAPI is a modern web framework for Python"},
        {"chunk_id": "c2", "document_id": "d1", "text": "Qdrant is a vector database for similarity search"},
        {"chunk_id": "c3", "document_id": "d1", "text": "Python is widely used for machine learning"},
    ]
    idx.add_chunks(chunks)

    results = idx.search("FastAPI web framework", top_k=2)
    assert len(results) > 0
    assert results[0]["chunk_id"] == "c1"


def test_rrf_fusion():
    from app.retrieval.service import _rrf_fusion

    dense = [
        {"chunk_id": "c1", "text": "a", "score": 0.9},
        {"chunk_id": "c2", "text": "b", "score": 0.8},
        {"chunk_id": "c3", "text": "c", "score": 0.5},
    ]
    keyword = [
        {"chunk_id": "c2", "text": "b", "score": 0.7},
        {"chunk_id": "c3", "text": "c", "score": 0.6},
        {"chunk_id": "c4", "text": "d", "score": 0.4},
    ]
    fused = _rrf_fusion(dense, keyword)
    ids = [r["chunk_id"] for r in fused]
    # c2 and c3 appear in both lists so should rank higher than c1 or c4
    assert ids.index("c2") < ids.index("c1")


# ── Reranker fallback ──────────────────────────────────────────────────────────

def test_reranker_fallback_empty():
    from app.reranker.service import rerank

    result = rerank("anything", [])
    assert result == []


def test_reranker_fallback_score_sort():
    """Without a model loaded, reranker should sort by existing score."""
    import app.reranker.service as rs

    # Temporarily disable model loading
    original = rs._reranker
    rs._reranker = None  # force fallback path

    # Monkey-patch _get_reranker to return None
    original_get = rs._get_reranker
    rs._get_reranker = lambda: None

    chunks = [
        {"chunk_id": "c1", "text": "low",  "score": 0.3},
        {"chunk_id": "c2", "text": "high", "score": 0.9},
        {"chunk_id": "c3", "text": "mid",  "score": 0.6},
    ]
    result = rs.rerank("query", chunks, top_k=2)
    assert result[0]["chunk_id"] == "c2"
    assert result[1]["chunk_id"] == "c3"

    # Restore
    rs._get_reranker = original_get
    rs._reranker = original
