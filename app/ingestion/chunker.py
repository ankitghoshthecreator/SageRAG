"""
Document Loader & Chunker
Handles: PDF, DOCX, Markdown, plain text
Returns a list of Chunk dicts ready for embedding.
"""
import os
import re
import uuid
import logging
from typing import List, Dict, Any

logger = logging.getLogger("sagerag.chunker")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Strip excess whitespace while keeping paragraph boundaries."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_into_chunks(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Split text into overlapping character-based chunks.
    We split on sentence/paragraph boundaries where possible.
    """
    if not text:
        return []

    # Split on double-newlines (paragraphs) or sentence endings
    paragraphs = re.split(r"\n\n+", text)
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 1 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # If this single paragraph is larger than chunk_size, hard-split it
            if len(para) > chunk_size:
                for start in range(0, len(para), chunk_size - overlap):
                    piece = para[start: start + chunk_size]
                    if piece.strip():
                        chunks.append(piece.strip())
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


# ── Loaders ────────────────────────────────────────────────────────────────────

def _load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Return list of {page_number, text} dicts from a PDF."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf is not installed. Run: pip install pypdf")

    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page_number": i + 1, "text": _clean_text(text)})
    return pages


def _load_docx(file_path: str) -> List[Dict[str, Any]]:
    """Return list of {page_number, text} dicts from a DOCX (no native pages)."""
    try:
        from docx import Document as DocxDocument
    except ImportError:
        raise RuntimeError("python-docx is not installed. Run: pip install python-docx")

    doc = DocxDocument(file_path)
    full_text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"page_number": 1, "text": _clean_text(full_text)}]


def _load_text(file_path: str) -> List[Dict[str, Any]]:
    """Return list with a single page dict for .txt and .md files."""
    try:
        import chardet
        with open(file_path, "rb") as f:
            raw = f.read()
        enc = chardet.detect(raw).get("encoding") or "utf-8"
        text = raw.decode(enc, errors="replace")
    except Exception:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    return [{"page_number": 1, "text": _clean_text(text)}]


# ── Public API ─────────────────────────────────────────────────────────────────

def load_and_chunk(
    file_path: str,
    document_id: str,
    filename: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Dict[str, Any]]:
    """
    Load a document, split into chunks, and return a list of chunk dicts:
    {
        "chunk_id": str,
        "document_id": str,
        "filename": str,
        "page_number": int,
        "chunk_index": int,
        "text": str,
    }
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        pages = _load_pdf(file_path)
    elif ext in {".docx", ".doc"}:
        pages = _load_docx(file_path)
    elif ext in {".md", ".txt", ".text"}:
        pages = _load_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    chunks: List[Dict[str, Any]] = []
    chunk_index = 0

    for page in pages:
        page_chunks = _split_into_chunks(
            page["text"], chunk_size=chunk_size, overlap=chunk_overlap
        )
        for text in page_chunks:
            if not text.strip():
                continue
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "document_id": document_id,
                "filename": filename,
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
                "text": text,
            })
            chunk_index += 1

    logger.info(
        f"Loaded '{filename}' → {len(pages)} page(s), {len(chunks)} chunk(s)"
    )
    return chunks
