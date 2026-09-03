"""
RAG Prompt Builder
==================
Constructs the LLM message list from:
  - A system instruction
  - Retrieved and reranked document chunks (with [Source N] labels)
  - Recent conversation history (multi-turn context)
  - The current user query

The final prompt structure sent to the LLM:

  [system]    Role + instructions + retrieved context
  [user]      Turn N-2
  [assistant] Turn N-2 response
  ...
  [user]      Current query
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.llm.service import LLMMessage

# Maximum number of previous turns to include for multi-turn context
MAX_HISTORY_TURNS = 6  # 3 user+assistant pairs


SYSTEM_TEMPLATE = """\
You are SageRAG, an enterprise knowledge assistant. You answer questions \
accurately and concisely using ONLY the information provided in the context \
below. If the context does not contain enough information to answer the \
question, say so clearly — do not hallucinate.

Always cite the source of your information using the [Source N] label that \
appears before each context block. Include the relevant source labels at the \
end of your answer.

--- CONTEXT ---
{context}
--- END CONTEXT ---
"""


def build_rag_prompt(
    query: str,
    chunks: List[Dict[str, Any]],
    history: List[Dict[str, str]] | None = None,
) -> List[LLMMessage]:
    """
    Build the full message list for the LLM.

    Parameters
    ----------
    query   : The current user question.
    chunks  : Reranked chunk dicts containing at least 'text', 'filename',
              'page_number'.
    history : Recent message dicts [{"role": "user"|"assistant", "content": "..."}]
              ordered oldest → newest. Only the last MAX_HISTORY_TURNS are used.

    Returns
    -------
    List[LLMMessage] ready to pass to llm_service.generate() / stream().
    """
    # Build context string with labelled source blocks
    context_blocks: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source_label = f"[Source {i}] {chunk.get('filename', 'unknown')} (page {chunk.get('page_number', '?')})"
        context_blocks.append(f"{source_label}\n{chunk['text'].strip()}")

    context_str = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."
    system_content = SYSTEM_TEMPLATE.format(context=context_str)

    messages: List[LLMMessage] = [LLMMessage(role="system", content=system_content)]

    # Inject recent conversation history
    if history:
        recent = history[-(MAX_HISTORY_TURNS * 2):]  # user+assistant pairs
        for turn in recent:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append(LLMMessage(role=role, content=content))

    # Append the current user query
    messages.append(LLMMessage(role="user", content=query))

    return messages


def extract_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract a clean citation list from the reranked chunks.
    Each citation contains the source metadata needed by the frontend.
    """
    citations = []
    for i, chunk in enumerate(chunks, start=1):
        citations.append({
            "source_index": i,
            "chunk_id": chunk.get("chunk_id", ""),
            "document_id": chunk.get("document_id", ""),
            "filename": chunk.get("filename", ""),
            "page_number": chunk.get("page_number", None),
            "chunk_index": chunk.get("chunk_index", None),
            "score": round(chunk.get("rerank_score", chunk.get("score", 0.0)), 4),
            "text_preview": chunk.get("text", "")[:200].strip(),
        })
    return citations
