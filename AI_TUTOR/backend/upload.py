"""Milestone 9: document upload → chunk → embed → add to FAISS index.

The tutor gains knowledge from uploaded PDFs and text files in real-time
without rebuilding the entire index. Uploaded files are also saved to
data/documents/ so they survive restarts.
"""
import os
import re
import uuid

import numpy as np
from pypdf import PdfReader

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text(path: str, filename: str) -> str:
    low = filename.lower()
    if low.endswith(".pdf"):
        reader = PdfReader(path)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def chunk_text(text: str) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                idx = text.rfind(sep, start, end)
                if idx > start + CHUNK_SIZE // 2:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if len(chunk) > 100:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def add_documents_to_index(
    index, documents: list[str], filenames: list[str],
    new_docs: list[str], new_names: list[str],
):
    """Embed new documents and add to existing FAISS index in-place."""
    from src.rag.embeddings import embed_texts

    if not new_docs:
        return index, documents, filenames

    embeddings = embed_texts(new_docs)
    vectors = np.array(embeddings, dtype="float32")
    index.add(vectors)
    documents.extend(new_docs)
    filenames.extend(new_names)
    return index, documents, filenames


def save_uploaded_file(file_content: bytes, original_name: str) -> str:
    """Persist to data/documents/ with a unique prefix to avoid collisions."""
    dest_dir = os.path.join(os.path.dirname(__file__), "..", "data", "documents")
    os.makedirs(dest_dir, exist_ok=True)
    safe = re.sub(r"[^\w.\-]", "_", original_name)
    dest = os.path.join(dest_dir, f"{uuid.uuid4().hex[:8]}_{safe}")
    with open(dest, "wb") as f:
        f.write(file_content)
    return dest
