import numpy as np

from src.config import TOP_K_DOCS
from src.rag.embeddings import get_embedder


def search_docs(query, index, documents, k=None):
    if k is None:
        k = TOP_K_DOCS

    if index.ntotal == 0 or not documents:
        return ""

    q_embed = get_embedder().encode([query])
    D, I = index.search(np.array(q_embed), min(k, index.ntotal))

    results = []
    for i in I[0]:
        if i < len(documents):
            results.append(documents[i])

    return "\n\n---\n\n".join(results) if results else ""
