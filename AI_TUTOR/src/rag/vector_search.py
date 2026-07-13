from sentence_transformers import SentenceTransformer
import numpy as np

from src.config import TOP_K_DOCS

model = SentenceTransformer("all-MiniLM-L6-v2")


def search_docs(query, index, documents, k=None):
    if k is None:
        k = TOP_K_DOCS

    if index.ntotal == 0 or not documents:
        return ""

    q_embed = model.encode([query])
    D, I = index.search(np.array(q_embed), min(k, index.ntotal))

    results = []
    for i in I[0]:
        if i < len(documents):
            results.append(documents[i])

    return "\n\n---\n\n".join(results) if results else ""
