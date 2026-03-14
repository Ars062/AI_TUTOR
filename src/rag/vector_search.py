from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def search_docs(query, index, documents, k=3):

    q_embed = model.encode([query])

    D, I = index.search(np.array(q_embed), k)

    results = []

    for i in I[0]:
        results.append(documents[i])

    return "\n".join(results)
