from sentence_transformers import SentenceTransformer

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def embed(texts):
    return get_embedder().encode(texts, normalize_embeddings=True, batch_size=32)