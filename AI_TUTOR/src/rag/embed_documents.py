import os
import re
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from src.config import DOCUMENTS_DIR, FAISS_INDEX_PATH, CHUNK_SIZE, CHUNK_OVERLAP

model = SentenceTransformer("all-MiniLM-L6-v2")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv"}


def _read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text, size=None, overlap=None):
    """Split text into overlapping chunks, breaking on paragraph/sentence/word."""
    if size is None:
        size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP

    text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if len(text) <= size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end >= len(text):
            piece = text[start:end].strip()
            if len(piece) > 50:
                chunks.append(piece)
            break
        cut = max(
            text.rfind("\n\n", start, end),
            text.rfind(". ", start, end),
            text.rfind(" ", start, end),
        )
        if cut > start + size // 2:
            end = cut + 1
        piece = text[start:end].strip()
        if len(piece) > 50:
            chunks.append(piece)
        start = max(end - overlap, start + 1)
    return chunks


def build_vector_index(doc_folder=None):
    if doc_folder is None:
        doc_folder = DOCUMENTS_DIR

    documents = []
    filenames = []

    if not os.path.isdir(doc_folder):
        print(f"Warning: document folder '{doc_folder}' not found")
        return _empty_index()

    for file in sorted(os.listdir(doc_folder)):
        ext = os.path.splitext(file)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        path = os.path.join(doc_folder, file)
        try:
            text = _read_file(path)
            if text.strip():
                for chunk in chunk_text(text):
                    documents.append(chunk)
                    filenames.append(file)
        except Exception as e:
            print(f"Warning: could not read {file}: {e}")

    if not documents:
        print("Warning: no documents found")
        return _empty_index()

    embeddings = model.encode(documents, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, documents, filenames


def _empty_index():
    index = faiss.IndexFlatL2(384)
    return index, [], []


def save_index(index, documents, filenames, path=None):
    if path is None:
        path = FAISS_INDEX_PATH
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    faiss.write_index(index, path)
    meta_path = path + ".meta"
    with open(meta_path, "wb") as f:
        pickle.dump({"documents": documents, "filenames": filenames}, f)


def load_index(path=None):
    if path is None:
        path = FAISS_INDEX_PATH
    if not os.path.exists(path):
        return _empty_index()
    index = faiss.read_index(path)
    meta_path = path + ".meta"
    if not os.path.exists(meta_path):
        return index, [], []
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return index, meta["documents"], meta["filenames"]
