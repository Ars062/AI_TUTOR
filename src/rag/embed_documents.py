import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


model = SentenceTransformer("all-MiniLM-L6-v2")


def build_vector_index(doc_folder):

    documents = []
    filenames = []

    for file in os.listdir(doc_folder):
        path = os.path.join(doc_folder, file)

        with open(path, "r", encoding="utf8") as f:
            text = f.read()

        documents.append(text)
        filenames.append(file)

    embeddings = model.encode(documents)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index, documents
