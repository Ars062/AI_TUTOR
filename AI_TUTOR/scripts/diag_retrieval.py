import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.rag.embed_documents import load_index
from src.rag.embeddings import embed

index, docs, files = load_index()
print("index size:", index.ntotal)

sizes = [len(d) for d in docs]
print("chunk sizes: min", min(sizes), "median", sorted(sizes)[len(sizes) // 2], "max", max(sizes))
tiny = sum(1 for s in sizes if s < 200)
print("chunks under 200 chars:", tiny)

for q in ["What is dynamic programming and when should I use it?", "Explain how quicksort works"]:
    print("\nQ:", q)
    D, I = index.search(np.array(embed([q])), 8)
    for rank, i in enumerate(I[0]):
        print(f"  {rank + 1}. {files[i]:<28} len={len(docs[i]):>5}  {docs[i][:55]!r}")
