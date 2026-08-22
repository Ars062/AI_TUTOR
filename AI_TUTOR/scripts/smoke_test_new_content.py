import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.embed_documents import load_index
from src.kg.kg_loader import load_kg
from src.config import KG_CSV_PATH
from src.tutor.tutor_engine import ask_tutor

load_kg(KG_CSV_PATH)
index, docs, files = load_index()

for q in [
    "What is dynamic programming and when should I use it?",
    "Explain how quicksort works",
]:
    answer, debug = ask_tutor(q, index, docs, files, use_cot=True)
    print("=" * 60)
    print("Q:", q)
    print("kg facts:", len((debug.get("kg_context") or "").splitlines()))
    print("doc_context chars:", len(debug.get("doc_context") or ""))
    print("cot steps:", len(debug.get("cot_steps") or []))
    val = debug.get("cot_validation") or {}
    print("grounded:", val.get("grounded_fraction"))
    print("answer chars:", len(answer))
    print("--- excerpt ---")
    print(answer[:400].replace("\n", " "))
    print()
