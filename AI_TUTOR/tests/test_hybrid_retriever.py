import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.hybrid_retriever import hybrid_retrieve
from src.rag.embed_documents import _empty_index


def test_hybrid_retrieve_empty():
    index, docs, filenames = _empty_index()
    result = hybrid_retrieve("What is recursion?", index, [])
    assert "kg_context" in result
    assert "doc_context" in result
    assert "kg_guided_docs" in result
    assert "entities" in result


def test_hybrid_retrieve_returns_dict():
    index, docs, filenames = _empty_index()
    result = hybrid_retrieve("test query", index, [])
    assert isinstance(result, dict)
    assert len(result) == 4


if __name__ == "__main__":
    test_hybrid_retrieve_empty()
    test_hybrid_retrieve_returns_dict()
    print("All hybrid retriever tests passed!")
