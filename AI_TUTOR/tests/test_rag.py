import sys
import os
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.embed_documents import build_vector_index, _empty_index, SUPPORTED_EXTENSIONS
from src.rag.vector_search import search_docs


def test_empty_index():
    index, docs, filenames = _empty_index()
    assert index.ntotal == 0
    assert docs == []
    assert filenames == []


def test_build_index_empty_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        index, docs, filenames = build_vector_index(tmpdir)
        assert index.ntotal == 0


def test_build_index_with_documents():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "test.txt"), "w") as f:
            f.write("Recursion is a programming technique.")

        index, docs, filenames = build_vector_index(tmpdir)
        assert index.ntotal == 1
        assert len(docs) == 1
        assert len(filenames) == 1
        assert filenames[0] == "test.txt"


def test_search_empty_index():
    index, docs, filenames = _empty_index()
    result = search_docs("recursion", index, [])
    assert result == ""


def test_supported_extensions():
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert ".md" in SUPPORTED_EXTENSIONS


if __name__ == "__main__":
    test_empty_index()
    test_build_index_empty_folder()
    test_build_index_with_documents()
    test_search_empty_index()
    test_supported_extensions()
    print("All RAG tests passed!")
