import sys
import os
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_config_defaults():
    from src.config import (
        GROQ_API_KEY,
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        DOCUMENTS_DIR,
        TOP_K_DOCS,
        KG_MAX_HOPS,
    )

    assert NEO4J_URI == "bolt://localhost:7687"
    assert NEO4J_USER == "neo4j"
    assert NEO4J_PASSWORD == "password"
    assert DOCUMENTS_DIR == "data/documents"
    assert TOP_K_DOCS >= 1
    assert KG_MAX_HOPS >= 1


def test_config_from_env():
    os.environ["NEO4J_URI"] = "bolt://custom:7687"
    os.environ["TOP_K_DOCS"] = "10"

    import importlib
    import src.config
    importlib.reload(src.config)

    assert src.config.NEO4J_URI == "bolt://custom:7687"
    assert src.config.TOP_K_DOCS == 10

    del os.environ["NEO4J_URI"]
    del os.environ["TOP_K_DOCS"]
    importlib.reload(src.config)


if __name__ == "__main__":
    test_config_defaults()
    test_config_from_env()
    print("All config tests passed!")
