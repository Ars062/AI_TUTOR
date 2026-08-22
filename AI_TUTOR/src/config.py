import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "data/documents")
KG_CSV_PATH = os.getenv("KG_CSV_PATH", "data/knowledge_graph/knowledge_triples.csv")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index.bin")
TOP_K_DOCS = int(os.getenv("TOP_K_DOCS", "5"))
KG_MAX_HOPS = int(os.getenv("KG_MAX_HOPS", "2"))
KG_RESULT_LIMIT = int(os.getenv("KG_RESULT_LIMIT", "15"))
KG_EMBED_ANCHORS = os.getenv("KG_EMBED_ANCHORS", "true").lower() == "true"
KG_ANCHOR_TOPK = int(os.getenv("KG_ANCHOR_TOPK", "5"))
KG_ANCHOR_THRESHOLD = float(os.getenv("KG_ANCHOR_THRESHOLD", "0.35"))
USE_ENSEMBLE = os.getenv("USE_ENSEMBLE", "true").lower() == "true"
USE_S2A = os.getenv("USE_S2A", "true").lower() == "true"
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))
ENABLE_CONTENT_SAFETY = os.getenv("ENABLE_CONTENT_SAFETY", "true").lower() == "true"
