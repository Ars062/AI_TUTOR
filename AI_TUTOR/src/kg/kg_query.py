import re

import numpy as np
from neo4j import GraphDatabase
from src.config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    KG_RESULT_LIMIT,
    KG_MAX_HOPS,
    KG_EMBED_ANCHORS,
    KG_ANCHOR_TOPK,
    KG_ANCHOR_THRESHOLD,
)

_STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "in", "to", "for", "on",
    "and", "or", "how", "why", "does", "do", "did", "explain", "define",
    "describe", "with", "about", "me", "you", "it", "its", "they", "them",
    "can", "could", "would", "should", "where", "when", "which", "who",
}


def _extract_keywords(question):
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+-]*", question.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _all_concept_names(limit=1000):
    """Fetch every Concept node name from the KG. Returns [] on failure."""
    try:
        driver = _get_driver()
    except Exception:
        return []
    try:
        with driver.session() as session:
            result = session.run(
                "MATCH (a:Concept) RETURN a.name AS name LIMIT $limit", limit=limit
            )
            names = [record["name"] for record in result]
            driver.close()
            return names
    except Exception:
        driver.close()
        return []


def find_anchor_nodes(question, topk=None, threshold=None, use_embedding=None):
    """Embedding-based KG node matching: find the Concept nodes closest to the question.

    Falls back to keyword extraction when embeddings are disabled or no
    sufficiently similar nodes are found.
    """
    if topk is None:
        topk = KG_ANCHOR_TOPK
    if threshold is None:
        threshold = KG_ANCHOR_THRESHOLD
    if use_embedding is None:
        use_embedding = KG_EMBED_ANCHORS

    names = _all_concept_names()
    if not names:
        return set()

    if use_embedding:
        try:
            from src.rag.embeddings import embed

            q_vec = embed([question])
            name_vecs = embed(names)
            sims = name_vecs @ q_vec[0]
            ranked = sorted(zip(names, sims), key=lambda x: x[1], reverse=True)
            anchors = {n for n, s in ranked[:topk] if s >= threshold}
            if anchors:
                return anchors
        except Exception as e:
            print(f"Warning: embedding KG matching failed ({e}); using keywords.")

    keywords = _extract_keywords(question)
    if keywords:
        return {k for k in keywords if k.capitalize() in names or k in names}

    return set()


def query_kg(keyword, max_hops=None, limit=None):
    if max_hops is None:
        max_hops = KG_MAX_HOPS
    if limit is None:
        limit = KG_RESULT_LIMIT

    if not keyword or not keyword.strip():
        return ""

    try:
        driver = _get_driver()
    except Exception as e:
        print(f"Warning: could not connect to Neo4j: {e}")
        return ""

    anchors = find_anchor_nodes(keyword)

    hop_pattern = f"[*1..{max_hops}]"

    try:
        triples = []
        with driver.session() as session:
            if anchors:
                result = session.run(
                    f"""
                    MATCH path = (a)-{hop_pattern}-(b)
                    WHERE a.name IN $anchors AND a <> b
                    UNWIND relationships(path) AS r
                    RETURN startNode(r).name AS src, type(r) AS rel, endNode(r).name AS dst
                    LIMIT $limit
                    """,
                    anchors=list(anchors),
                    max_hops=max_hops,
                    limit=limit,
                )
                for record in result:
                    t = f"{record['src']} {record['rel']} {record['dst']}"
                    if t not in triples:
                        triples.append(t)
            else:
                keywords = _extract_keywords(keyword) or [keyword.strip()]
                result = session.run(
                    f"""
                    MATCH path = (a)-{hop_pattern}->(b)
                    WHERE toLower(a.name) CONTAINS toLower($keyword)
                      AND a <> b
                    UNWIND relationships(path) AS r
                    RETURN startNode(r).name AS src, type(r) AS rel, endNode(r).name AS dst
                    LIMIT $limit
                    """,
                    keyword=keyword if keyword else keywords[0],
                    max_hops=max_hops,
                    limit=limit,
                )
                for record in result:
                    t = f"{record['src']} {record['rel']} {record['dst']}"
                    if t not in triples:
                        triples.append(t)
            driver.close()
            return "\n".join(triples) if triples else ""

    except Exception as e:
        print(f"Warning: KG query failed: {e}")
        driver.close()
        return ""


def query_kg_entities(keyword, limit=None):
    """Return the anchor Concept nodes matching a question (embedding-based)."""
    if limit is None:
        limit = KG_RESULT_LIMIT

    return {e for e in find_anchor_nodes(keyword)}
