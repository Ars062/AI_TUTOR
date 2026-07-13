from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, KG_RESULT_LIMIT, KG_MAX_HOPS


def _get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


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

    cypher = """
    MATCH (a)-[r]->(b)
    WHERE a.name CONTAINS $keyword
    RETURN a.name, type(r) AS rel, b.name
    LIMIT $limit
    """

    try:
        with driver.session() as session:
            result = session.run(cypher, keyword=keyword, limit=limit)
            triples = []
            for record in result:
                triples.append(f"{record['a.name']} {record['rel']} {record['b.name']}")

            if max_hops > 1:
                names = set()
                for t in triples:
                    parts = t.split(" ", 1)
                    if parts:
                        names.add(parts[0])
                if names:
                    expand_cypher = """
                    MATCH (a)-[r]->(b)
                    WHERE a.name IN $names
                    RETURN a.name, type(r) AS rel, b.name
                    LIMIT $limit2
                    """
                    exp_result = session.run(expand_cypher, names=list(names), limit2=limit)
                    for record in exp_result:
                        t = f"{record['a.name']} {record['rel']} {record['b.name']}"
                        if t not in triples:
                            triples.append(t)

            driver.close()
            return "\n".join(triples) if triples else ""

    except Exception as e:
        print(f"Warning: KG query failed: {e}")
        driver.close()
        return ""


def query_kg_entities(keyword, limit=None):
    if limit is None:
        limit = KG_RESULT_LIMIT

    try:
        driver = _get_driver()
    except Exception as e:
        return set()

    cypher = """
    MATCH (a:Concept)
    WHERE a.name CONTAINS $keyword
    RETURN a.name
    LIMIT $limit
    """

    try:
        with driver.session() as session:
            result = session.run(cypher, keyword=keyword, limit=limit)
            entities = {record["a.name"] for record in result}
            driver.close()
            return entities
    except Exception as e:
        driver.close()
        return set()
