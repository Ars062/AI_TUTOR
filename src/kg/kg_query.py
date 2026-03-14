from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"

driver = GraphDatabase.driver(uri, auth=(user, password))


def query_kg(keyword):

    cypher = """
    MATCH (a)-[r]->(b)
    WHERE a.name CONTAINS $keyword
    RETURN a.name, type(r), b.name
    LIMIT 10
    """

    with driver.session() as session:
        result = session.run(cypher, keyword=keyword)

        triples = []

        for record in result:
            triples.append(f"{record['a.name']} {record['type(r)']} {record['b.name']}")

    return "\n".join(triples)
