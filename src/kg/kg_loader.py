import pandas as pd
from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"

driver = GraphDatabase.driver(uri, auth=(user, password))


def load_kg(csv_path):

    df = pd.read_csv(csv_path)

    with driver.session() as session:
        for _, row in df.iterrows():
            source = row["source"]
            relation = row["relation"]
            target = row["target"]

            query = f"""
            MERGE (a:Concept {{name: $source}})
            MERGE (b:Concept {{name: $target}})
            MERGE (a)-[:{relation.upper()}]->(b)
            """

            session.run(query, source=source, target=target)

    print("Knowledge graph loaded.")
