import pandas as pd
from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


def _get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _neo4j_is_reachable():
    import socket
    try:
        host = NEO4J_URI.replace("bolt://", "").replace("https://", "").replace("http://", "")
        if ":" in host:
            host, port = host.split(":")
            port = int(port)
        else:
            port = 7687
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        result = s.connect_ex((host, port))
        s.close()
        return result == 0
    except Exception:
        return False


def load_kg(csv_path=None):
    from src.config import KG_CSV_PATH
    if csv_path is None:
        csv_path = KG_CSV_PATH

    if not _neo4j_is_reachable():
        print("Neo4j is not running. Skipping KG load.")
        return

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Warning: KG CSV not found at {csv_path}")
        return
    except Exception as e:
        print(f"Warning: could not read KG CSV: {e}")
        return

    required_cols = {"source", "relation", "target"}
    if not required_cols.issubset(df.columns):
        print(f"Warning: KG CSV missing columns {required_cols - set(df.columns)}")
        return

    try:
        driver = _get_driver()
    except Exception as e:
        print(f"Warning: could not connect to Neo4j: {e}")
        return

    try:
        with driver.session() as session:
            for _, row in df.iterrows():
                source = str(row["source"]).strip()
                relation = str(row["relation"]).strip()
                target = str(row["target"]).strip()

                if not source or not relation or not target:
                    continue

                rel_type = relation.upper().replace(" ", "_")

                query = (
                    "MERGE (a:Concept {name: $source}) "
                    "MERGE (b:Concept {name: $target}) "
                    f"MERGE (a)-[:{rel_type}]->(b)"
                )

                session.run(query, source=source, target=target)

        print("Knowledge graph loaded successfully.")
    except Exception as e:
        print(f"Warning: Neo4j query failed: {e}")
    finally:
        driver.close()
