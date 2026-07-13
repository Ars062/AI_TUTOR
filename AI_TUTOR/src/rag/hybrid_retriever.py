from src.rag.vector_search import search_docs
from src.kg.kg_query import query_kg, query_kg_entities
from src.config import TOP_K_DOCS


def hybrid_retrieve(question, index, documents, k=None, kg_keyword=None):
    if k is None:
        k = TOP_K_DOCS

    kg_context = query_kg(kg_keyword or question)

    doc_context = search_docs(question, index, documents, k=k)

    entities = query_kg_entities(kg_keyword or question)
    kg_guided_docs = ""
    if entities and documents:
        kg_entity_text = " ".join(entities)
        kg_guided_docs = search_docs(question + " " + kg_entity_text, index, documents, k=k)

    return {
        "kg_context": kg_context,
        "doc_context": doc_context,
        "kg_guided_docs": kg_guided_docs,
        "entities": entities,
    }
