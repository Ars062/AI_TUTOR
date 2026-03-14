def build_prompt(question, kg_context, doc_context):

    prompt = f"""

You are an AI tutor.

Use the knowledge graph and documents to answer.

Knowledge Graph:
{kg_context}

Documents:
{doc_context}

Question:
{question}

Explain step-by-step.

"""

    return prompt
