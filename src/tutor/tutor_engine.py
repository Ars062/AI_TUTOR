import google.generativeai as genai

from src.kg.kg_query import query_kg
from src.rag.vector_search import search_docs
from src.prompts.prompt_builder import build_prompt


genai.configure(api_key="YOUR_GEMINI_API_KEY")

model = genai.GenerativeModel("gemini-1.5-pro")


def ask_tutor(question, index, documents):

    kg_context = query_kg(question)

    doc_context = search_docs(question, index, documents)

    prompt = build_prompt(question, kg_context, doc_context)

    response = model.generate_content(prompt)

    return response.text
