import google.generativeai as genai
from dotenv import load_dotenv
from src.kg.kg_query import query_kg
from src.rag.vector_search import search_docs
from src.prompts.prompt_builder import build_prompt
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-pro")


def ask_tutor(question, index, documents):

    kg_context = query_kg(question)

    doc_context = search_docs(question, index, documents)

    prompt = build_prompt(question, kg_context, doc_context)

    response = model.generate_content(prompt)

    return response.text
