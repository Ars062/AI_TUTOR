import streamlit as st

from src.rag.embed_documents import build_vector_index
from src.tutor.tutor_engine import ask_tutor


index, documents = build_vector_index("data/documents")


st.title("AI Tutor")

question = st.text_input("Ask a question")

if question:
    answer = ask_tutor(question, index, documents)

    st.write(answer)
