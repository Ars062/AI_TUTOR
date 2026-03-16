import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from src.rag.embed_documents import build_vector_index
from src.tutor.tutor_engine import ask_tutor


# Build vector index
index, documents = build_vector_index("data/documents")

st.title("AI Tutor")

# Input box
question = st.text_input("Ask a question")

# Submit button
if st.button("Ask Tutor"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = ask_tutor(question, index, documents)

        st.subheader("Answer")
        st.write(answer)
