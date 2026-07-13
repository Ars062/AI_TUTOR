import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

st.set_page_config(page_title="AI Tutor", layout="wide")

st.title("AI Tutor")
st.caption("Knowledge-Grounded + Chain-of-Thought Tutoring System")

if "initialized" not in st.session_state:
    with st.spinner("Loading vector index..."):
        from src.rag.embed_documents import load_index, build_vector_index, save_index
        index, documents, filenames = load_index()
        if index.ntotal == 0:
            index, documents, filenames = build_vector_index()
            save_index(index, documents, filenames)
        st.session_state.index = index
        st.session_state.documents = documents
        st.session_state.filenames = filenames

    try:
        from src.kg.kg_loader import load_kg
        from src.config import KG_CSV_PATH
        load_kg(KG_CSV_PATH)
    except Exception:
        pass

    st.session_state.initialized = True
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    use_cot = st.checkbox("Chain-of-Thought reasoning", value=True)
    show_debug = st.checkbox("Show debug info", value=False)

    if st.button("Clear conversation"):
        from src.tutor.tutor_engine import clear_memory
        clear_memory()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("System Info")
    st.text(f"Documents loaded: {len(st.session_state.documents)}")
    st.text(f"FAISS index size: {st.session_state.index.ntotal}")
    from src.tutor.tutor_engine import get_memory
    mem = get_memory()
    if mem.topics_covered:
        st.text(f"Topics covered: {', '.join(list(mem.topics_covered)[:5])}")

    st.divider()
    st.subheader("Quick Questions")
    if st.button("What is recursion?"):
        st.session_state.input_question = "What is recursion?"
    if st.button("Explain binary search"):
        st.session_state.input_question = "Explain binary search"

question = st.chat_input("Ask a question about computer science...")

if "input_question" in st.session_state:
    question = st.session_state.input_question
    del st.session_state.input_question

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("debug") and show_debug:
            with st.expander("Debug Info"):
                st.json(msg["debug"])

if question and question.strip():
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                from src.tutor.tutor_engine import ask_tutor
                answer, debug_info = ask_tutor(
                    question=question,
                    index=st.session_state.index,
                    documents=st.session_state.documents,
                    filenames=st.session_state.filenames,
                    session_id="default",
                    use_cot=use_cot,
                )
                st.markdown(answer)

                if show_debug:
                    with st.expander("Debug Info"):
                        debug_display = {
                            "kg_context_length": len(debug_info["kg_context"]),
                            "doc_context_length": len(debug_info["doc_context"]),
                            "kg_guided_docs_length": len(debug_info["kg_guided_docs"]),
                            "entities_found": debug_info["entities"][:10],
                            "history_count": len(debug_info["memory"]["history"]),
                            "topics_covered": debug_info["memory"]["topics_covered"][:10],
                        }
                        st.json(debug_display)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                answer = f"Error: {e}"
                debug_info = {}

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "debug": debug_info,
        })

elif question is not None and not question.strip():
    st.warning("Please enter a question.")
