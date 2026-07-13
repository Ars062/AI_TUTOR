import os
from groq import Groq
from dotenv import load_dotenv

from src.rag.hybrid_retriever import hybrid_retrieve
from src.prompts.prompt_builder import build_prompt, build_cot_prompt, build_ensemble_prompts
from src.config import GROQ_API_KEY, LLM_MODEL, USE_ENSEMBLE, MAX_HISTORY

load_dotenv()

client = Groq(api_key=GROQ_API_KEY)


class ConversationMemory:
    def __init__(self):
        self.history = []
        self.topics_covered = set()
        self.student_level = "beginner"

    def add_exchange(self, question, answer):
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > MAX_HISTORY:
            self.history.pop(0)

    def get_context(self):
        if not self.history:
            return ""
        recent = self.history[-3:]
        lines = []
        for h in recent:
            lines.append(f"Student: {h['question']}")
            lines.append(f"Tutor: {h['answer'][:200]}...")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "history": self.history,
            "topics_covered": list(self.topics_covered),
            "student_level": self.student_level,
        }


_memory_store = {}


def _get_memory(session_id="default"):
    if session_id not in _memory_store:
        _memory_store[session_id] = ConversationMemory()
    return _memory_store[session_id]


def _call_llm(prompt):
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2048,
    )
    return response.choices[0].message.content


def ask_tutor(question, index, documents, filenames=None, session_id="default", use_cot=True):
    memory = _get_memory(session_id)

    retrieval = hybrid_retrieve(question, index, documents)

    if use_cot:
        prompt = build_cot_prompt(
            question=question,
            kg_context=retrieval["kg_context"],
            doc_context=retrieval["doc_context"],
            kg_guided_docs=retrieval["kg_guided_docs"],
        )
    else:
        prompt = build_prompt(
            question=question,
            kg_context=retrieval["kg_context"],
            doc_context=retrieval["doc_context"],
        )

    conv_context = memory.get_context()
    if conv_context:
        prompt = f"## Recent Conversation\n{conv_context}\n\n{prompt}"

    if USE_ENSEMBLE:
        ensemble_prompts = build_ensemble_prompts(
            question=question,
            kg_context=retrieval["kg_context"],
            doc_context=retrieval["doc_context"],
        )
        ensemble_prompts[0] = prompt

        responses = []
        for i, p in enumerate(ensemble_prompts):
            try:
                text = _call_llm(p)
                if text:
                    responses.append(text)
            except Exception as e:
                print(f"Ensemble prompt {i} failed: {e}")

        if responses:
            final_response = _aggregate_responses(responses, question)
        else:
            final_response = "Sorry, I encountered an error generating a response."
    else:
        try:
            final_response = _call_llm(prompt)
            if not final_response:
                final_response = "No response generated."
        except Exception as e:
            final_response = f"Sorry, an error occurred: {e}"

    memory.add_exchange(question, final_response)
    for ent in retrieval.get("entities", set()):
        memory.topics_covered.add(ent)

    return final_response, {
        "kg_context": retrieval["kg_context"],
        "doc_context": retrieval["doc_context"],
        "kg_guided_docs": retrieval["kg_guided_docs"],
        "entities": list(retrieval.get("entities", set())),
        "memory": memory.to_dict(),
    }


def _aggregate_responses(responses, original_question):
    agg_prompt = f"""Below are multiple attempts to answer the question: "{original_question}"

--- Attempt 1 ---
{responses[0]}

--- Attempt 2 ---
{responses[1] if len(responses) > 1 else "N/A"}

--- Attempt 3 ---
{responses[2] if len(responses) > 2 else "N/A"}

Synthesize the best answer from these attempts. Combine the strongest points, resolve any contradictions, and present a clear, accurate final answer."""

    try:
        return _call_llm(agg_prompt) or responses[0]
    except Exception:
        return responses[0]


def clear_memory(session_id="default"):
    _memory_store[session_id] = ConversationMemory()


def get_memory(session_id="default"):
    return _get_memory(session_id)
