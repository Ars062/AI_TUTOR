import os
import re
import time
from groq import Groq
from dotenv import load_dotenv

from src.rag.hybrid_retriever import hybrid_retrieve
from src.prompts.prompt_builder import build_prompt, build_cot_prompt, build_ensemble_prompts, LEARNER_LEVELS
from src.evaluation.evaluation_metrics import extract_cot_steps, validate_cot_steps_against_kg
from src.config import GROQ_API_KEY, LLM_MODEL, USE_ENSEMBLE, MAX_HISTORY

load_dotenv()

client = Groq(api_key=GROQ_API_KEY)


# Content-safety guard (9.1): block clearly harmful / off-topic-for-tutoring intents
_UNSAFE_PATTERNS = [
    r"(?i)\bhow (to|do i) (build|make|create) (a|an)?\s*(bomb|weapon|explosive)\b",
    r"(?i)\b(suicide|self[- ]harm|harm myself)\b",
    r"(?i)\bhow (to )?(kill|murder|assault|hurt) \w+\b",
    r"(?i)\b(drugs? recipe|synthesize\s+\w+ (drug|narcotic))\b",
]


def _is_unsafe(question):
    return any(p for p in _UNSAFE_PATTERNS if p and __import__("re").search(p, question))


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


def _call_llm(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            wait = None
            msg = str(e)
            if "rate_limit_exceeded" in msg:
                wait = _parse_retry_seconds(msg)
                wait = max(5, min(wait, 600))
            if attempt == max_retries - 1 or wait is None:
                raise
            print(f"  rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _parse_retry_seconds(msg):
    """Extract seconds from Groq's 'try again in 14m11.04s' message."""
    try:
        m = re.search(r"try again in (\d+)m(\d+\.?\d*)s", msg)
        if m:
            return int(m.group(1)) * 60 + float(m.group(2))
        m = re.search(r"try again in (\d+\.?\d*)s", msg)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return 30


def ask_tutor(question, index, documents, filenames=None, session_id="default", use_cot=True, use_kg=True, learner_level=None):
    from src.config import ENABLE_CONTENT_SAFETY

    if ENABLE_CONTENT_SAFETY and _is_unsafe(question):
        refusal = ("I'm your tutor, but this request is outside the scope of safe tutoring. "
                   "I can't help with that. If you're trying to learn computer science, "
                   "just ask me about the topic!")
        return refusal, {"content_safety": "blocked", "memory": _get_memory(session_id).to_dict()}

    memory = _get_memory(session_id)
    if learner_level and str(learner_level).strip().lower() in LEARNER_LEVELS:
        memory.student_level = str(learner_level).strip().lower()

    retrieval = hybrid_retrieve(question, index, documents, use_kg=use_kg)

    if use_cot:
        prompt = build_cot_prompt(
            question=question,
            kg_context=retrieval["kg_context"],
            doc_context=retrieval["doc_context"],
            kg_guided_docs=retrieval["kg_guided_docs"],
            learner_level=memory.student_level,
        )
    else:
        prompt = build_prompt(
            question=question,
            kg_context=retrieval["kg_context"],
            doc_context=retrieval["doc_context"],
            learner_level=memory.student_level,
        )
        prompt += (
            "\n\n## Response Style\n"
            "Answer the student directly and concisely in a short paragraph. "
            "Do NOT write step-by-step sections, numbered procedures, or tables."
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

    cot_steps = extract_cot_steps(final_response)
    kg_validation = validate_cot_steps_against_kg(cot_steps, retrieval["kg_context"])

    return final_response, {
        "kg_context": retrieval["kg_context"],
        "doc_context": retrieval["doc_context"],
        "kg_guided_docs": retrieval["kg_guided_docs"],
        "entities": list(retrieval.get("entities", set())),
        "memory": memory.to_dict(),
        "cot_steps": cot_steps,
        "cot_validation": kg_validation,
        "content_safety": "passed",
        "use_cot": use_cot,
        "use_kg": use_kg,
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
