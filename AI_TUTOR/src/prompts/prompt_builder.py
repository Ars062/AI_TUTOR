import re

QUESTION_TYPES = {
    "conceptual": "conceptual",
    "procedural": "procedural",
    "application": "application",
}


def _classify_question(question):
    q = question.lower()
    if any(w in q for w in ["what is", "define", "explain", "what does", "describe"]):
        return "conceptual"
    if any(w in q for w in ["how to", "how do", "steps", "process", "write", "implement"]):
        return "procedural"
    if any(w in q for w in ["example", "apply", "use", "solve", "find"]):
        return "application"
    return "conceptual"


def _s2a_filter(question):
    noise_patterns = [
        r"(?i)\b(ignore|forget|disregard)\s+(the\s+)?(above|previous|prior)\s+(instructions|context|prompt)",
        r"(?i)\b(skip|ignore)\s+(all\s+)?(previous|prior)",
        r"(?i)\b(you are|act as|pretend)",
    ]
    cleaned = question
    for pattern in noise_patterns:
        cleaned = re.sub(pattern, "", cleaned).strip()
    return cleaned


LEARNER_LEVELS = ("beginner", "intermediate", "advanced")

_LEVEL_GUIDES = {
    "beginner": (
        "Assume no prior knowledge of this topic. Use simple language and everyday "
        "analogies, define every technical term the first time it appears, and keep "
        "each reasoning step short and friendly."
    ),
    "intermediate": (
        "Assume basic computer-science familiarity. Balance intuition with correct "
        "terminology, and briefly define only genuinely advanced terms."
    ),
    "advanced": (
        "Assume a solid CS background. Skip elementary definitions, go deeper into "
        "underlying mechanics, edge cases, correctness arguments and complexity "
        "analysis, using precise terminology throughout."
    ),
}


def _level_block(learner_level):
    """Adaptive-depth directive per proposal 4.2 ('vary CoT length and detail
    based on learner level'). Returns '' for unknown/absent levels."""
    if not learner_level:
        return ""
    key = str(learner_level).strip().lower()
    guide = _LEVEL_GUIDES.get(key)
    if not guide:
        return ""
    return f"\n## Learner Level: {key.capitalize()}\n{guide}\n"


def build_prompt(question, kg_context, doc_context, cot_steps=None, learner_level=None):
    qtype = _classify_question(question)
    cleaned_q = _s2a_filter(question)
    level = _level_block(learner_level)

    cot_instruction = ""
    if cot_steps:
        steps_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(cot_steps))
        cot_instruction = f"""
Chain-of-Thought Steps:
{steps_text}

Please follow these steps carefully in your response, validating each step against the provided knowledge.
"""
    else:
        cot_instruction = """
Please think step-by-step and show your reasoning clearly. Break down your answer into logical steps.
"""

    type_guides = {
        "conceptual": "Provide a clear definition and explanation of the concept. Include key characteristics and relationships.",
        "procedural": "List the steps or procedure clearly. Explain each step and why it matters.",
        "application": "Work through an example or application. Show the process and reasoning behind each part.",
    }

    guide = type_guides.get(qtype, type_guides["conceptual"])

    prompt = f"""You are an AI tutor. Answer the student's question using the provided knowledge.

## Knowledge Graph Context
{kg_context if kg_context else "(No knowledge graph context available)"}

## Relevant Documents
{doc_context if doc_context else "(No document context available)"}

{cot_instruction}

## Question Type: {qtype.capitalize()}
{guide}
{level}
## Question
{cleaned_q}

## Instructions
- Base your answer on the provided context.
- If the context is insufficient, state what additional information would be needed.
- Use clear, educational language and address the learner directly as "you"; never refer to "the student".
- {cot_instruction.strip()}
"""

    return prompt


def build_ensemble_prompts(question, kg_context, doc_context):
    base = build_prompt(question, kg_context, doc_context)

    prompt_v2 = f"""You are a helpful AI tutor. Using the information below, answer the student's question thoroughly.

Knowledge:
{kg_context if kg_context else "N/A"}
Documents:
{doc_context if doc_context else "N/A"}

Question: {question}

Walk through your reasoning step by step, then provide the answer."""

    prompt_v3 = f"""As an expert tutor, I need you to explain this topic to a student.

Reference Material:
- Knowledge Graph: {kg_context if kg_context else "N/A"}
- Documents: {doc_context if doc_context else "N/A"}

Student asks: {question}

First, identify what the student needs to know. Then explain it clearly with examples."""

    return [base, prompt_v2, prompt_v3]


def build_cot_prompt(question, kg_context, doc_context, kg_guided_docs=None, learner_level=None):
    cleaned_q = _s2a_filter(question)
    level = _level_block(learner_level)

    extra_context = ""
    if kg_guided_docs:
        extra_context = f"\n## KG-Guided Context\n{kg_guided_docs}"

    prompt = f"""You are an AI tutor that uses chain-of-thought reasoning.

## Knowledge Graph Facts
{kg_context if kg_context else "(No knowledge graph context available)"}

## Document Context
{doc_context if doc_context else "(No document context available)"}
{extra_context}

## Question
{cleaned_q}
{level}
## Style Rule
Address the learner directly using "you". NEVER refer to "the student" in your answer.

## Chain-of-Thought Instructions
Please work through this problem step by step:

Step 1 - Understand: Restate the question in your own words and identify what needs to be explained.
Step 2 - Recall: List the relevant concepts, definitions, or procedures from the provided knowledge. Present them as a markdown table with EXACTLY two columns and header row "| Concept | What it means |", one concept per row.
Step 3 - Reason: Apply the concepts to answer the question, showing your work.
Step 4 - Verify: Check your answer against the knowledge graph and documents for accuracy.
Step 5 - Conclude: Present the final answer clearly and concisely, summarizing the key points you need to understand.
"""

    return prompt
