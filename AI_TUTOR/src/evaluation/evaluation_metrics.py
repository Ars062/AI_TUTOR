from difflib import SequenceMatcher
import re


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def extractive_coverage(hypothesis, reference):
    hyp_words = set(re.findall(r'\w+', hypothesis.lower()))
    ref_words = set(re.findall(r'\w+', reference.lower()))
    if not ref_words:
        return 0.0
    return len(hyp_words & ref_words) / len(ref_words)


def contains_key_concepts(answer, concepts):
    answer_lower = answer.lower()
    found = []
    for concept in concepts:
        if concept.lower() in answer_lower:
            found.append(concept)
    return {
        "total_concepts": len(concepts),
        "found_concepts": len(found),
        "concept_coverage": len(found) / len(concepts) if concepts else 0.0,
        "found": found,
        "missing": [c for c in concepts if c.lower() not in answer_lower],
    }


def evaluate_cot_steps(answer):
    step_pattern = re.compile(r'Step\s+\d+', re.IGNORECASE)
    steps = step_pattern.findall(answer)
    return {
        "num_steps": len(steps),
        "has_steps": len(steps) > 0,
    }


def evaluate_response(answer, reference=None, concepts=None):
    result = {}

    if reference:
        result["similarity"] = similarity(answer, reference)
        result["extractive_coverage"] = extractive_coverage(answer, reference)

    if concepts:
        result["concept_analysis"] = contains_key_concepts(answer, concepts)

    result["cot_analysis"] = evaluate_cot_steps(answer)
    result["length"] = len(answer.split())

    return result
