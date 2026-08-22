import hashlib
import json
import os
from collections import defaultdict

from src.evaluation.evaluation_metrics import (
    extractive_coverage,
    contains_key_concepts,
)


def load_quiz(quiz_path="data/evaluation/quiz.json"):
    path = os.path.abspath(quiz_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["quiz"]


def split_pre_post(quiz, seed=None):
    """Split quiz into pre/post sets with balanced topic coverage."""
    by_topic = defaultdict(list)
    for item in quiz:
        by_topic[item["topic"]].append(item)
    pre, post = [], []
    for topic in sorted(by_topic):
        items = by_topic[topic]
        for i, item in enumerate(items):
            (pre if i % 2 == 0 else post).append(item)
    return pre, post


def score_answer(student_answer, reference, concepts):
    """Automatic proxy score (0-100) using extractive coverage + concept coverage."""
    cov = extractive_coverage(student_answer or "", reference)
    concepts_result = contains_key_concepts(student_answer or "", concepts or [])
    coverage_score = concepts_result["concept_coverage"] * 100
    return round(0.5 * cov * 100 + 0.5 * coverage_score, 1)


def score_answers(answers, quiz_items):
    return [
        {"id": item.get("id"), "topic": item.get("topic"), "score": score_answer(
            answers.get(str(i)), item["reference"], item.get("concepts"))}
        for i, item in enumerate(quiz_items)
    ]


def analyze_study(pre_scores, post_scores, group="B"):
    """Compute learning gain. pre_scores/post_scores are lists of 0-100 scores.

    If two groups given (A/B), use ttest. Here single group -> pair-wise gain.
    """
    deltas = [post - pre for pre, post in zip(pre_scores, post_scores)]
    pre_avg = sum(pre_scores) / len(pre_scores) if pre_scores else 0.0
    post_avg = sum(post_scores) / len(post_scores) if post_scores else 0.0
    gain = post_avg - pre_avg
    rel_gain = (gain / pre_avg * 100) if pre_avg else None

    report = {
        "pre_mean": round(pre_avg, 2),
        "post_mean": round(post_avg, 2),
        "absolute_gain": round(gain, 2),
        "relative_gain_pct": round(rel_gain, 2) if rel_gain is not None else None,
        "n": len(deltas),
    }
    if len(deltas) >= 2:
        from scipy import stats
        t, p = stats.ttest_rel(pre_scores, post_scores)
        report["paired_t"] = round(float(t), 3)
        report["p_value"] = round(float(p), 4)
        report["significant"] = bool(p < 0.05)
    return report


def assign_group(participant_id, seed=2026):
    """Deterministic 50/50 assignment: same id always gets the same group.

    Group A = RAG-only baseline tutor (no KG, no CoT).
    Group B = full system (KG-RAG + CoT + prompt engineering).
    """
    digest = hashlib.sha256(f"{seed}:{participant_id}".encode("utf-8")).hexdigest()
    return "A" if int(digest[:16], 16) % 2 == 0 else "B"


def analyze_groups(pre_a, post_a, pre_b, post_b):
    """Compare learning gains between Group A and Group B (proposal §5.2).

    Uses an independent t-test on per-participant gains; significant result
    with higher B mean supports the proposal's success criterion.
    """
    gain_a = [p - q for q, p in zip(pre_a, post_a)]
    gain_b = [p - q for q, p in zip(pre_b, post_b)]

    report = {
        "group_a": analyze_study(pre_a, post_a, group="A"),
        "group_b": analyze_study(pre_b, post_b, group="B"),
        "n_a": len(gain_a),
        "n_b": len(gain_b),
    }

    if len(gain_a) >= 2 and len(gain_b) >= 2:
        from scipy import stats
        t, p = stats.ttest_ind(gain_b, gain_a, equal_var=False)
        mean_a = sum(gain_a) / len(gain_a)
        mean_b = sum(gain_b) / len(gain_b)
        report["gain_mean_a"] = round(mean_a, 2)
        report["gain_mean_b"] = round(mean_b, 2)
        report["gain_diff_t"] = round(float(t), 3)
        report["gain_diff_p"] = round(float(p), 4)
        report["b_significantly_better"] = bool(p < 0.05 and mean_b > mean_a)
    return report


SURVEY_TEMPLATE = {
    "title": "AI Tutor Post-Session Survey",
    "scale": "1 = Strongly disagree, 5 = Strongly agree",
    "items": {
        "trust": "I trusted the accuracy of the tutor's answers.",
        "clarity": "The explanations were clear and easy to understand.",
        "satisfaction": "I am satisfied with the tutoring session.",
        "learning_gain": "I feel I learned more than with a standard chatbot.",
        "reasoning_visible": "Seeing the tutor's step-by-step reasoning helped me understand.",
        "no_bias": "The tutor's responses seemed fair and unbiased.",
        "predictability": "The tutor produced consistent, predictable answers.",
    },
}


def analyze_survey(responses):
    """responses: list of dicts {question: rating (1-5)}. Returns per-item means + overall."""
    items = SURVEY_TEMPLATE["items"]
    agg = {}
    for key in items:
        vals = [r.get(key) for r in responses if r.get(key) is not None]
        agg[key] = round(sum(vals) / len(vals), 2) if vals else None
    valid = [v for v in agg.values() if v is not None]
    agg["overall_mean"] = round(sum(valid) / len(valid), 2) if valid else None
    return agg


def format_study_report(pre_report, post_report, survey_agg=None):
    lines = []
    lines.append("=" * 50)
    lines.append("Human Study Report")
    lines.append("=" * 50)
    lines.append(str(pre_report))
    lines.append(str(post_report))
    if survey_agg:
        lines.append("\nSurvey (1-5, higher = better):")
        for k, v in survey_agg.items():
            lines.append(f"  {k:<18}: {v}")
    return "\n".join(lines)