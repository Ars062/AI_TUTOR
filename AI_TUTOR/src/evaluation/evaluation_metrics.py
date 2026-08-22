import os
import re

os.environ.setdefault("PYTHONSAFEPATH", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from difflib import SequenceMatcher

# Heavy optional imports are lazily loaded and cached
_bleu_metric = None
_rouge_scorer = None
_bert_loaded = False


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


def extract_cot_steps(answer):
    """Split a CoT answer into (label, text) blocks. Returns [] if none."""
    pattern = re.compile(r"(?m)^\s*\*{0,2}\s*Step\s+\d+\s*[-–:.]?\s+([^\n*]+)\*{0,2}")
    matches = list(pattern.finditer(answer))
    if not matches:
        return []
    steps = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(answer)
        body = "\n".join(
            line.strip() for line in answer[body_start:body_end].splitlines() if line.strip()
        )
        steps.append({"label": f"Step {i + 1} - {title}", "text": body})
    return steps


def validate_cot_steps_against_kg(steps, kg_context):
    """Per-step KG validation: each step should reference a concept shown in the KG context.

    Returns per-step verdicts plus an overall grounding score (fraction of steps grounded).
    """
    if not kg_context or not kg_context.strip():
        return {
            "validated": None,
            "ungrounded_steps": [],
            "grounded_fraction": None,
            "note": "no KG context available for validation",
        }

    text = kg_context.lower()
    concepts = re.findall(r"[A-Za-z][A-Za-z0-9_+-]*", text)
    concept_set = set(c for c in concepts if len(c) > 2)

    results = []
    for s in steps:
        body_words = set(re.findall(r"[A-Za-z][A-Za-z0-9_+'-]*", s["text"].lower()))
        hits = concept_set & body_words
        results.append({"label": s["label"], "grounded": bool(hits), "matched": sorted(hits)[:8]})

    grounded = [r for r in results if r["grounded"]]
    return {
        "validated": True,
        "per_step": results,
        "ungrounded_steps": [r["label"] for r in results if not r["grounded"]],
        "grounded_fraction": round(len(grounded) / len(results), 2) if results else None,
    }


def score_logical_consistency(steps):
    """Heuristic Logical Consistency rubric score (0-5).

    Starts at 5, deducts for missing steps, contradictions, or jumps.
    NOTE: intended as a pre-screen; final scores should come from a human rubric.
    """
    score = 5.0
    if not steps:
        return {"score": 0.0, "notes": ["no CoT steps present"]}
    for r in steps:
        if not r["text"] or len(r["text"].split()) < 3:
            score -= 1.0
    if len(steps) < 3:
        score -= 1.0
    tracked = " ".join(r["text"].lower() for r in steps)
    for neg in ["i made a mistake", "ignore that", "this contradicts", "wait, that's wrong"]:
        if neg in tracked:
            score -= 0.5
    score = max(0.0, min(5.0, score))
    return {"score": round(score, 1), "num_steps": len(steps)}


def score_explainability(answer, has_graph_example=False):
    """Heuristic Explainability rubric score (1-5).

    Rewards definitions, examples, tables/lists, and explicit reasoning structure.
    NOTE: intended as a pre-screen; final scores should come from a human rubric.
    """
    score = 1.0
    lower = answer.lower()
    if re.search(r"is (a|an|the) |refers to|defined as|means that", lower):
        score += 1.0
    if re.search(r"example|for instance|such as", lower):
        score += 1.0
    if re.search(r"\*\*|(^|\n)\s*[-*] ", lower) or "|" in answer:
        score += 1.0
    if re.search(r"step \d", lower) or len(answer.split()) > 80:
        score += 1.0
    score = max(1.0, min(5.0, score))
    return {"score": round(score, 1)}


def _get_bleu_metric():
    global _bleu_metric
    if _bleu_metric is None:
        import evaluate
        _bleu_metric = evaluate.load("bleu")
    return _bleu_metric


def _get_rouge_scorer():
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer
        _rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
    return _rouge_scorer


def compute_bleu(predictions, references):
    """BLEU score using the `evaluate` library (nltk-backed)."""
    try:
        metric = _get_bleu_metric()
        refs = [[r] for r in references]
        result = metric.compute(predictions=predictions, references=refs)
        return {"bleu": round(result["bleu"], 4)}
    except Exception as e:
        return {"bleu": None, "error": str(e)}


def compute_rouge(predictions, references):
    """ROUGE scores using the rouge_score package (lightweight, cached)."""
    try:
        scorer = _get_rouge_scorer()
        agg = {k: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
               for k in ["rouge1", "rouge2", "rougeL"]}
        n = len(predictions)
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for k in agg:
                agg[k]["precision"] += scores[k].precision
                agg[k]["recall"] += scores[k].recall
                agg[k]["fmeasure"] += scores[k].fmeasure
        if n == 0:
            return {k: v for k, v in agg.items()}
        for k in agg:
            for m in agg[k]:
                agg[k][m] = round(agg[k][m] / n, 4)
        return agg
    except Exception as e:
        return {"error": str(e)}


def compute_bertscore(predictions, references):
    """BERTScore (P/R/F1) via huggingface bert-score library (lightweight model)."""
    try:
        import bert_score
        P, R, F1 = bert_score.score(
            predictions, references, lang="en", verbose=False,
            model_type="distilbert-base-uncased",
        )
        return {
            "bertscore_precision": round(float(P.mean()), 4),
            "bertscore_recall": round(float(R.mean()), 4),
            "bertscore_f1": round(float(F1.mean()), 4),
        }
    except Exception as e:
        return {"bertscore": None, "error": str(e)}


def evaluate_response(answer, reference=None, concepts=None, heavy_metrics=True):
    """Evaluate a single answer. heavy_metrics=False keeps it fast (no BLEU/ROUGE/BERTScore)."""
    result = {}

    if reference:
        result["similarity"] = round(similarity(answer, reference), 4)
        result["extractive_coverage"] = round(extractive_coverage(answer, reference), 4)

    if concepts:
        result["concept_analysis"] = contains_key_concepts(answer, concepts)

    result["cot_analysis"] = evaluate_cot_steps(answer)
    result["length"] = len(answer.split())

    if reference and heavy_metrics:
        result["bleu"] = compute_bleu([answer], [reference]).get("bleu")
        result["rouge"] = compute_rouge([answer], [reference])
        result["bertscore"] = compute_bertscore([answer], [reference])

    return result


def run_batch_evaluation(predictions, references, concepts_list=None, heavy_metrics=True):
    """Evaluate a batch of tutor answers against reference answers."""
    report = {
        "n": len(predictions),
        "samples": [],
        "aggregate": {},
    }
    bleu_scores = []
    rouge1 = []
    rougeL = []
    bert_f1 = []
    cov_scores = []
    sim_scores = []
    logical_scores = []
    explain_scores = []

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        sample = evaluate_response(pred, reference=ref, heavy_metrics=heavy_metrics)
        report["samples"].append(sample)

        if sample.get("bleu") is not None:
            bleu_scores.append(sample["bleu"])
        if "rouge" in sample and "rouge1" in sample["rouge"]:
            rouge1.append(sample["rouge"]["rouge1"]["fmeasure"])
            rougeL.append(sample["rouge"]["rougeL"]["fmeasure"])
        if "bertscore" in sample and "bertscore_f1" in sample["bertscore"]:
            bert_f1.append(sample["bertscore"]["bertscore_f1"])
        cov_scores.append(sample.get("extractive_coverage", 0.0))
        sim_scores.append(sample.get("similarity", 0.0))

        steps = extract_cot_steps(pred)
        sample["logical_consistency"] = score_logical_consistency(steps)
        sample["explainability"] = score_explainability(pred)
        logical_scores.append(sample["logical_consistency"]["score"])
        explain_scores.append(sample["explainability"]["score"])

    def avg(vals):
        return round(sum(vals) / len(vals), 4) if vals else None

    report["aggregate"] = {
        "avg_bleu": avg(bleu_scores),
        "avg_rouge1_f1": avg(rouge1),
        "avg_rougeL_f1": avg(rougeL),
        "avg_bertscore_f1": avg(bert_f1),
        "avg_extractive_coverage": avg(cov_scores),
        "avg_similarity": avg(sim_scores),
        "avg_logical_consistency": avg(logical_scores),
        "avg_explainability": avg(explain_scores),
    }
    return report


def format_report(report):
    """Human-readable summary of a batch evaluation report."""
    agg = report["aggregate"]
    lines = []
    lines.append("=" * 50)
    lines.append("AI Tutor Evaluation Report")
    lines.append(f"Samples evaluated: {report['n']}")
    lines.append("=" * 50)
    headers = [
        ("BLEU", "avg_bleu"),
        ("ROUGE-1 F1", "avg_rouge1_f1"),
        ("ROUGE-L F1", "avg_rougeL_f1"),
        ("BERTScore F1", "avg_bertscore_f1"),
        ("Extractive Coverage", "avg_extractive_coverage"),
        ("Similarity", "avg_similarity"),
        ("Logical Consistency (rubric)", "avg_logical_consistency"),
        ("Explainability (rubric)", "avg_explainability"),
    ]
    for label, key in headers:
        lines.append(f"{label:<24}: {agg.get(key)}")
    return "\n".join(lines)