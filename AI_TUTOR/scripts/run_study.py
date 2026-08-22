import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.study_harness import (
    load_quiz,
    split_pre_post,
    score_answers,
    score_answer,
    analyze_study,
    analyze_groups,
    analyze_survey,
    assign_group,
    format_study_report,
)
from src.tutor.tutor_engine import ask_tutor, clear_memory
from src.rag.embed_documents import build_vector_index


OUT_DIR = os.path.join("data", "study")
RESPONSE_SAVE = os.path.join(OUT_DIR, "responses.json")


def _run_quiz_set(index, documents, filenames, items, use_kg, use_cot, group_tag):
    session_id = f"pilot-{group_tag}"
    clear_memory(session_id)
    label = "B(full)" if (use_kg and use_cot) else "A(baseline)"
    scores = []
    for i, item in enumerate(items):
        try:
            answer, debug = ask_tutor(
                item["question"], index, documents, filenames,
                session_id=session_id, use_cot=use_cot, use_kg=use_kg,
            )
        except Exception as e:
            answer = f"ERROR: {e}"
            debug = {}
        score = score_answer(answer, item["reference"], item.get("concepts"))
        scores.append(score)
        print(f"  [{i+1}/{len(items)}][{label:>10}] {item['question'][:38]:<40} score={score}")
    return scores


def run_pilot(n_sessions=3, save=None):
    """Automated pilot simulating the proposal's two-group design with no humans:
    Group A answers via RAG-only baseline (no KG, no CoT);
    Group B answers via the full KG-RAG + CoT system.
    Scores are compared between groups on identical questions."""
    quiz = load_quiz()
    pre_items, post_items = split_pre_post(quiz)

    print(f"Pre-quiz: {len(pre_items)} Q (unused in pilot), Post-quiz: {len(post_items)} Q per group\n")

    index, documents, filenames = build_vector_index()

    scores_a, scores_b = [], []
    for session in range(n_sessions):
        print(f"--- Session {session + 1}/{n_sessions} ---")
        print("Group A: RAG-only baseline")
        scores_a.extend(_run_quiz_set(index, documents, filenames, post_items,
                                      use_kg=False, use_cot=False, group_tag="A"))
        print("Group B: full KG-RAG + CoT")
        scores_b.extend(_run_quiz_set(index, documents, filenames, post_items,
                                      use_kg=True, use_cot=True, group_tag="B"))

    mean_a = round(sum(scores_a) / len(scores_a), 2) if scores_a else 0.0
    mean_b = round(sum(scores_b) / len(scores_b), 2) if scores_b else 0.0

    report = {
        "n_questions_per_group": len(post_items),
        "n_sessions": n_sessions,
        "group_a_mean": mean_a,
        "group_b_mean": mean_b,
        "b_higher": bool(mean_b > mean_a),
    }
    if len(scores_a) >= 2 and len(scores_b) >= 2:
        from scipy import stats
        t, p = stats.ttest_ind(scores_b, scores_a, equal_var=False)
        report["gain_diff_t"] = round(float(t), 3)
        report["gain_diff_p"] = round(float(p), 4)
        report["b_significantly_better"] = bool(p < 0.05 and mean_b > mean_a)

    lines = ["=" * 50, "Pilot Report (Group A vs Group B)", "=" * 50]
    for k, v in report.items():
        lines.append(f"  {k:<22}: {v}")
    print("\n".join(lines))

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(save, "w", encoding="utf-8") as f:
            json.dump({"report": report}, f, indent=2)
        print(f"\nSaved to {save}")
    return report


def collect_session(participant_id, pre_answers, post_answers, survey_responses, save=RESPONSE_SAVE):
    """Save a single real human session (answers are dicts {question_id: text}).

    The participant is deterministically assigned to Group A or B and the
    group is stored with the record for later between-group analysis."""
    quiz = load_quiz()
    pre_items, post_items = split_pre_post(quiz)
    group = assign_group(participant_id)
    pre_scores = [r["score"] for r in score_answers(pre_answers, pre_items)]
    post_scores = [r["score"] for r in score_answers(post_answers, post_items)]
    report = analyze_study(pre_scores, post_scores, group=group)
    survey_agg = analyze_survey(survey_responses)

    os.makedirs(OUT_DIR, exist_ok=True)
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "participant_id": participant_id,
        "group": group,
        "pre_answers": pre_answers,
        "post_answers": post_answers,
        "survey": survey_responses,
        "report": report,
        "survey_agg": survey_agg,
    }
    with open(save, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    print(format_study_report(report, report, survey_agg))
    return record


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Tutor human-study harness")
    parser.add_argument("--pilot", action="store_true", help="Run automated pilot demo")
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    if args.pilot:
        run_pilot(n_sessions=args.sessions, save=args.save or os.path.join(OUT_DIR, "pilot_report.json"))
    else:
        parser.print_help()