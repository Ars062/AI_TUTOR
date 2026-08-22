import sys
import os
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.embed_documents import build_vector_index
from src.tutor.tutor_engine import ask_tutor, clear_memory
from src.evaluation.evaluation_metrics import (
    run_batch_evaluation,
    format_report,
)


def run_evaluation(quiz_path=None, use_cot=True, save_output=None, heavy_metrics=True):
    if quiz_path is None:
        quiz_path = os.path.join("data", "evaluation", "quiz.json")

    with open(quiz_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    quiz = data["quiz"]

    print(f"Building vector index...")
    index, documents, filenames = build_vector_index()

    predictions = []
    references = []
    concepts_list = []
    timings = []

    for i, item in enumerate(quiz):
        q = item["question"]
        clear_memory()
        start = time.time()
        try:
            answer, debug = ask_tutor(
                question=q,
                index=index,
                documents=documents,
                filenames=filenames,
                use_cot=use_cot,
            )
        except Exception as e:
            answer = f"ERROR: {e}"
        elapsed = round(time.time() - start, 2)
        timings.append(elapsed)

        predictions.append(answer)
        references.append(item["reference"])
        concepts_list.append(item.get("concepts", []))

        print(f"[{i+1}/{len(quiz)}] ({elapsed}s) {q[:50]}...")

    report = run_batch_evaluation(predictions, references, heavy_metrics=heavy_metrics)
    report["use_cot"] = use_cot
    report["avg_latency_s"] = round(sum(timings) / len(timings), 2) if timings else None

    print()
    print(format_report(report))

    if save_output:
        with open(save_output, "w", encoding="utf-8") as f:
            json.dump({"report": report, "predictions": predictions, "references": references}, f, indent=2, default=str)
        print(f"\nSaved to {save_output}")

    return report


def _save_path(base, suffix):
    if not base:
        return None
    root, ext = os.path.splitext(base)
    return f"{root}_{suffix}{ext}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run AI Tutor evaluation")
    parser.add_argument("--cot", action="store_true", default=True, help="Use chain-of-thought (default: True)")
    parser.add_argument("--no-cot", dest="cot", action="store_false", help="Disable chain-of-thought")
    parser.add_argument("--save", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--quiz", type=str, default=None, help="Quiz JSON path")
    parser.add_argument("--fast", action="store_true", help="Skip BLEU/ROUGE/BERTScore (faster)")
    parser.add_argument("--compare", action="store_true", help="Run both CoT and non-CoT and compare")
    args = parser.parse_args()

    if args.compare:
        print("\n########## RUN WITH CHAIN-OF-THOUGHT ##########\n")
        report_cot = run_evaluation(quiz_path=args.quiz, use_cot=True, save_output=_save_path(args.save, "cot"), heavy_metrics=not args.fast)
        print("\n########## RUN WITHOUT CHAIN-OF-THOUGHT ##########\n")
        report_nocot = run_evaluation(quiz_path=args.quiz, use_cot=False, save_output=_save_path(args.save, "nocot"), heavy_metrics=not args.fast)

        print("\n\n########## COMPARISON ##########")
        a, b = report_cot["aggregate"], report_nocot["aggregate"]
        metrics = ["avg_bleu", "avg_rouge1_f1", "avg_rougeL_f1", "avg_bertscore_f1",
                   "avg_extractive_coverage", "avg_similarity",
                   "avg_logical_consistency", "avg_explainability"]
        print(f"{'Metric':<24} {'CoT':>8} {'No-CoT':>8} {'Diff':>8}")
        print("-" * 50)
        for m in metrics:
            va, vb = a.get(m), b.get(m)
            if va is None or vb is None:
                print(f"{m:<24} {str(va):>8} {str(vb):>8} {'n/a':>8}")
                continue
            delta = va - vb
            print(f"{m:<24} {va:>8.4f} {vb:>8.4f} {delta:>+8.4f}")
        print(f"\nAvg latency: CoT={report_cot.get('avg_latency_s')}s  No-CoT={report_nocot.get('avg_latency_s')}s")
    else:
        run_evaluation(quiz_path=args.quiz, use_cot=args.cot, save_output=args.save, heavy_metrics=not args.fast)
