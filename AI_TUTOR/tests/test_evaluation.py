import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.evaluation_metrics import (
    similarity,
    extractive_coverage,
    contains_key_concepts,
    evaluate_cot_steps,
    evaluate_response,
)


def test_similarity_identical():
    assert similarity("hello world", "hello world") == 1.0


def test_similarity_different():
    assert similarity("abc", "xyz") < 1.0


def test_extractive_coverage_full():
    cov = extractive_coverage("recursion is a technique", "recursion is a technique")
    assert cov > 0.9


def test_extractive_coverage_empty():
    assert extractive_coverage("", "some reference") == 0.0


def test_contains_key_concepts_all_found():
    result = contains_key_concepts(
        "Recursion requires a base case",
        ["Recursion", "base case"],
    )
    assert result["concept_coverage"] == 1.0


def test_contains_key_concepts_none_found():
    result = contains_key_concepts(
        "Hello world",
        ["Recursion", "Base Case"],
    )
    assert result["concept_coverage"] == 0.0


def test_evaluate_cot_steps():
    answer = "Step 1: Understand\nStep 2: Reason\nStep 3: Conclude"
    result = evaluate_cot_steps(answer)
    assert result["num_steps"] == 3
    assert result["has_steps"] is True


def test_evaluate_cot_steps_none():
    result = evaluate_cot_steps("Just an answer without steps")
    assert result["num_steps"] == 0
    assert result["has_steps"] is False


def test_evaluate_response_full():
    result = evaluate_response(
        answer="Recursion is a technique where a function calls itself.",
        reference="Recursion is a programming technique.",
        concepts=["Recursion", "function", "base case"],
    )
    assert "similarity" in result
    assert "extractive_coverage" in result
    assert "concept_analysis" in result
    assert "cot_analysis" in result
    assert "length" in result


if __name__ == "__main__":
    test_similarity_identical()
    test_similarity_different()
    test_extractive_coverage_full()
    test_extractive_coverage_empty()
    test_contains_key_concepts_all_found()
    test_contains_key_concepts_none_found()
    test_evaluate_cot_steps()
    test_evaluate_cot_steps_none()
    test_evaluate_response_full()
    print("All evaluation tests passed!")
