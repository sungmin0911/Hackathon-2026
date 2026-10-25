"""Tests for the inconsistency detector."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.consistency import (
    detect_cross_paper_conflicts,
    detect_intra_paper_inconsistencies,
    annotate_rows_with_consistency,
)

FIELDS = ["Model", "Dataset", "Accuracy"]


def test_intra_paper_detects_abstract_vs_table():
    rows = [
        {"Model": "ResNet-110", "Dataset": "CIFAR-10", "Accuracy": "6.43% error", "quote": "abstract"},
        {"Model": "ResNet-110", "Dataset": "CIFAR-10", "Accuracy": "6.61% error", "quote": "table"},
    ]
    conflicts = detect_intra_paper_inconsistencies(rows, FIELDS)
    assert len(conflicts) == 1
    assert set(conflicts[0]["values"]) == {"6.43", "6.61"}
    assert conflicts[0]["type"] == "intra"


def test_intra_paper_no_false_positive_when_same_number():
    rows = [
        {"Model": "ResNet-110", "Dataset": "CIFAR-10", "Accuracy": "6.43% error"},
        {"Model": "ResNet-110", "Dataset": "CIFAR-10", "Accuracy": "6.43% error"},
    ]
    assert detect_intra_paper_inconsistencies(rows, FIELDS) == []


def test_cross_paper_conflict():
    rows = [
        {"paper": "PaperA", "Model": "ViT", "Dataset": "ImageNet", "Accuracy": "88.5%"},
        {"paper": "PaperB", "Model": "ViT", "Dataset": "ImageNet", "Accuracy": "87.1%"},
    ]
    conflicts = detect_cross_paper_conflicts(rows, FIELDS)
    assert len(conflicts) == 1
    assert conflicts[0]["type"] == "cross"
    assert "PaperA" in conflicts[0]["values_by_paper"]
    assert "PaperB" in conflicts[0]["values_by_paper"]


def test_annotate_adds_flag():
    rows = [
        {"Model": "X", "Dataset": "Y", "Accuracy": "1.0"},
        {"Model": "X", "Dataset": "Y", "Accuracy": "2.0"},
    ]
    annotated = annotate_rows_with_consistency(rows, FIELDS)
    assert all(r.get("inconsistency") == "intra-paper" for r in annotated)


# --- Regression tests for bugs found in review ---

def test_works_with_any_field_names():
    """Demo fields are Model, Benchmark, Score (no 'Dataset')."""
    rows = [{"paper": "A", "Model": "GPT-4o", "Benchmark": "MMLU", "Score": "87.2"},
            {"paper": "B", "Model": "GPT-4o", "Benchmark": "MMLU", "Score": "88.7"}]
    assert len(detect_cross_paper_conflicts(rows, ["Model", "Benchmark", "Score"])) == 1


def test_other_department_fields():
    rows = [{"paper": "A", "Drug": "Metformin", "Dose": "500mg", "Outcome": "-1.2% HbA1c"},
            {"paper": "B", "Drug": "Metformin", "Dose": "500mg", "Outcome": "-0.9% HbA1c"}]
    c = detect_cross_paper_conflicts(rows, ["Drug", "Dose", "Outcome"])
    assert len(c) == 1 and c[0]["key"] == {"Drug": "metformin", "Dose": "500mg"}


def test_uses_first_number_not_random_one():
    """'88.3 (5-shot)' must compare as 88.3, every run (was picked randomly from a set)."""
    rows = [{"paper": "A", "Model": "X", "Dataset": "MMLU", "Accuracy": "88.3 (5-shot)"},
            {"paper": "B", "Model": "X", "Dataset": "MMLU", "Accuracy": "88.3"}]
    assert detect_cross_paper_conflicts(rows, FIELDS) == []


def test_same_paper_twice_is_not_cross_conflict():
    rows = [{"paper": "A", "Model": "X", "Dataset": "Y", "Accuracy": "1.0"},
            {"paper": "A", "Model": "X", "Dataset": "Y", "Accuracy": "2.0"}]
    assert detect_cross_paper_conflicts(rows, FIELDS) == []
    assert len(detect_intra_paper_inconsistencies(rows, FIELDS)) == 1


def test_json_with_citation_brackets_in_quotes():
    import json
    from app.extractor import parse_json_array
    raw = "Here you go:\n```json\n" + json.dumps([
        {"Model": "Kimi-K2", "Score": "89.5", "quote": "assessed using MMLU [24], MMLU-Redux [18]", "page": 15},
        {"Model": "GPT-4.1", "Score": "90.4", "quote": "MMLU (EM) 89.5 89.4", "page": 16},
    ]) + "\n```"
    assert len(parse_json_array(raw)) == 2


def test_json_trailing_comma():
    from app.extractor import parse_json_array
    assert len(parse_json_array('[{"a": "1"}, {"a": "2"},]')) == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
