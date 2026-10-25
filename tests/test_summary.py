"""Tests for the field-based summary (no LLM needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.summary as sm

FIELDS = ["Model", "Benchmark", "Score"]
ROWS = [
    {"paper": "r1", "Model": "Claude-3.5", "Benchmark": "MMLU", "Score": "88.3"},
    {"paper": "r1", "Model": "GPT-4o", "Benchmark": "MMLU", "Score": "87.2"},
    {"paper": "r1", "Model": "DeepSeek-R1", "Benchmark": "MMLU", "Score": "90.8"},
    {"paper": "r1", "Model": "Claude-3.5", "Benchmark": "GPQA", "Score": "65.0"},
    {"paper": "r1", "Model": "GPT-4o", "Benchmark": "GPQA", "Score": "49.9"},
    {"paper": "r1", "Model": "DeepSeek-R1", "Benchmark": "GPQA", "Score": "71.5"},
]


def test_facts_follow_the_users_fields():
    f = sm.build_facts(ROWS, FIELDS)
    assert (f["entity_column"], f["group_column"], f["score_column"]) == ("Model", "Benchmark", "Score")
    assert f["better_is"] == "higher"
    best = {g["Benchmark"]: g["best"]["Model"] for g in f["by_group"]}
    assert best == {"MMLU": "DeepSeek-R1", "GPQA": "DeepSeek-R1"}
    assert f["most_often_best"][0] == {"Model": "DeepSeek-R1", "times_best": 2}
    assert f["distinct_values"] == {"Model": 3, "Benchmark": 2}


def test_error_rates_mean_lower_is_better():
    rows = [{"Model": "ResNet-110", "Dataset": "CIFAR-10", "Accuracy": "6.43% error"},
            {"Model": "DenseNet", "Dataset": "CIFAR-10", "Accuracy": "3.46% error"}]
    f = sm.build_facts(rows, ["Model", "Dataset", "Accuracy"])
    assert f["better_is"] == "lower" and f["by_group"][0]["best"]["Model"] == "DenseNet"


def test_focus_filters_rows():
    assert len(sm.filter_focus(ROWS, FIELDS, "gpqa")) == 3
    assert len(sm.filter_focus(ROWS, FIELDS, "claude")) == 2
    assert sm.filter_focus(ROWS, FIELDS, "") == ROWS


def test_summary_flags_numbers_not_in_the_data():
    sm.complete = lambda s, u: "DeepSeek-R1 leads MMLU with 90.8 and GPQA with 71.5, about 12.4 points ahead."
    out = sm.summarize(ROWS, FIELDS, excluded=1)
    assert out["unsupported"] == ["12.4"], out["unsupported"]
    assert out["rows_used"] == 6 and out["facts"]["excluded_unverified_rows"] == 1


def test_focus_with_no_matches_returns_empty():
    sm.complete = lambda s, u: "should not be called"
    out = sm.summarize(ROWS, FIELDS, excluded=0, focus="Gemini")
    assert out["text"] == "" and out["rows_used"] == 0


def test_computed_difference_is_flagged_even_if_a_count_matches():
    rows = ROWS + [{"paper": "r1", "Model": f"M{i}", "Benchmark": "MMLU", "Score": "80.0"} for i in range(3)]
    facts = sm.build_facts(rows, FIELDS)          # MMLU now has 6 results, so "6" exists as a count
    text = "DeepSeek-R1 tops MMLU at 90.8, about 6 points ahead of the field; 6 results in total."
    assert sm.unsupported_numbers(text, facts, rows) == ["6"]   # the measurement "6 points" only


def test_real_scores_as_measurements_are_fine():
    facts = sm.build_facts(ROWS, FIELDS)
    assert sm.unsupported_numbers("DeepSeek-R1 scores 90.8% on MMLU and 71.5 points on GPQA.", facts, ROWS) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
