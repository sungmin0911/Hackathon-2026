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


# --- Focus with column names and math symbols (Bivariate Simplex paper style) ---
SF = ["Sample size", "Measure", "Parameter", "Value"]
SROWS = [
    {"paper": "simplex", "Sample size": "50", "Measure": "Bias", "Parameter": "µ1", "Value": "0.001",
     "setting": "Table 3: θ = (0.5, 0.5, 2, 2, 1)"},          # micro sign, as the PDF text has it
    {"paper": "simplex", "Sample size": "50", "Measure": "Bias", "Parameter": "σ2 1", "Value": "−0.046",
     "setting": "Table 3: θ = (0.5, 0.5, 2, 2, 1)"},
    {"paper": "simplex", "Sample size": "50", "Measure": "Bias", "Parameter": "λ", "Value": "−0.150",
     "setting": "Table 3: θ = (0.5, 0.5, 2, 2, 1)"},
    {"paper": "simplex", "Sample size": "100", "Measure": "Bias", "Parameter": "µ1", "Value": "0.000",
     "setting": "Table 6: θ = (0.5, 0.5, 2, 2, −1)"},
]


def test_symbols_match_regardless_of_style():
    for focus, n in (("mu1", 2), ("μ₁", 2), ("µ1", 2), ("MU 1", 2), ("sigma", 1), ("σ₁²", 0),
                     ("lambda", 1), ("table 3", 3), ("θ = (0.5", 4)):
        assert len(sm.filter_focus(SROWS, SF, focus)) == n, focus


def test_column_name_focus_groups_by_that_column():
    assert sm.focus_column(SF, "parameter") == "Parameter"
    assert sm.focus_column(SF, "Parameters") == "Parameter"
    sm.complete = lambda s, u: "By parameter, λ has the largest bias."
    out = sm.summarize(SROWS, SF, excluded=0, focus="parameter")
    assert out["mode"] == "column" and out["rows_used"] == 4
    assert out["facts"]["group_column"] == "Parameter"


def test_no_match_suggests_values():
    sm.complete = lambda s, u: "should not be called"
    out = sm.summarize(SROWS, SF, excluded=0, focus="gemini")
    assert out["text"] == "" and "Parameter" in out["examples"] and "µ1" in out["examples"]["Parameter"]


def test_measures_are_compared_separately_with_the_right_direction():
    rows = SROWS + [{"paper": "simplex", "Sample size": "50", "Measure": "RMSE", "Parameter": "µ1", "Value": "0.027"},
                    {"paper": "simplex", "Sample size": "50", "Measure": "RMSE", "Parameter": "λ", "Value": "0.284"}]
    f = sm.build_facts(rows, SF, group_by="Parameter")
    assert f["measure_column"] == "Measure" and "better_is" not in f
    g = {(x["Measure"], x["Parameter"]): x for x in f["by_group"]}
    assert g[("Bias", "µ1")]["better_is"] == "closer to 0"
    assert g[("Bias", "µ1")]["best"]["Value"] == "0.000"            # |0.000| < |0.001|
    assert all(x["better_is"] in ("closer to 0", "lower") for x in f["by_group"])
    f2 = sm.build_facts(rows, SF)                                       # grouped by Parameter by default too
    assert {x["Measure"] for x in f2["by_group"]} == {"Bias", "RMSE"}


def test_typographic_minus_in_summary_is_not_flagged():
    facts = sm.build_facts(SROWS, SF, group_by="Parameter")
    text = "λ shows the largest bias (−0.150), while µ1 is nearly unbiased (0.001); σ2 1 is −0.046."
    assert sm.unsupported_numbers(text, facts, SROWS) == []
    assert sm.unsupported_numbers("λ has a bias of −0.350.", facts, SROWS) == ["−0.350"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
