"""Number reading for statistics-style values (no LLM needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.numbers import first_number, looks_numeric, signed_numbers
from app.pdf_reader import Page
from app.verify import verify_row

PAGE = Page(5, """Table 3: OLS regression of log wage
Variable        Coefficient   Std. error   p-value
Education       0.083***      (0.012)      <0.001
Experience      −0.021*       (0.010)      0.034
Female          −.154**       (0.051)      .003
N = 2,431   R² = 0.27   Ages 25–64""")
F = ["Variable", "Coefficient", "Std. error", "p-value"]


def status(**row):
    return verify_row(dict(row, page=5), F, [PAGE])["status"]


def test_signs_and_formats_are_read_correctly():
    assert signed_numbers("Experience −0.021* (0.010) 0.034") == ["-0.021", "0.01", "0.034"]
    assert signed_numbers("−.154 .003 2,431 6.430%") == ["-0.154", "0.003", "2431", "6.43"]
    assert signed_numbers("DeepSeek-R1 MMLU-Pro ages 25–64 1.9e−2") == ["1", "25", "64", "1.9", "2"]
    assert first_number("−0.021***") == -0.021 and first_number("(0.012)") == 0.012


def test_numeric_values_vs_names():
    for v in ("79.8", "−0.021", "(0.012)", ".003", "<0.001", "+3.2", "6.43% error"):
        assert looks_numeric(v), v
    for v in ("DeepSeek-R1", "AIME 2024", "GPT-4", "Experience"):
        assert not looks_numeric(v), v


def test_stats_rows_verify_and_sign_flips_are_caught():
    q = "Experience −0.021* (0.010) 0.034"
    assert status(Variable="Experience", Coefficient="−0.021", **{"Std. error": "(0.010)", "p-value": "0.034"}, quote=q) == "verified"
    assert status(Variable="Experience", Coefficient="-0.021", **{"Std. error": "0.010", "p-value": "0.034"}, quote=q) == "verified"
    assert status(Variable="Experience", Coefficient="0.021", **{"Std. error": "0.010", "p-value": "0.034"}, quote=q) == "mismatch"
    assert status(Variable="Experience", Coefficient="-0.021", **{"Std. error": "0.010", "p-value": "0.043"}, quote=q) == "mismatch"
    assert status(Variable="Female", Coefficient="−.154", **{"Std. error": "(0.051)", "p-value": ".003"},
                  quote="Female −.154** (0.051) .003") == "verified"


def test_negative_scores_rank_correctly():
    from app.summary import build_facts
    rows = [{"Variable": "A", "Model": "m", "Coefficient": "−0.50"}, {"Variable": "B", "Model": "m", "Coefficient": "0.20"}]
    f = build_facts(rows, ["Variable", "Model", "Coefficient"])
    assert f["by_group"][0]["best"]["Variable"] == "B" and f["by_group"][0]["range"] == "-0.5 to 0.2"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
