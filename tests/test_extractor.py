"""Tests for response parsing and compact-table expansion (no LLM needed)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.extractor import expand_table, make_chunks, parse_response
from app.pdf_reader import Page

FIELDS = ["Model", "Benchmark", "Score"]
TABLE = {
    "page": 16, "column_field": "Model", "row_field": "Benchmark", "value_field": "Score",
    "columns": ["Kimi-K2-Instruct", "Claude\nOpus 4", "GPT-4.1"],
    "rows": [{"label": "MMLU (EM)", "values": ["89.5", "92.9", "90.4"], "quote": "MMLU (EM) 89.5 92.9 90.4"},
             {"label": "SimpleQA", "values": ["31.0", "-", "42.3"], "quote": "SimpleQA 31.0 - 42.3"}],
}


def test_expand_one_row_per_cell():
    rows = expand_table(TABLE, FIELDS)
    assert len(rows) == 5  # 3 + 2 (the "-" cell is skipped)
    opus = [r for r in rows if r["Model"] == "Claude Opus 4"][0]  # header line break joined
    assert opus == {"Model": "Claude Opus 4", "Benchmark": "MMLU (EM)", "Score": "92.9",
                    "quote": "MMLU (EM) 89.5 92.9 90.4", "page": 16}


def test_misaligned_row_is_flagged():
    t = dict(TABLE, rows=[{"label": "MMLU", "values": ["89.5", "92.9"], "quote": "MMLU 89.5 92.9"}])
    assert all(r.get("alignment") == "uncertain" for r in expand_table(t, FIELDS))


def test_parse_new_format_with_tables_and_results():
    raw = "```json\n" + json.dumps({"tables": [TABLE], "results": [
        {"Model": "X", "Benchmark": "AIME [3]", "Score": "79.8", "quote": "X achieves 79.8 on AIME [3]", "page": 1}]}) + "\n```"
    rows = parse_response(raw, FIELDS)
    assert len(rows) == 6 and rows[-1]["Score"] == "79.8"


def test_parse_old_flat_format_still_works():
    raw = json.dumps([{"Model": "A", "Benchmark": "B", "Score": "1.0", "quote": "q", "page": 1}])
    assert len(parse_response(raw, FIELDS)) == 1


def test_unknown_field_names_fall_back_sensibly():
    t = dict(TABLE, column_field="model name", row_field="task", value_field="accuracy")
    r = expand_table(t, FIELDS)[0]
    assert r["Model"] == "Kimi-K2-Instruct" and r["Benchmark"] == "MMLU (EM)" and r["Score"] == "89.5"


def test_dense_pages_get_their_own_chunk():
    sparse = lambda n: Page(n, "text 1.1 2.2 3.3")
    dense = Page(5, " ".join(f"{i}.5" for i in range(60)))
    chunks = make_chunks([sparse(1), sparse(2), dense, sparse(6)], size=6)
    assert [[p.number for p in c] for c in chunks] == [[1, 2], [5], [6]]


def test_pipeline_keeps_verified_copy_of_duplicates():
    from app.pipeline import keep_best_copies
    rows = [{"Model": "Kimi", "Benchmark": "GPQA", "Score": "75.1", "status": "unverified", "page": 1},
            {"Model": "Kimi", "Benchmark": "GPQA", "Score": "75.1", "status": "verified", "page": 16},
            {"Model": "GPT", "Benchmark": "GPQA", "Score": "66.3", "status": "verified", "page": 16}]
    out = keep_best_copies(rows, FIELDS)
    assert len(out) == 2 and out[0]["page"] == 16 and out[0]["status"] == "verified"


def test_parser_accepts_common_shape_variations():
    """Models often return a slightly different shape than asked; none of these may parse to 0 rows."""
    upper = {"Tables": [TABLE], "Results": []}                      # capitalized keys
    single = TABLE                                                  # one table, no wrapper
    bare_list = [TABLE]                                             # list of tables
    wrapped = {"data": {"tables": [TABLE]}}                         # one-key wrapper
    tables_obj = {"tables": TABLE}                                  # "tables" as an object
    for shape in (upper, single, bare_list, wrapped, tables_obj):
        assert len(parse_response(json.dumps(shape), FIELDS)) == 5, shape


def test_real_problems_warn_but_no_match_is_silent():
    import io, contextlib, tempfile
    import app.extractor as ex
    ex.RAW_REPLY_DIR = tempfile.mkdtemp()
    dense = [Page(13, " ".join(f"{i}.5" for i in range(60)))]
    for reply, expected in (("", "EMPTY reply"),
                            ('{"tables": [{"rows": [', "couldn't read"),
                            ('{"tables": [], "results": []}', None)):   # fields not in paper: no warning
        ex.complete = lambda s, u, r=reply: r
        log = io.StringIO()
        with contextlib.redirect_stderr(log):
            assert ex.extract(FIELDS, dense, label="t") == []
        if expected:
            assert expected in log.getvalue() and "raw reply:" in log.getvalue(), log.getvalue()
        else:
            assert log.getvalue() == "", log.getvalue()


def test_text_before_json_with_empty_example_is_ignored():
    """Reasoning text that echoes the empty example must not hide the real answer after it."""
    raw = ('Thinking... the instructions say if nothing is relevant return {"tables": [], "results": []}. '
           'But page 16 has a table, so here is the answer:\n' + json.dumps({"tables": [TABLE], "results": []}))
    assert len(parse_response(raw, FIELDS)) == 5


def test_suggestions_are_categories_not_values():
    import json as _j
    import app.extractor as ex
    pages = [Page(1, "Abstract"), Page(13, "MATH-500 (Pass@1) 97.3 96.4 90.2 GPQA 71.5 75.7")]
    reply = _j.dumps({"suggestions": [
        {"fields": ["Model", "Benchmark", "Score"], "why": "6 models on 20 benchmarks",
         "example": {"Model": "DeepSeek-R1", "Benchmark": "MATH-500", "Score": "97.3"}},
        {"fields": ["MATH-500", "Pass@1", "GPQA"]},                       # values posing as columns
        {"fields": ["DeepSeek-R1", "Benchmark", "Accuracy"],               # a model name as a column
         "example": {"DeepSeek-R1": "x", "Benchmark": "GPQA", "Accuracy": "71.5"}},
        {"fields": ["Model", "Benchmark", "Score"]}]})                      # duplicate
    ex.complete = lambda s, u: "Thinking first...\n" + reply
    out = ex.suggest_fields_from_pages(pages)
    assert [s["fields"] for s in out] == [["Model", "Benchmark", "Score"], ["Benchmark", "Accuracy"]], out
    assert out[0]["why"] == "6 models on 20 benchmarks"
    assert out[0]["example"] == {"Model": "DeepSeek-R1", "Benchmark": "MATH-500", "Score": "97.3"}


def test_suggestions_accept_older_shapes():
    import app.extractor as ex
    pages = [Page(1, "a 1.1 2.2 3.3")]
    ex.complete = lambda s, u: '[["Model", "Dataset", "Accuracy"], ["Model", "Parameters"]]'
    assert [s["fields"] for s in ex.suggest_fields_from_pages(pages)] == [["Model", "Dataset", "Accuracy"],
                                                                         ["Model", "Parameters"]]
    ex.complete = lambda s, u: "no idea"
    assert ex.suggest_fields_from_pages(pages) == []


def test_metric_names_with_digits_are_allowed():
    from app.extractor import _looks_like_value
    assert not _looks_like_value("F1 score") and not _looks_like_value("Top-1 Accuracy")
    assert _looks_like_value("GPT-4") and _looks_like_value("MATH-500") and _looks_like_value("Pass@1")

def test_diagnose_fields_explains_each_field():
    import json as _j
    import app.extractor as ex
    pages = [Page(2, "We discuss training time briefly; training was efficient."),
             Page(13, "Table 4 MMLU 88.3 87.2 90.8 GPQA 65.0 49.9 71.5 Claude GPT DeepSeek-R1")]
    reply = _j.dumps({"fields": [
        {"field": "Model", "status": "reported", "example": "DeepSeek-R1", "page": 13},
        {"field": "Benchmark", "status": "reported", "example": "MMLU", "page": 13},
        {"field": "Training time", "status": "not_found",                 # AI says not found...
         "explanation": "The paper never states training time for any model.", "closest": "Parameters"}]})
    ex.complete = lambda s, u: "Let me check.\n" + reply
    d = {x["field"]: x for x in ex.diagnose_fields(pages, ["Model", "Benchmark", "Training time"])}
    assert d["Model"]["status"] == "reported" and d["Model"]["example"] == "DeepSeek-R1"
    # ...but the words ARE on page 2, so the plain search overrides "not_found" with "mentioned_only"
    assert d["Training time"]["status"] == "mentioned_only" and d["Training time"]["mentioned_on"] == [2]
    assert d["Training time"]["closest"] == "Parameters"
    assert "never states" in d["Training time"]["explanation"]


def test_diagnose_falls_back_to_plain_search_if_ai_fails():
    import app.extractor as ex
    pages = [Page(1, "Our method trains fast. Training time is 3 days.")]
    ex.complete = lambda s, u: "sorry"
    d = {x["field"]: x for x in ex.diagnose_fields(pages, ["Model", "Training time"])}
    assert d["Training time"]["status"] == "mentioned_only" and d["Model"]["status"] == "not_found"


def test_progress_reports_every_part_and_keeps_page_order():
    import json as _j, re as _re, time as _t
    import app.extractor as ex
    pages = [Page(n, " ".join(f"{i}.5" for i in range(60)) + f" PAGE{n}") for n in (1, 2, 3, 4)]  # 4 dense pages
    def fake(system, user):
        n = int(_re.search(r"PAGE(\d)", user).group(1))
        _t.sleep(0.05 * (5 - n))                  # later pages finish FIRST
        return _j.dumps({"results": [{"Model": f"M{n}", "Benchmark": "B", "Score": f"{n}.0",
                                      "quote": f"PAGE{n}", "page": n}]})
    ex.complete = fake
    seen = []
    rows = ex.extract(FIELDS, pages, on_progress=lambda d, n, pg: seen.append((d, n, pg)))
    assert [d for d, n, _ in seen] == [1, 2, 3, 4] and all(n == 4 for _, n, _ in seen)
    assert [r["Model"] for r in rows] == ["M1", "M2", "M3", "M4"], "results stay in page order"


def test_timeout_is_retried_once_then_raised():
    import app.llm as llm
    calls = {"n": 0}
    class APITimeoutError(Exception):
        pass
    def slow():
        calls["n"] += 1
        raise APITimeoutError("Request timed out.")
    try:
        llm._with_retries(slow)
        assert False, "should raise after the retry"
    except APITimeoutError:
        pass
    assert calls["n"] == 2


def test_max_calls_keeps_most_table_heavy_parts_in_page_order():
    import io, contextlib, json as _j, re as _re
    import app.extractor as ex
    dense = lambda n, k: Page(n, " ".join(f"{i}.5" for i in range(k)) + f" PAGE{n}")
    pages = [dense(1, 50), dense(2, 200), dense(3, 45), dense(4, 300)]      # each its own part
    ex.complete = lambda s, u: _j.dumps({"results": [{"Model": "M" + _re.search(r"PAGE(\d)", u).group(1),
                                                      "Benchmark": "B", "Score": "1.0", "quote": "q", "page": 1}]})
    log = io.StringIO()
    with contextlib.redirect_stderr(log):
        rows = ex.extract(FIELDS, pages, max_calls=2)
    assert [r["Model"] for r in rows] == ["M2", "M4"], rows      # the two densest, in page order
    assert "skipped 2" in log.getvalue()


def test_table_setting_is_attached_and_keeps_rows_apart():
    from app.pipeline import keep_best_copies
    t3 = dict(TABLE, setting="Table 3: TP-1A, constant diffusion")
    t4 = dict(TABLE, setting="Table 4: TP-2B, high frequency")
    rows = expand_table(t3, FIELDS) + expand_table(t4, FIELDS)
    assert rows[0]["setting"] == "Table 3: TP-1A, constant diffusion"
    for r in rows:
        r["status"] = "verified"
    assert len(keep_best_copies(rows, FIELDS)) == 10, "identical values in different tables are both kept"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
