"""Tiger Cloud / Postgres tests. They need a database and are skipped otherwise:

    set TEST_DATABASE_URL=postgresql://user:pass@host:port/dbname   (a test DB: tables get created)
    python tests/test_db.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

URL = os.getenv("TEST_DATABASE_URL")


def _setup():
    os.environ["DATABASE_URL"] = URL
    import psycopg
    with psycopg.connect(URL, autocommit=True) as c:
        c.execute("DROP MATERIALIZED VIEW IF EXISTS trex_events_daily CASCADE")
        c.execute("DROP TABLE IF EXISTS trex_events CASCADE")      # a hypertable must be dropped on its own
        c.execute("DROP TABLE IF EXISTS trex_rows, trex_runs, trex_papers CASCADE")
    from app import db
    db._schema_ready.clear()


def _make_pdf(path: Path, text_lines: list[str]):
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    for i, line in enumerate(text_lines):
        page.insert_text((72, 72 + 18 * i), line)
    doc.save(path)


def test_cache_save_load_and_library():
    _setup()
    import app.extractor as ex
    from app import db
    from app.pipeline import process_paper

    tmp = Path(tempfile.mkdtemp())
    pdf = tmp / "resnet.pdf"
    _make_pdf(pdf, ["Table 2: ResNet-110 has 6.43% error on CIFAR-10.", "ResNet-56 has 6.97% error on CIFAR-10."])
    calls = {"n": 0}

    def fake(system, user):
        calls["n"] += 1
        return json.dumps({"tables": [], "results": [
            {"Model": "ResNet-110", "Dataset": "CIFAR-10", "Accuracy": "6.43% error",
             "quote": "ResNet-110 has 6.43% error on CIFAR-10", "page": 1}]})
    ex.complete = fake
    fields = ["Model", "Dataset", "Accuracy"]

    first = process_paper(pdf, fields)
    assert calls["n"] == 1 and len(first) == 1 and first[0]["status"] == "verified"
    assert not first[0].get("cached")

    second = process_paper(pdf, [" model ", "DATASET", "accuracy"])   # same fields, different spelling
    assert calls["n"] == 1, "second run must come from the cache, with no AI call"
    assert second[0]["cached"] and second[0]["Accuracy"] == "6.43% error" and second[0]["status"] == "verified"

    renamed = tmp / "renamed.pdf"
    renamed.write_bytes(pdf.read_bytes())
    assert process_paper(renamed, fields)[0]["cached"], "same content under a new name is still cached"

    process_paper(pdf, fields, use_cache=False)                       # forced re-run replaces, not duplicates
    assert calls["n"] == 2
    runs = db.list_runs()
    assert len(runs) == 1 and runs[0]["rows"] == 1 and runs[0]["verified"] == 1 and runs[0]["has_pdf"]

    out_dir = tmp / "restored"
    loaded = db.load_runs([runs[0]["run_id"]], pdf_dir=out_dir)
    assert len(loaded) == 1 and Path(loaded[0]["path"]).read_bytes() == pdf.read_bytes()

    c = db.check()
    assert c["ok"] and c["papers"] == 1 and c["rows"] == 1


def test_bad_url_never_breaks_extraction():
    os.environ["DATABASE_URL"] = "postgresql://nobody:wrong@127.0.0.1:1/none"
    import io, contextlib
    import app.extractor as ex
    from app import db
    from app.pipeline import process_paper
    db._schema_ready.clear()
    tmp = Path(tempfile.mkdtemp())
    pdf = tmp / "p.pdf"
    _make_pdf(pdf, ["ResNet-110 has 6.43% error on CIFAR-10."])
    ex.complete = lambda s, u: json.dumps({"results": [
        {"Model": "ResNet-110", "Accuracy": "6.43% error", "quote": "ResNet-110 has 6.43% error", "page": 1}]})
    log = io.StringIO()
    with contextlib.redirect_stderr(log):
        rows = process_paper(pdf, ["Model", "Accuracy"])
    assert len(rows) == 1 and "database" in log.getvalue()
    assert not db.check()["ok"]


def _fake_extract_and_two_runs():
    import app.extractor as ex
    from app.pipeline import process_paper
    tmp = Path(tempfile.mkdtemp())
    pdf = tmp / "resnet.pdf"
    _make_pdf(pdf, ["Table 2: ResNet-110 has 6.43% error on CIFAR-10."])
    ex.complete = lambda s, u: json.dumps({"results": [
        {"Model": "ResNet-110", "Dataset": "CIFAR-10", "Accuracy": "6.43% error",
         "quote": "ResNet-110 has 6.43% error on CIFAR-10", "page": 1},
        {"Model": "ViT", "Dataset": "CIFAR-10", "Accuracy": "99.5%", "quote": "invented quote", "page": 1}]})
    fields = ["Model", "Dataset", "Accuracy"]
    process_paper(pdf, fields)          # fresh: 1 AI call
    process_paper(pdf, fields)          # cache hit: saves 1 call
    process_paper(pdf, fields)          # cache hit again


def test_activity_log_with_timescale():
    _setup()
    os.environ.pop("TREX_DISABLE_TIMESCALE", None)
    from app import db
    _fake_extract_and_two_runs()
    feats = db.timescale_features()
    if not feats["version"]:
        print("  (TimescaleDB not installed here: checked the fallback only)")
    else:
        assert feats["hypertable"] and feats["continuous_aggregate"] and feats["compression"], feats
    recent = db.recent_activity()
    assert len(recent) == 1 and recent[0]["kind"] == "cache_hit" and recent[0]["verified"] == 1
    day = db.daily_activity()[-1]
    assert (day["fresh_runs"], day["cache_hits"], day["ai_calls"], day["calls_saved"]) == (1, 2, 1, 2), day
    assert (day["rows"], day["verified"], day["unverified"]) == (6, 3, 3), day


def test_activity_log_fallback_without_timescale():
    _setup()
    os.environ["TREX_DISABLE_TIMESCALE"] = "1"
    try:
        from app import db
        _fake_extract_and_two_runs()
        day = db.daily_activity()[-1]
        assert (day["fresh_runs"], day["cache_hits"], day["calls_saved"]) == (1, 2, 2), day
        assert not db.timescale_features()["hypertable"]
    finally:
        os.environ.pop("TREX_DISABLE_TIMESCALE", None)


def test_history_moves_reopened_run_to_top():
    _setup()
    os.environ.pop("TREX_DISABLE_TIMESCALE", None)
    from app import db
    _fake_extract_and_two_runs()
    first = db.history()[0]["run_id"]
    tmp = Path(tempfile.mkdtemp()); pdf = tmp / "other.pdf"
    _make_pdf(pdf, ["Another paper."])
    other = db.save_run(db.file_sha256(pdf), "other.pdf", "Other", 1, ["Model"], [])
    assert [h["run_id"] for h in db.history()] == [other, first]
    db.log_event("reopen", [], run_id=first)
    assert [h["run_id"] for h in db.history()] == [first, other]
    day = db.daily_activity()[-1]  # reopens don't inflate the totals
    assert (day["fresh_runs"], day["cache_hits"], day["rows"]) == (1, 2, 6), day


if __name__ == "__main__":
    if not URL:
        print("skipped: set TEST_DATABASE_URL to run database tests")
        sys.exit(0)
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
