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


if __name__ == "__main__":
    if not URL:
        print("skipped: set TEST_DATABASE_URL to run database tests")
        sys.exit(0)
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
