"""PDF -> extracted rows -> verified rows. Shared by the CLI and the web UI.

If a Tiger Cloud database is configured (DATABASE_URL), results are cached: the same PDF with the
same fields is loaded from the database instead of calling the AI again, and every new run is saved.
"""
import os
import re
from pathlib import Path

from . import db
from .consistency import annotate_rows_with_consistency
from .extractor import extract
from .pdf_reader import paper_title, read_pdf
from .verify import verify_row

_RANK = {"verified": 0, "mismatch": 1, "unverified": 2}
MAX_PDF_BYTES_TO_STORE = 20 * 1024 * 1024  # keep very large PDFs out of the database


def keep_best_copies(rows: list[dict], fields: list[str]) -> list[dict]:
    """If the same result (same field values) was extracted more than once, e.g. from a chart
    and from a table, keep only the copy that verifies best. Order is otherwise preserved."""
    best: dict[tuple, int] = {}
    for i, r in enumerate(rows):
        key = tuple(re.sub(r"\s+", " ", str(r.get(f) or "")).strip().lower() for f in fields)
        if key not in best or _RANK.get(r.get("status"), 3) < _RANK.get(rows[best[key]].get("status"), 3):
            best[key] = i
    keep = set(best.values())
    return [r for i, r in enumerate(rows) if i in keep]


def process_paper(path: str | Path, fields: list[str], check_consistency: bool = True,
                  use_cache: bool = True, on_progress=None, max_calls: int | None = None) -> list[dict]:
    path = Path(path)
    file_hash = db.file_sha256(path) if db.enabled() else None

    if use_cache and file_hash:
        cached = db.get_cached(file_hash, fields)
        if cached is not None:
            return [{**r, "file": path.name, "path": str(path), "cached": True} for r in cached]

    pages = read_pdf(path)
    title = paper_title(path)
    rows = extract(fields, pages, label=path.stem, on_progress=on_progress, max_calls=max_calls)
    verified = [{"paper": title, "file": path.name, **verify_row(r, fields, pages)} for r in rows]
    verified = keep_best_copies(verified, fields)
    if check_consistency and rows:
        verified = annotate_rows_with_consistency(verified, fields)

    if file_hash:
        pdf_bytes = path.read_bytes() if path.stat().st_size <= MAX_PDF_BYTES_TO_STORE else None
        db.save_run(file_hash, path.name, title, len(pages), fields, verified,
                    pdf_bytes=pdf_bytes, model=os.getenv("LLM_MODEL"))
    return verified
