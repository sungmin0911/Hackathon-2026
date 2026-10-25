"""Tiger Cloud (PostgreSQL) storage: a cache of past extractions and a browsable library.

Everything here is optional. With no DATABASE_URL set, `enabled()` is False and TREX works exactly
as before. Database errors never stop an extraction; they are printed as warnings instead.

Each function opens a short-lived connection, which is simple and safe with Streamlit's threads.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

SCHEMA = Path(__file__).resolve().parent.parent / "schema.sql"
_schema_ready: set[str] = set()


def _url() -> str:
    return (os.getenv("DATABASE_URL") or "").strip()


def enabled() -> bool:
    return bool(_url())


def _warn(msg: str) -> None:
    print(f"  [warning] database: {msg}", file=sys.stderr, flush=True)


def _connect():
    import psycopg

    url = _url()
    conn = psycopg.connect(url, autocommit=True, connect_timeout=10)
    if url not in _schema_ready:
        conn.execute(SCHEMA.read_text())
        _schema_ready.add(url)
    return conn


# ---------------- helpers ----------------

def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fields_key(fields: list[str]) -> str:
    """Normalized key so 'Model, Benchmark, Score' and 'model,benchmark,score' match."""
    return "|".join(" ".join(f.lower().split()) for f in fields)


def check() -> dict:
    """Connection test for the UI/CLI: {"ok": bool, "papers": int, "rows": int, "error": str|None}."""
    if not enabled():
        return {"ok": False, "papers": 0, "rows": 0, "error": "DATABASE_URL is not set"}
    try:
        with _connect() as conn:
            papers = conn.execute("SELECT count(*) FROM trex_papers").fetchone()[0]
            rows = conn.execute("SELECT count(*) FROM trex_rows").fetchone()[0]
        return {"ok": True, "papers": papers, "rows": rows, "error": None}
    except Exception as e:
        return {"ok": False, "papers": 0, "rows": 0, "error": f"{type(e).__name__}: {e}"}


# ---------------- cache ----------------

def get_cached(file_hash: str, fields: list[str]) -> list[dict] | None:
    """Rows from a previous run of this exact PDF with these fields, or None if there isn't one."""
    if not enabled():
        return None
    try:
        with _connect() as conn:
            run = conn.execute(
                """SELECT r.id FROM trex_runs r JOIN trex_papers p ON p.id = r.paper_id
                   WHERE p.file_hash = %s AND r.fields_key = %s""",
                (file_hash, fields_key(fields)),
            ).fetchone()
            if not run:
                return None
            rows = conn.execute("SELECT data FROM trex_rows WHERE run_id = %s ORDER BY id", (run[0],)).fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        _warn(f"couldn't read the cache ({type(e).__name__}: {e})")
        return None


def save_run(file_hash: str, file_name: str, title: str, n_pages: int, fields: list[str],
             rows: list[dict], pdf_bytes: bytes | None = None, model: str | None = None) -> bool:
    """Store (or replace) the results of one paper + fields run. Returns True on success."""
    if not enabled():
        return False
    try:
        with _connect() as conn, conn.transaction():
            paper_id = conn.execute(
                """INSERT INTO trex_papers (file_hash, file_name, title, pages, pdf)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (file_hash) DO UPDATE
                     SET file_name = EXCLUDED.file_name, title = EXCLUDED.title,
                         pages = EXCLUDED.pages, pdf = COALESCE(EXCLUDED.pdf, trex_papers.pdf)
                   RETURNING id""",
                (file_hash, file_name, title, n_pages, pdf_bytes),
            ).fetchone()[0]
            run_id = conn.execute(
                """INSERT INTO trex_runs (paper_id, fields, fields_key, model, row_count)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (paper_id, fields_key) DO UPDATE
                     SET fields = EXCLUDED.fields, model = EXCLUDED.model,
                         row_count = EXCLUDED.row_count, created_at = now()
                   RETURNING id""",
                (paper_id, fields, fields_key(fields), model, len(rows)),
            ).fetchone()[0]
            conn.execute("DELETE FROM trex_rows WHERE run_id = %s", (run_id,))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO trex_rows (run_id, status, page, quote, data) VALUES (%s, %s, %s, %s, %s)",
                    [(run_id, r.get("status"), r.get("found_page"), r.get("quote"),
                      json.dumps({k: v for k, v in r.items() if k not in ("path", "cached")}, default=str))
                     for r in rows],
                )
        return True
    except Exception as e:
        _warn(f"couldn't save results ({type(e).__name__}: {e})")
        return False


# ---------------- library ----------------

def list_runs(limit: int = 200) -> list[dict]:
    """Saved runs, newest first, for the library picker."""
    if not enabled():
        return []
    try:
        with _connect() as conn:
            rows = conn.execute(
                """SELECT r.id, p.title, p.file_name, r.fields, r.row_count, r.created_at,
                          (p.pdf IS NOT NULL) AS has_pdf,
                          (SELECT count(*) FROM trex_rows x WHERE x.run_id = r.id AND x.status = 'verified')
                   FROM trex_runs r JOIN trex_papers p ON p.id = r.paper_id
                   ORDER BY r.created_at DESC LIMIT %s""",
                (limit,),
            ).fetchall()
        return [{"run_id": r[0], "title": r[1], "file_name": r[2], "fields": list(r[3]), "rows": r[4],
                 "created_at": r[5], "has_pdf": r[6], "verified": r[7]} for r in rows]
    except Exception as e:
        _warn(f"couldn't list saved runs ({type(e).__name__}: {e})")
        return []


def load_runs(run_ids: list[int], pdf_dir: str | Path | None = None) -> list[dict]:
    """All rows of the given runs. If pdf_dir is given, stored PDFs are written there and each
    row gets a "path" so the UI can render highlighted source pages."""
    if not enabled() or not run_ids:
        return []
    try:
        with _connect() as conn:
            rows = conn.execute(
                """SELECT x.data, p.file_name, p.file_hash
                   FROM trex_rows x JOIN trex_runs r ON r.id = x.run_id JOIN trex_papers p ON p.id = r.paper_id
                   WHERE r.id = ANY(%s) ORDER BY r.id, x.id""",
                (list(run_ids),),
            ).fetchall()
            paths = {}
            if pdf_dir is not None:
                folder = Path(pdf_dir)
                folder.mkdir(parents=True, exist_ok=True)
                for name, fhash, pdf in conn.execute(
                        """SELECT DISTINCT p.file_name, p.file_hash, p.pdf FROM trex_papers p
                           JOIN trex_runs r ON r.paper_id = p.id WHERE r.id = ANY(%s) AND p.pdf IS NOT NULL""",
                        (list(run_ids),)).fetchall():
                    path = folder / f"{fhash[:8]}_{name}"
                    path.write_bytes(bytes(pdf))
                    paths[fhash] = str(path)
        out = []
        for data, name, fhash in rows:
            row = dict(data)
            if fhash in paths:
                row["path"] = paths[fhash]
            out.append(row)
        return out
    except Exception as e:
        _warn(f"couldn't load saved runs ({type(e).__name__}: {e})")
        return []


def runs_since(hours: float = 24) -> int | None:
    """How many fresh extractions (not cache hits) were saved in the last N hours. None if unknown."""
    if not enabled():
        return None
    try:
        with _connect() as conn:
            return conn.execute("SELECT count(*) FROM trex_runs WHERE created_at > now() - %s * interval '1 hour'",
                                (hours,)).fetchone()[0]
    except Exception as e:
        _warn(f"couldn't count recent runs ({type(e).__name__}: {e})")
        return None
