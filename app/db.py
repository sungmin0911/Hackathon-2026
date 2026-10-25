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
        conn.execute("ALTER TABLE trex_runs ADD COLUMN IF NOT EXISTS ai_calls INT")
        _setup_activity_log(conn)
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

def get_cached_run(file_hash: str, fields: list[str]) -> dict | None:
    """{"run_id", "ai_calls", "rows"} from a previous run of this exact PDF with these fields, or None."""
    if not enabled():
        return None
    try:
        with _connect() as conn:
            run = conn.execute(
                """SELECT r.id, r.ai_calls FROM trex_runs r JOIN trex_papers p ON p.id = r.paper_id
                   WHERE p.file_hash = %s AND r.fields_key = %s""",
                (file_hash, fields_key(fields)),
            ).fetchone()
            if not run:
                return None
            rows = conn.execute("SELECT data FROM trex_rows WHERE run_id = %s ORDER BY id", (run[0],)).fetchall()
        return {"run_id": run[0], "ai_calls": run[1] or 0, "rows": [r[0] for r in rows]}
    except Exception as e:
        _warn(f"couldn't read the cache ({type(e).__name__}: {e})")
        return None


def get_cached(file_hash: str, fields: list[str]) -> list[dict] | None:
    """Rows from a previous run of this exact PDF with these fields, or None if there isn't one."""
    hit = get_cached_run(file_hash, fields)
    return hit["rows"] if hit else None


def save_run(file_hash: str, file_name: str, title: str, n_pages: int, fields: list[str],
             rows: list[dict], pdf_bytes: bytes | None = None, model: str | None = None,
             ai_calls: int | None = None) -> int | None:
    """Store (or replace) the results of one paper + fields run. Returns the run id, or None."""
    if not enabled():
        return None
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
                """INSERT INTO trex_runs (paper_id, fields, fields_key, model, row_count, ai_calls)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (paper_id, fields_key) DO UPDATE
                     SET fields = EXCLUDED.fields, model = EXCLUDED.model, row_count = EXCLUDED.row_count,
                         ai_calls = EXCLUDED.ai_calls, created_at = now()
                   RETURNING id""",
                (paper_id, fields, fields_key(fields), model, len(rows), ai_calls),
            ).fetchone()[0]
            conn.execute("DELETE FROM trex_rows WHERE run_id = %s", (run_id,))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO trex_rows (run_id, status, page, quote, data) VALUES (%s, %s, %s, %s, %s)",
                    [(run_id, r.get("status"), r.get("found_page"), r.get("quote"),
                      json.dumps({k: v for k, v in r.items() if k not in ("path", "cached")}, default=str))
                     for r in rows],
                )
        return run_id
    except Exception as e:
        _warn(f"couldn't save results ({type(e).__name__}: {e})")
        return None


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


# ---------------- activity log (TimescaleDB on Tiger Cloud) ----------------
#
# Every extraction and every cache hit is one event in trex_events. On Tiger Cloud:
#   - trex_events is a HYPERTABLE (time-partitioned), the natural fit for an append-only event stream
#   - trex_events_daily is a CONTINUOUS AGGREGATE (real-time) that the Activity dashboard reads,
#     so the charts stay instant no matter how many events pile up
#   - a COMPRESSION policy compresses events older than 7 days
# Without TimescaleDB the same table works as a plain table and the dashboard aggregates on the fly.

EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS trex_events (
    time         TIMESTAMPTZ NOT NULL DEFAULT now(),
    kind         TEXT NOT NULL,            -- 'extract' (fresh AI run) | 'cache_hit' (loaded from Tiger) | 'reopen' (History click)
    run_id       INT,
    paper        TEXT,
    file_name    TEXT,
    fields       TEXT,
    rows         INT NOT NULL DEFAULT 0,
    verified     INT NOT NULL DEFAULT 0,
    mismatch     INT NOT NULL DEFAULT 0,
    unverified   INT NOT NULL DEFAULT 0,
    ai_calls     INT NOT NULL DEFAULT 0,   -- AI requests made (fresh runs)
    calls_saved  INT NOT NULL DEFAULT 0,   -- AI requests avoided (cache hits)
    duration_ms  INT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS trex_events_time_idx ON trex_events (time DESC);
CREATE INDEX IF NOT EXISTS trex_events_run_idx ON trex_events (run_id, time DESC);
"""

CAGG_SQL = """
CREATE MATERIALIZED VIEW IF NOT EXISTS trex_events_daily
WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
SELECT time_bucket('1 day', time) AS day,
       count(*) FILTER (WHERE kind = 'extract')   AS fresh_runs,
       count(*) FILTER (WHERE kind = 'cache_hit') AS cache_hits,
       sum(rows) AS rows, sum(verified) AS verified, sum(mismatch) AS mismatch, sum(unverified) AS unverified,
       sum(ai_calls) AS ai_calls, sum(calls_saved) AS calls_saved,
       avg(duration_ms) FILTER (WHERE kind = 'extract') AS avg_ms
FROM trex_events
GROUP BY day
WITH NO DATA
"""

PLAIN_DAILY_SQL = """
SELECT date_trunc('day', time) AS day,
       count(*) FILTER (WHERE kind = 'extract'), count(*) FILTER (WHERE kind = 'cache_hit'),
       sum(rows), sum(verified), sum(mismatch), sum(unverified), sum(ai_calls), sum(calls_saved),
       avg(duration_ms) FILTER (WHERE kind = 'extract')
FROM trex_events WHERE time > now() - %s * interval '1 day'
GROUP BY 1 ORDER BY 1
"""

_timescale: dict[str, bool] = {}   # url -> whether the continuous aggregate is usable


def _try(conn, sql: str, params=None) -> bool:
    try:
        conn.execute(sql, params)
        return True
    except Exception as e:
        _warn(f"activity log: skipped a TimescaleDB step ({type(e).__name__}: {str(e).splitlines()[0][:120]})")
        return False


def _setup_activity_log(conn) -> None:
    conn.execute(EVENTS_SQL)
    url = _url()
    _timescale[url] = False
    if os.getenv("TREX_DISABLE_TIMESCALE"):
        return
    has_ext = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'").fetchone()
    if not has_ext and not _try(conn, "CREATE EXTENSION IF NOT EXISTS timescaledb"):
        return
    if not _try(conn, "SELECT create_hypertable('trex_events', 'time', if_not_exists => TRUE, migrate_data => TRUE)"):
        return
    if _try(conn, CAGG_SQL):
        _try(conn, """SELECT add_continuous_aggregate_policy('trex_events_daily',
                        start_offset => INTERVAL '90 days', end_offset => INTERVAL '1 hour',
                        schedule_interval => INTERVAL '1 hour', if_not_exists => TRUE)""")
        _timescale[url] = True
    compressed = conn.execute("""SELECT compression_enabled FROM timescaledb_information.hypertables
                                 WHERE hypertable_name = 'trex_events'""").fetchone()
    if not (compressed and compressed[0]):
        _try(conn, """ALTER TABLE trex_events SET (timescaledb.compress,
                        timescaledb.compress_segmentby = 'kind', timescaledb.compress_orderby = 'time DESC')""")
    _try(conn, "SELECT add_compression_policy('trex_events', INTERVAL '7 days', if_not_exists => TRUE)")


def log_event(kind: str, rows: list[dict], *, run_id: int | None = None, paper: str | None = None,
              file_name: str | None = None, fields: list[str] | None = None, ai_calls: int = 0,
              calls_saved: int = 0, duration_ms: int = 0) -> None:
    """Record one extraction or cache hit. Never raises."""
    if not enabled():
        return
    counts = {s: sum(1 for r in rows if r.get("status") == s) for s in ("verified", "mismatch", "unverified")}
    try:
        with _connect() as conn:
            conn.execute(
                """INSERT INTO trex_events (kind, run_id, paper, file_name, fields, rows, verified, mismatch,
                                            unverified, ai_calls, calls_saved, duration_ms)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (kind, run_id, paper, file_name, ", ".join(fields or []), len(rows), counts["verified"],
                 counts["mismatch"], counts["unverified"], ai_calls, calls_saved, duration_ms))
    except Exception as e:
        _warn(f"couldn't log activity ({type(e).__name__}: {e})")


def recent_activity(limit: int = 10) -> list[dict]:
    """Most recently used runs (fresh or reloaded), newest first, like a recent-chats list."""
    if not enabled():
        return []
    try:
        with _connect() as conn:
            rows = conn.execute(
                """SELECT e.run_id, e.time, e.kind, coalesce(p.title, e.paper), p.file_name, r.fields,
                          r.row_count, (SELECT count(*) FROM trex_rows x WHERE x.run_id = e.run_id
                                        AND x.status = 'verified')
                   FROM (SELECT DISTINCT ON (run_id) run_id, time, kind, paper
                         FROM trex_events WHERE run_id IS NOT NULL ORDER BY run_id, time DESC) e
                   JOIN trex_runs r ON r.id = e.run_id JOIN trex_papers p ON p.id = r.paper_id
                   ORDER BY e.time DESC LIMIT %s""",
                (limit,),
            ).fetchall()
        return [{"run_id": r[0], "time": r[1], "kind": r[2], "title": r[3], "file_name": r[4],
                 "fields": list(r[5]), "rows": r[6], "verified": r[7]} for r in rows]
    except Exception as e:
        _warn(f"couldn't read recent activity ({type(e).__name__}: {e})")
        return []


def daily_activity(days: int = 30) -> list[dict]:
    """Per-day totals for the dashboard: from the continuous aggregate on Tiger, computed otherwise."""
    if not enabled():
        return []
    cols = ["day", "fresh_runs", "cache_hits", "rows", "verified", "mismatch", "unverified",
            "ai_calls", "calls_saved", "avg_ms"]
    try:
        with _connect() as conn:
            if _timescale.get(_url()):
                data = conn.execute(f"SELECT {', '.join(cols)} FROM trex_events_daily "
                                    "WHERE day > now() - %s * interval '1 day' ORDER BY day", (days,)).fetchall()
            else:
                data = conn.execute(PLAIN_DAILY_SQL, (days,)).fetchall()
        return [{c: (v if c == "day" or v is None else float(v)) for c, v in zip(cols, row)} for row in data]
    except Exception as e:
        _warn(f"couldn't read daily activity ({type(e).__name__}: {e})")
        return []


def timescale_features() -> dict:
    """Which TimescaleDB features are active, for the dashboard badges."""
    out = {"hypertable": False, "continuous_aggregate": False, "compression": False,
           "compressed_chunks": 0, "version": None}
    if not enabled():
        return out
    try:
        with _connect() as conn:
            v = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'").fetchone()
            if not v:
                return out
            out["version"] = v[0]
            h = conn.execute("""SELECT compression_enabled FROM timescaledb_information.hypertables
                                WHERE hypertable_name = 'trex_events'""").fetchone()
            out["hypertable"] = bool(h)
            out["compression"] = bool(h and h[0])
            out["continuous_aggregate"] = bool(conn.execute(
                """SELECT 1 FROM timescaledb_information.continuous_aggregates
                   WHERE view_name = 'trex_events_daily'""").fetchone())
            out["compressed_chunks"] = conn.execute(
                """SELECT count(*) FROM timescaledb_information.chunks
                   WHERE hypertable_name = 'trex_events' AND is_compressed""").fetchone()[0]
    except Exception as e:
        _warn(f"couldn't read TimescaleDB status ({type(e).__name__}: {e})")
    return out


def history(limit: int = 300) -> list[dict]:
    """Every saved run, most recently used first (fresh run, reload or reopen), for the History list."""
    if not enabled():
        return []
    try:
        with _connect() as conn:
            rows = conn.execute(
                """SELECT r.id, coalesce(p.title, p.file_name), p.file_name, r.fields, r.row_count,
                          (SELECT count(*) FROM trex_rows x WHERE x.run_id = r.id AND x.status = 'verified'),
                          greatest(r.created_at, coalesce(max(e.time), r.created_at)) AS last_used
                   FROM trex_runs r
                   JOIN trex_papers p ON p.id = r.paper_id
                   LEFT JOIN trex_events e ON e.run_id = r.id
                   GROUP BY r.id, p.id
                   ORDER BY last_used DESC
                   LIMIT %s""",
                (limit,),
            ).fetchall()
        return [{"run_id": r[0], "title": r[1], "file_name": r[2], "fields": list(r[3]), "rows": r[4],
                 "verified": r[5], "last_used": r[6]} for r in rows]
    except Exception as e:
        _warn(f"couldn't read history ({type(e).__name__}: {e})")
        return []
