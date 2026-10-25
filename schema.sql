-- TREX schema for Tiger Cloud (PostgreSQL). Safe to run repeatedly.

-- One row per unique PDF (identified by its content hash, so renaming a file doesn't matter).
-- The PDF itself is stored too, so the hosted website can show highlighted source pages
-- for saved results without needing the file on disk.
CREATE TABLE IF NOT EXISTS trex_papers (
    id          SERIAL PRIMARY KEY,
    file_hash   TEXT UNIQUE NOT NULL,      -- sha256 of the PDF bytes
    file_name   TEXT NOT NULL,
    title       TEXT,
    pages       INT,
    pdf         BYTEA,
    added_at    TIMESTAMPTZ DEFAULT now()
);

-- One extraction run = one paper + one set of fields. Re-running the same pair replaces it.
CREATE TABLE IF NOT EXISTS trex_runs (
    id          SERIAL PRIMARY KEY,
    paper_id    INT NOT NULL REFERENCES trex_papers(id) ON DELETE CASCADE,
    fields      TEXT[] NOT NULL,           -- as the user typed them, e.g. {Model,Benchmark,Score}
    fields_key  TEXT NOT NULL,             -- normalized, e.g. "model|benchmark|score"
    model       TEXT,                      -- which LLM produced it
    row_count   INT NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE (paper_id, fields_key)
);

-- Every extracted value with its evidence and verification result.
-- "data" holds the whole row (user-defined fields + quote + page + status ...), so any
-- department's columns fit without changing the schema.
CREATE TABLE IF NOT EXISTS trex_rows (
    id          BIGSERIAL PRIMARY KEY,
    run_id      INT NOT NULL REFERENCES trex_runs(id) ON DELETE CASCADE,
    status      TEXT,                      -- verified | mismatch | unverified
    page        INT,
    quote       TEXT,
    data        JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS trex_rows_run_idx ON trex_rows (run_id);
CREATE INDEX IF NOT EXISTS trex_rows_status_idx ON trex_rows (status);

-- Which runs each visitor has used (fresh run, reload from cache, or reopen), for their own History.
-- "visitor" is a random ID kept in the visitor's browser; there are no accounts.
CREATE TABLE IF NOT EXISTS trex_visitor_runs (
    visitor     TEXT NOT NULL,
    run_id      INT NOT NULL REFERENCES trex_runs(id) ON DELETE CASCADE,
    last_used   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (visitor, run_id)
);
CREATE INDEX IF NOT EXISTS trex_visitor_runs_recent_idx ON trex_visitor_runs (visitor, last_used DESC);
