# 🦖 TREX: Traceable Research Extractor

Upload research papers, name the facts you want (e.g. *Model, Dataset, Accuracy*), and get back one
comparison table. Every value is linked to a verbatim quote from the paper, and plain code (not AI)
checks that the quote really exists.

| Status | Meaning |
|---|---|
| ✅ verified | Quote found in the paper, and every number in the values appears in the quote |
| ⚠️ mismatch | Quote found, but a value's number isn't in it (the AI may have changed it) |
| ❌ unverified | Quote not found in the paper (the AI may have made it up) |

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in your LLM API key (+ Tiger DATABASE_URL, optional)
python tests/test_verify.py   # offline tests, no API key needed

# put PDFs in papers/
python cli.py papers/ --fields "Model,Dataset,Accuracy"   # prints rows, writes out/results.csv
streamlit run streamlit_app.py                            # web UI
```

## Tiger Cloud (optional database)

With a Tiger Cloud connection string in `.env`, TREX saves every extraction and reuses it:

```
DATABASE_URL=postgres://user:password@host:port/tsdb?sslmode=require
```

- **Cache:** the same PDF (matched by content, so renaming doesn't matter) with the same fields loads
  instantly from the database. No AI call, no cost. Use `--no-cache` (CLI) or untick
  "Reuse saved results" (website) to force a fresh extraction.
- **Library:** the website's "🐯 Saved results" panel reopens any past run, including the PDF, so
  highlighted source pages work even on the hosted site.
- Tables (`trex_papers`, `trex_runs`, `trex_rows`) are created automatically. See `schema.sql`.
- If the database is unreachable, TREX keeps working and prints a warning.

```bash
python cli.py papers --fields x --check-db      # test the connection
```

Database tests (use a throwaway database, since they drop and recreate the TREX tables):
```bash
TEST_DATABASE_URL=postgres://... python tests/test_db.py
```

## How it works

```
PDF ──► pdf_reader.py ──► extractor.py ──► verify.py ──► db.py (Tiger) ──► UI / CSV
        text per page     LLM returns       code checks    stores rows
                          values + quote    quote exists
                          + page as JSON    + numbers match
```

| File | Job |
|---|---|
| `app/pdf_reader.py` | PDF → list of pages with page numbers |
| `app/llm.py` | Calls Anthropic or OpenAI (set `LLM_PROVIDER` in `.env`) |
| `app/extractor.py` | Prompt + JSON parsing; fields are whatever the user types |
| `app/verify.py` | Fuzzy quote matching + number check → status |
| `app/pipeline.py` | Glues the above together for one paper |
| `app/db.py`, `schema.sql` | Tiger Cloud cache + library (skipped if no `DATABASE_URL`) |
| `cli.py` | Run on a folder of PDFs |
| `streamlit_app.py` | Upload-and-extract web UI |

---

# Build plan

## Workstreams (one owner each)

The pieces only talk through the **row format** below, so people can work in parallel without
waiting on each other.

```python
# one extracted result, after verification
{"paper": "Deep Residual Learning...", "file": "resnet.pdf",
 "Model": "ResNet-110", "Accuracy": "6.43% error",      # user-defined fields
 "quote": "ResNet-110 has 6.43% error", "page": 7,
 "found_page": 7, "match_score": 100.0, "status": "verified"}
```

### A. Extraction quality (core): owner: ______
Files: `app/extractor.py`, `app/llm.py`
- [x] Improved system prompt for multi-row extraction, exact quotes, no unit conversion
- [ ] Run on 10 real papers and fix prompt failures (missed results, merged rows, wrong pages)
- [x] Handle long papers: `select_relevant_pages` keeps abstract + results/experiments sections
- [x] Make sure one paper → many rows works (prompt + tests for multi-result)
- [x] Optional: `suggest_fields(topic)` helper

### B. Verification + inconsistency detector (our differentiator): owner: ______
Files: `app/verify.py`, new `app/consistency.py`, `tests/`
- [x] Tune `QUOTE_MATCH_THRESHOLD` (lowered to 85) + more robust number normalization
- [x] Inconsistency detector: `detect_intra_paper_inconsistencies` (abstract vs. table)
- [x] Cross-paper conflicts: `detect_cross_paper_conflicts`
- [x] Add a test for each new rule (`tests/test_consistency.py`)
- [x] Review fixes: JSON parsing with `[24]`-style citations in quotes; detector works with any field names (not just Model/Dataset); score uses the first number, not a random one
- [ ] Long-paper filter keeps most pages on real reports (words like "results" are everywhere). C's vector search should replace it

### C. Tiger Data (database + AI search): owner: ______ (cache + library built)
Files: `app/db.py`, `schema.sql`, new `app/search.py`
- [x] Cache of past extractions + saved-results library (tested on PostgreSQL 16)
- [ ] Confirm on your Tiger instance: `python cli.py papers --fields x --check-db`
- [ ] Store page chunks + embeddings (pgvector / Tiger AI features); for each field, retrieve the top chunks so long papers fit in the prompt
- [ ] Leaderboard query: best result per model per dataset
- [ ] Trend query by publication `year` (continuous aggregate) for the "progress over time" chart
- [ ] Ask Tiger reps what counts as an "innovative use" for their prize

### D. UI + demo: owner: ______ (base built: see streamlit_app.py)
Files: `streamlit_app.py`
- [x] Table with status icons, filters (status, paper, search), CSV download
- [x] Click a row → quote panel + rendered PDF page with the quote highlighted
- [x] Leaderboard tab + chart (in pandas for now; swap to C's SQL later)
- [x] Table summary generated only from verified rows
- [ ] Polish: app name/logo, empty states, test with 30 real papers

### E. Data, pitch, Devpost: owner: ______ (prep done: see docs/)
- [x] Test papers: 7 public AI technical reports in `papers/` (see `papers/SOURCES.md`), fields `Model, Benchmark, Score`
- [ ] 5 papers from a non-CS field for the "works in any department" demo
- [x] Answer key: `eval/gold.csv` (38 hand-checked rows) + `eval/score.py` to measure accuracy
- [ ] Run the manual baseline timing (`docs/BASELINE_AND_VIDEO.md`) and the accuracy check, then fill in the numbers
- [x] Pitch script + judge Q&A (`docs/PITCH.md`), Devpost draft (`docs/DEVPOST.md`), video shot list
- [ ] Pick a project name; record the video; paste into Devpost
- [x] Findings on real PDFs for A and B: `docs/NOTES_FOR_A_AND_B.md`

## Timeline

| When | Milestone |
|---|---|
| Sat by midnight | **Devpost draft created.** CLI works end to end on 10 papers (A, B basic) |
| Sun ~4 AM | DB storing results, UI shows the table, 30+ papers run (C, D) |
| Sun ~9 AM | Inconsistency detector + leaderboard + summary (B, C, D) |
| Sun 12 PM | **Hard submission deadline.** Feature freeze |
| Sun 12–1:45 PM | Demo rehearsal, Devpost polish, final edits by 1:45 PM |

## Rules for the team
- Don't commit `.env` (API keys / DB URL).
- Commit often (judges look at the GitHub history).
- If something blocks you for more than 30 min, flag it and cut scope; a working core beats half-built extras.
