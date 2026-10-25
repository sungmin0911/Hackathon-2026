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
