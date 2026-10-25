# 🦖 TREX: Traceable Research Extractor

**Pull the numbers you need out of research papers, with every value traced to its exact page and quote.**

Researchers spend hours copying results out of papers into spreadsheets for literature reviews and
comparisons. AI can speed that up, but it also makes numbers up. TREX uses AI to *find* the values
and plain code to *check* them: every number must come with a verbatim quote, and the quote must
really exist in the paper.

Built at HackPSU (Fall 2026) for the Research Track.

---

## What it does

1. **Upload papers and name the columns you want**: `Model, Benchmark, Score` for ML,
   `Variable, Coefficient, Std. error, p-value` for statistics, `Drug, Dose, Outcome` for medicine.
   Any field works.
2. **TREX extracts every matching result**, including whole comparison tables with every competitor
   column, not just the paper's own model.
3. **Code verifies each value** against the paper:

   | Status | Meaning |
   |---|---|
   | ✅ Verified | The quote is in the paper, and every number in the value appears in it (signs included) |
   | ⚠️ Mismatch | The quote is real, but a number differs (wrong column, rounding, flipped sign) |
   | ❌ Not found | The quote isn't in the paper. Likely invented, so not trusted |

4. **Click any row** to see the original PDF page with the quote highlighted.

### Features

- **Any field, any discipline**: tested on AI/ML technical reports, statistics papers and numerical analysis papers
- **✨ Suggest fields**: reads your paper and proposes columns it actually reports (e.g. *6 models compared on 20 benchmarks*)
- **"Why nothing matched"**: if your fields aren't in the paper, TREX explains field by field
  (✅ found / 🟡 mentioned without values / ❌ not in this paper), suggests the closest alternative, and
  offers one-click fixes. It never invents values to fill the gap
- **Conflicts**: flags the same result reported with different numbers, within a paper or across papers
- **Leaderboard**: ranks results across papers using only verified rows; detects lower-is-better
  metrics (error, loss, latency, p-value)
- **Summary**: a written takeaway built from facts that code computes from *your* columns. Any number
  the AI writes that isn't in the data is flagged
- **Statistics-aware numbers**: typographic minus signs (`−0.021`), `(0.012)` standard errors, `.003`,
  `<0.001`, `2,431`, significance stars. A flipped sign is caught
- **Tiger Cloud memory**: every extraction is saved. The same paper with the same fields loads
  instantly, with no AI call and no cost, and past runs can be reopened anytime, PDF included
- **🕘 History sidebar**: every past extraction, newest first and grouped by day like a chat history.
  Search it, reopen one with a click, or open several to compare
- **Live progress** for long papers, request timeouts, and automatic retries on rate limits

## Results

- **38 / 38** hand-checked values extracted correctly from two real AI technical reports
  (DeepSeek-R1, Kimi-K2), every one ✅ verified. See `eval/gold.csv` and `eval/score.py`.
- **800+ values** from 2 papers in about 3 minutes, including every competitor model (Claude, GPT, Gemini, Llama).
- Verification caught the AI swapping two models' scores taken from a bar chart.

The answer key is small (38 values), so read this as "verified on 38 hand-checked values", not as a general accuracy rate.

---

## How it works

```
PDF ─► pdf_reader ─► extractor (AI) ─► verify (code) ─► consistency ─► db (Tiger Cloud) ─► web UI / CLI / CSV
       text per      compact tables      quote exists?     same result,     cache + library
       page          + quotes per row    numbers match?    other numbers?
```

- **Extraction** sends papers to the AI a few pages at a time, in parallel. Comparison tables come
  back in a compact form (headers once, then label + values + one quote per row), which keeps big
  tables (300+ numbers on one page) from being cut short. Charts are skipped, because their PDF text is scrambled.
- **Verification** is deterministic code: fuzzy quote matching against the page text, then an exact
  check that the values' numbers, including signs, appear in that quote.
- **The AI never grades itself.**

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env              # Windows: copy .env.example .env
```

Fill in `.env`:

```
LLM_PROVIDER=xai                  # or anthropic / openai
LLM_MODEL=<model name from your provider's console>
XAI_API_KEY=...
DATABASE_URL=postgres://...       # optional: Tiger Cloud, for the cache and library
```

**Website**

```bash
streamlit run streamlit_app.py
```

**Command line** (reads every PDF in a folder):

```bash
python cli.py papers --fields "Model,Benchmark,Score" --out out/results.csv
python cli.py papers --fields x --check-db         # test the Tiger Cloud connection
python cli.py papers --fields "..." --no-cache     # force a fresh extraction
```

**Measure accuracy against the answer key**

```bash
python eval/score.py out/results.csv eval/gold.csv
```

## Tiger Cloud (optional)

With `DATABASE_URL` set, TREX stores papers (by content hash, so renaming doesn't matter), runs and
every result (`trex_papers`, `trex_runs`, `trex_rows`; see `schema.sql`), created automatically.

- **Cache**: the same paper with the same fields loads from the database instead of calling the AI.
- **🕘 History**: every saved run in the sidebar, grouped into Today / Yesterday / Previous 7 days / Older,
  like a chat history. One click reopens a run (PDF included, so highlighted source pages work on the
  hosted site) and moves it to the top. Search by paper or field; **Open several at once** loads
  multiple runs together for comparison. The **▶ Try the demo** button loads saved runs the same way.
- **Activity log**, built on TimescaleDB features (in the database, not shown on the page):
  - every extraction, cache hit and History reopen is an event in **`trex_events`, a hypertable**;
    History uses it to put recently opened runs first
  - daily totals (fresh runs, reloads, AI calls made and saved, % verified) are kept in
    **`trex_events_daily`, a real-time continuous aggregate** refreshed hourly
  - a **compression policy** compresses events older than 7 days
  - without TimescaleDB the same log works as a plain table
- If the database is unreachable, TREX keeps working and shows a warning.

## Deploying (Streamlit Community Cloud)

1. Push to a public GitHub repo (`.gitignore` keeps `.env`, `.venv` and PDFs out).
2. On share.streamlit.io: **Create app** → this repo, branch `main`, file `streamlit_app.py`.
3. **Advanced settings → Secrets**:

```toml
LLM_PROVIDER = "xai"
LLM_MODEL = "..."
XAI_API_KEY = "..."
DATABASE_URL = "postgres://..."
TREX_PUBLIC = "true"
```

`TREX_PUBLIC` turns on visitor limits so a public link can't drain the API credits: papers per run,
pages and size per paper, AI requests per paper, live runs per visit and per day. All limits are
adjustable; see `.env.example`.

History and the demo contain other visitors' papers, so on the public site they
are **owner-only**: set `TREX_UNLOCK = "your-code"` in the secrets and enter the code under
**🔑 Owner unlock** in the sidebar to see them and remove the limits for your session.

## Project structure

| Path | Purpose |
|---|---|
| `streamlit_app.py` | Web interface |
| `cli.py` | Command-line runner |
| `app/extractor.py` | AI prompts, compact-table parsing, chunking, field suggestions and diagnosis |
| `app/verify.py` | Quote matching and number checks |
| `app/numbers.py` | Sign- and format-aware number reading |
| `app/consistency.py` | Conflict detection; works out which column is the score |
| `app/summary.py` | Field-based summary with number checking |
| `app/db.py`, `schema.sql` | Tiger Cloud cache and library |
| `app/llm.py` | AI provider wrapper (xAI/Grok, Anthropic, OpenAI) with timeouts and retries |
| `app/pdf_reader.py`, `app/highlight.py` | PDF text, rendering and quote highlighting |
| `eval/` | Hand-checked answer key and scoring script |
| `tests/` | Automated tests |
| `docs/` | Pitch, Devpost draft, build plan, notes |

## Tests

```bash
python tests/test_verify.py
python tests/test_extractor.py
python tests/test_consistency.py
python tests/test_summary.py
python tests/test_numbers.py
TEST_DATABASE_URL=postgres://... python tests/test_db.py   # needs a throwaway database
```

## Limitations

- **Numbers are verified exactly; text values** (e.g. a license name) are only checked for a real
  quote. Open the source for those.
- **Charts and figures** are skipped, and **scanned (image-only) PDFs** can't be read yet (no OCR).
- A ✅ on a table row means the number is in that row of the paper; the column assignment is the
  AI's, which is why the highlighted source is one click away.

## Built with

Python · PyMuPDF · RapidFuzz · pandas · Streamlit · Tiger Cloud (PostgreSQL) · xAI Grok API
