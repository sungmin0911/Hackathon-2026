# Devpost draft

Paste into devpost.hackpsu.org. Fill in `[ ]` with real numbers before the 1:45 PM deadline.
**Prize categories to select:** Research Track, Grand Prize, Tiger Data (if listed as a category).

---

**Project name:** [ ] *(e.g. "TableProof", "Citecheck", "PaperTable"; pick one)*

**Tagline:** Turn a pile of research papers into a verified comparison table, with every number traced to its source.

## Inspiration
Before starting any research project, you need to know what's already been done. In practice that
means opening dozens of papers, hunting for results tables, and copying numbers into a spreadsheet by
hand. It's one of the slowest, most error-prone steps of a literature review. AI tools can help, but
they make up numbers, and in research a wrong number is worse than no number.

## What it does
- Upload papers and type the columns you want (e.g. *Model, Benchmark, Score* or *Drug, Dose, Outcome*). It works for any field.
- An LLM extracts every matching result, plus an exact quote and page number for each one.
- **Plain code, not AI, verifies each value**: the quote must exist in the paper, and the numbers must appear in the quote.
  - ✅ verified · ⚠️ number mismatch · ❌ quote not found
- Click any row to see the original PDF page with the quote highlighted.
- A leaderboard ranks results across papers using only verified rows, and a summary is written only from verified data.

## How we built it
Python, PyMuPDF (PDF text, page rendering, highlighting), [LLM provider], RapidFuzz (fuzzy quote
matching), Streamlit (UI), and **Tiger Data** (Postgres) for storing papers and extractions [+ vector search, fill in].

## Challenges we ran into
- PDF tables extract as one number per line, with model names far away in the header, so matching numbers to the right model is hard.
- Different reports test different versions of the same model (e.g. DeepSeek-V3 vs DeepSeek-V3-0324), so "conflicting" numbers often aren't errors.
- [ ]

## Accomplishments we're proud of
- On [38] hand-checked values from real AI technical reports, [ ]% were extracted correctly.
- [ ]

## What we learned
[ ]

## What's next
- Structured table parsing and vision models for complex tables
- OCR for scanned papers
- Export to systematic-review tools; Zotero integration
- [Nittany AI Challenge prototype, if you decide to continue]

**Built with:** python, pymupdf, streamlit, postgresql, tiger-data, rapidfuzz, [llm]

**Links:** GitHub: [public repo URL] · Demo video: [URL]
