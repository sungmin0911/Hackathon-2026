"""Ask the LLM to pull user-defined fields out of a paper, with evidence quotes.

The model returns two kinds of output:
  - "results": single results stated in sentences, one object each (as before)
  - "tables":  comparison tables in a COMPACT form: column headers once, then each row as
               label + list of values + one quote. Code expands these into one row per cell.

The compact table form matters: a big results table can hold ~300 numbers. Writing one JSON
object (with its own quote) per number makes the model stop early and skip most of the table.
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .llm import complete
from .pdf_reader import Page, pages_to_prompt_text, select_relevant_pages

SYSTEM = """You extract experimental results from research papers into JSON.
Return ONLY one JSON object, no prose, no markdown fences:

{
  "tables": [
    {
      "page": <int, from the nearest "=== PAGE N ===" marker above the table>,
      "column_field": "<which requested field the column headers fill, e.g. Model>",
      "row_field": "<which requested field the row labels fill, e.g. Benchmark>",
      "value_field": "<which requested field the cell values fill, e.g. Score>",
      "constant": {"<other requested field>": "<value that applies to the whole table, or null>"},
      "columns": ["<header 1>", "<header 2>", ...],
      "rows": [
        {"label": "<row label>", "values": ["<v1>", "<v2>", ...],
         "quote": "<row label and its numbers copied verbatim, e.g. MMLU (EM) 89.5 89.4 87.0>"}
      ]
    }
  ],
  "results": [
    {"<each requested field>": "<value or null>", "quote": "<verbatim span, <= 25 words>", "page": <int>}
  ]
}

Rules:
- Use "tables" for every comparison table. Include EVERY row and EVERY column: baselines and
  competitors (GPT, Claude, Gemini, Llama, ...) as well as the paper's own model. Do not stop early.
- "values" must have exactly one entry per column, in column order. Use "-" for an empty cell.
  Tables often appear as one value per line: count carefully against the header.
- Copy values EXACTLY as written. Do not convert units, compute complements, or round.
- Column headers are often split across lines (e.g. "Claude\\nSonnet 4"): join them into one clean name.
  Skip group headers like "Open Source" / "Proprietary" that span several columns.
- Use "results" only for numbers stated in sentences (e.g. "achieves 79.8% on AIME 2024").
- SKIP charts and figures (bar charts, plots, radar charts). Their text comes out of the PDF scrambled,
  so you can't tell which number belongs to which bar. Only use tables and sentences.
- column_field, row_field and value_field must be names from the requested fields.
- Never invent numbers, names, or quotes. If there is nothing relevant, return {"tables": [], "results": []}.
"""

PAGES_PER_CALL = 6     # normal pages are grouped
DENSE_PAGE_DECIMALS = 40  # pages with this many decimals (big tables) get their own call

SUGGEST_SYSTEM = """You are helping choose extraction fields for a research paper comparison.
Given a topic or paper description, propose 4-8 concise field names that would be useful
for a comparison table (e.g. Model, Dataset, Accuracy, Parameters, Year).
Return ONLY a JSON array of strings. No other text.
"""


def build_prompt(fields: list[str], paper_text: str) -> str:
    return (
        f"Requested fields: {json.dumps(fields)}\n\n"
        f"Paper text (pages marked with === PAGE N ===):\n{paper_text}\n\n"
        "Return the JSON object now."
    )


# ---------------- JSON parsing ----------------

def _decode_first(text: str, opener: str):
    """Decode the first complete JSON value starting with `opener` ('[' or '{').
    Uses a real JSON decoder, so brackets inside strings (e.g. citations like "[24]") are safe."""
    decoder = json.JSONDecoder()
    for m in re.finditer(re.escape(opener), text):
        try:
            data, _ = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        return data
    return None


def _decode_first_array(text: str) -> list | None:
    data = _decode_first(text, "[")
    return data if isinstance(data, list) else None


def _lower_keys(d: dict) -> dict:
    return {str(k).strip().lower(): v for k, v in d.items()}


def _looks_like_table(d) -> bool:
    return isinstance(d, dict) and "rows" in _lower_keys(d) and "columns" in _lower_keys(d)


def _normalize(data):
    """Accept the shapes models commonly return instead of exactly the one we asked for:
    {"Tables": ...} (any key case), a single table object, a bare list of tables,
    a one-key wrapper like {"data": {...}}, or the old flat list of row objects."""
    if isinstance(data, list):
        if data and all(_looks_like_table(x) for x in data if isinstance(x, dict)) and any(
                isinstance(x, dict) for x in data):
            return {"tables": [_lower_keys(x) for x in data if isinstance(x, dict)], "results": []}
        return data  # flat rows (or an unrelated list)
    if not isinstance(data, dict):
        return None
    d = _lower_keys(data)
    if "tables" in d or "results" in d:
        tables = d.get("tables") or []
        if isinstance(tables, dict):
            tables = [tables]
        return {"tables": [_lower_keys(x) for x in tables if isinstance(x, dict)],
                "results": [x for x in (d.get("results") or []) if isinstance(x, dict)]}
    if _looks_like_table(d):
        return {"tables": [d], "results": []}
    if len(d) == 1:  # unwrap {"data": {...}} / {"output": [...]}
        return _normalize(next(iter(d.values())))
    return None


def _size(norm) -> int:
    """How much data a normalized reply holds (cells in tables + sentence results)."""
    if isinstance(norm, list):
        return sum(1 for x in norm if isinstance(x, dict))
    if isinstance(norm, dict):
        cells = sum(len(r.get("values") or []) for t in norm.get("tables", [])
                    for r in (t.get("rows") or []) if isinstance(r, dict))
        return cells + len(norm.get("results", []))
    return 0


def _all_json_values(text: str):
    """Every complete JSON object/array in the text, outermost first (nested ones are skipped)."""
    decoder = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        j = min([k for k in (text.find("{", i), text.find("[", i)) if k != -1], default=-1)
        if j == -1:
            return
        try:
            value, end = decoder.raw_decode(text, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        yield value
        i = end


def _decode_response(raw: str):
    """Return the model's reply as {"tables": [...], "results": [...]} or a flat list; None if unreadable.

    Some models write text before the JSON (reasoning, or an echo of the instructions like
    'return {"tables": [], "results": []}'). So every JSON block is read and the one with the
    MOST data wins, not simply the first one."""
    raw = raw or ""
    best, best_size = None, -1
    for text in (raw, re.sub(r",\s*([}\]])", r"\1", raw)):  # retry after removing trailing commas
        for value in _all_json_values(text):
            norm = _normalize(value)
            if norm is not None and _size(norm) > best_size:
                best, best_size = norm, _size(norm)
        if best_size > 0:
            break
    return best


def parse_json_array(raw: str) -> list[dict]:
    """Old flat format: a JSON array of row objects (kept for compatibility)."""
    data = _decode_response(raw)
    if isinstance(data, dict):
        data = data.get("results") or []
    return [d for d in (data or []) if isinstance(d, dict)]


# ---------------- table expansion ----------------

def _pick_field(name, fields: list[str], fallback: str) -> str:
    """Match the model's field name to a requested field (case-insensitive), else fall back."""
    for f in fields:
        if str(name or "").strip().lower() == f.lower():
            return f
    return fallback


def _clean(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def expand_table(t: dict, fields: list[str]) -> list[dict]:
    """Turn one compact table into one row per cell, in the shared row format."""
    if len(fields) < 2:
        return []
    value_f = _pick_field(t.get("value_field"), fields, fields[-1])
    others = [f for f in fields if f != value_f]
    col_f = _pick_field(t.get("column_field"), others, others[0])
    rest = [f for f in others if f != col_f]
    row_f = _pick_field(t.get("row_field"), rest, rest[0]) if rest else None
    constant = {k: v for k, v in (t.get("constant") or {}).items() if k in fields}

    columns = [_clean(c) for c in (t.get("columns") or [])]
    page = t.get("page") if isinstance(t.get("page"), int) else None
    out = []
    for r in t.get("rows") or []:
        values = r.get("values") or []
        aligned = len(values) == len(columns)
        for col, val in zip(columns, values):
            val = _clean(val)
            if not col or val in ("", "-", "–", "—", "n/a", "N/A", "null", "None"):
                continue
            row = {f: None for f in fields}
            row.update(constant)
            row[col_f] = col
            if row_f:
                row[row_f] = _clean(r.get("label"))
            row[value_f] = val
            row["quote"] = _clean(r.get("quote")) or _clean(f"{r.get('label', '')} {' '.join(map(str, values))}")
            row["page"] = page
            if not aligned:
                # value count didn't match the header: column assignment may be off
                row["alignment"] = "uncertain"
            out.append(row)
    return out


def parse_response(raw: str, fields: list[str]) -> list[dict]:
    """Parse a model reply (new object format or old flat array) into flat rows."""
    data = _decode_response(raw)
    if isinstance(data, list):  # old format
        return [d for d in data if isinstance(d, dict)]
    if not isinstance(data, dict):
        return []
    rows = []
    for t in data.get("tables") or []:
        if isinstance(t, dict):
            rows.extend(expand_table(t, fields))
    rows.extend(d for d in (data.get("results") or []) if isinstance(d, dict))
    return rows


# ---------------- page selection + chunking ----------------

_DECIMAL = re.compile(r"\d+\.\d+")


def _decimals(p: Page) -> int:
    return len(_DECIMAL.findall(p.text))


def pages_with_results(pages: list[Page], min_decimals: int = 3) -> list[Page]:
    """Keep the first 2 pages plus pages with several decimal numbers (tables, result sentences).
    Skips references and pure-prose pages, which saves cost and keeps the model focused."""
    keep = [p for i, p in enumerate(pages) if i < 2 or _decimals(p) >= min_decimals]
    return keep if len(keep) > 2 else pages


def make_chunks(pages: list[Page], size: int = PAGES_PER_CALL,
                dense: int = DENSE_PAGE_DECIMALS) -> list[list[Page]]:
    """Dense table pages go alone; other pages are grouped up to `size`, in page order."""
    chunks, group = [], []
    for p in pages:
        if _decimals(p) >= dense:
            if group:
                chunks.append(group)
                group = []
            chunks.append([p])
        else:
            group.append(p)
            if len(group) == size:
                chunks.append(group)
                group = []
    if group:
        chunks.append(group)
    return chunks


def _chunks(pages: list[Page], size: int) -> list[list[Page]]:  # kept for compatibility
    return make_chunks(pages, size)


def _dedupe(rows: list[dict], fields: list[str]) -> list[dict]:
    """Remove exact repeats (same values on the same page). Repeats across pages are kept here
    so the pipeline can keep the copy that verifies best."""
    seen, out = set(), []
    for r in rows:
        key = tuple(_clean(r.get(f)).lower() for f in fields) + (r.get("page"),)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


RAW_REPLY_DIR = os.getenv("RAW_REPLY_DIR", "out/raw_replies")


def _save_raw(label: str, chunk: list[Page], raw: str) -> str:
    """Keep each model reply on disk so empty or odd results can be inspected."""
    try:
        folder = Path(RAW_REPLY_DIR)
        folder.mkdir(parents=True, exist_ok=True)
        pages_txt = f"{chunk[0].number}-{chunk[-1].number}" if len(chunk) > 1 else str(chunk[0].number)
        path = folder / f"{label or 'paper'}_p{pages_txt}.txt"
        path.write_text(raw or "", encoding="utf-8")
        return str(path)
    except OSError:
        return ""


def _warn(msg: str) -> None:
    print(f"  [warning] {msg}", file=sys.stderr, flush=True)


def extract(fields: list[str], pages: list[Page], max_chars: int = 180_000,
            pages_per_call: int = PAGES_PER_CALL, workers: int = 4, label: str = "",
            on_progress=None, max_calls: int | None = None) -> list[dict]:
    """Extract rows chunk by chunk (in parallel), then merge and remove duplicates.

    on_progress(done, total, pages_text) is called after each chunk finishes, from the calling
    thread (safe for Streamlit), so long papers show progress instead of looking frozen."""
    relevant = pages_with_results(pages)
    if sum(len(p.text) for p in relevant) > max_chars * 3:
        relevant = select_relevant_pages(relevant)

    def run(chunk: list[Page]) -> list[dict]:
        text = pages_to_prompt_text(chunk, max_chars=max_chars)
        raw = complete(SYSTEM, build_prompt(fields, text))
        saved = _save_raw(label, chunk, raw)
        pages_txt = ", ".join(str(p.number) for p in chunk)
        where = f" (raw reply: {saved})" if saved else ""
        if not raw.strip():
            _warn(f"the model returned an EMPTY reply for page(s) {pages_txt}{where}")
            return []
        if _decode_response(raw) is None:
            _warn(f"couldn't read the model's reply for page(s) {pages_txt} (cut off or not JSON){where}")
            return []
        return parse_response(raw, fields)

    chunks = make_chunks(relevant, pages_per_call)
    if max_calls and len(chunks) > max_calls:
        # keep the parts with the most table data (where results live), then restore page order
        keep = sorted(range(len(chunks)), key=lambda i: -sum(_decimals(p) for p in chunks[i]))[:max_calls]
        skipped = len(chunks) - max_calls
        chunks = [chunks[i] for i in sorted(keep)]
        _warn(f"limit: read the {max_calls} most table-heavy parts of this paper, skipped {skipped} "
              f"(public demo limit)")
    results: list[list[dict]] = [[] for _ in chunks]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, c): i for i, c in enumerate(chunks)}
        for done, fut in enumerate(as_completed(futures), start=1):
            i = futures[fut]
            results[i] = fut.result()  # re-raises a chunk's error, as before
            if on_progress:
                c = chunks[i]
                pages_txt = f"p.{c[0].number}" + (f"-{c[-1].number}" if len(c) > 1 else "")
                on_progress(done, len(chunks), pages_txt)
    return _dedupe([r for rows in results for r in rows], fields)  # page order kept


PAPER_SUGGEST_SYSTEM = """You help a researcher decide which COLUMNS to extract from a research paper.
You see text from the paper's results pages.

Think in two steps:
1. Find the paper's most important comparisons: WHAT is being compared (e.g. models, methods, drugs),
   ON WHAT (e.g. benchmarks, datasets, patient groups), and MEASURED HOW (e.g. score, accuracy, dose, outcome).
2. Turn each comparison into GENERAL column names (categories), never specific values.
   Correct:   ["Model", "Benchmark", "Score"]  where "MATH-500" and "GPQA" are VALUES of Benchmark.
   Wrong:     ["MATH-500", "Pass@1", "GPQA"]   (these are specific benchmarks and metrics, i.e. values)
   Wrong:     ["DeepSeek-R1", "Accuracy"]      (DeepSeek-R1 is a value of Model)

Rules:
- 3-4 columns per set: identifying columns first, the main NUMERIC measurement LAST.
- Column names are short, plain, capitalized category words (Model, Method, Benchmark, Dataset, Task,
  Metric, Score, Accuracy, Parameters, Drug, Dose, Sample size, Outcome, ...).
- Only propose comparisons that are really in the text. Most important (biggest results table) first.
- Give 1-3 sets.

Return ONLY JSON:
{"suggestions": [
  {"fields": ["Model", "Benchmark", "Score"],
   "why": "<one short sentence: what this comparison shows, e.g. 6 models compared on 20 benchmarks>",
   "example": {"Model": "<a real value from the text>", "Benchmark": "<real value>", "Score": "<real value>"}}
]}
"""

# Column names are categories ("Model", "Benchmark"). Names with digits or "@" are almost always
# specific values ("DeepSeek-R1", "MATH-500", "Pass@1", "GPT-4"), except a few real metric names.
_METRIC_WITH_DIGITS = re.compile(r"(?i)\b(f1|r2|top-?\d+)\b")


def _looks_like_value(name: str) -> bool:
    if "@" in name:
        return True
    if re.search(r"\d", name):
        return not _METRIC_WITH_DIGITS.search(name)
    return False


def _clean_suggestion(item: dict) -> dict | None:
    """Keep a suggestion only if its column names look like categories, not specific values."""
    if not isinstance(item, dict):
        return None
    fields = [_clean(f) for f in (item.get("fields") or []) if isinstance(f, str) and _clean(f)]
    example = item.get("example") if isinstance(item.get("example"), dict) else {}
    example_values = {_clean(v).lower() for v in example.values() if isinstance(v, (str, int, float))}
    fields = [f for f in fields if not _looks_like_value(f) and f.lower() not in example_values]
    if len(fields) < 2:
        return None
    return {"fields": fields[:4], "why": _clean(item.get("why")),
            "example": {f: _clean(example.get(f)) for f in fields[:4] if example.get(f) not in (None, "")}}


def suggest_fields_from_pages(pages: list[Page], max_chars: int = 40_000) -> list[dict]:
    """Propose column sets this paper actually reports, by showing the model its results pages.

    Returns e.g. [{"fields": ["Model", "Benchmark", "Score"],
                   "why": "6 models compared on 20 benchmarks",
                   "example": {"Model": "DeepSeek-R1", "Benchmark": "MATH-500", "Score": "97.3"}}]"""
    relevant = sorted(pages_with_results(pages), key=_decimals, reverse=True)  # table-heavy pages first
    text = pages_to_prompt_text(sorted(relevant[:6], key=lambda p: p.number), max_chars=max_chars)
    raw = complete(PAPER_SUGGEST_SYSTEM, f"Paper text:\n{text}\n\nReturn the JSON now.")

    items = []
    for value in _all_json_values(raw or ""):
        if isinstance(value, dict):
            d = _lower_keys(value)
            if isinstance(d.get("suggestions"), list):
                items = d["suggestions"]
                break
            if "fields" in d:
                items = [d]
                break
        elif isinstance(value, list):  # older shapes: [[...], [...]] or ["Model", ...]
            if value and all(isinstance(x, str) for x in value):
                items = [{"fields": value}]
            else:
                items = [{"fields": x} if isinstance(x, list) else x for x in value]
            break

    out, seen = [], set()
    for item in items:
        s = _clean_suggestion(item)
        if s and tuple(f.lower() for f in s["fields"]) not in seen:
            seen.add(tuple(f.lower() for f in s["fields"]))
            out.append(s)
    return out[:3]


DIAGNOSE_SYSTEM = """A researcher asked to extract these columns from a paper, but no complete rows were found.
For EACH requested column, decide from the text whether the paper reports values for it.

Statuses:
- "reported": the text gives concrete values for this column (e.g. model names, scores).
- "mentioned_only": the topic is mentioned, but no concrete per-item values are given
  (e.g. "training was efficient" without numbers, or one total figure instead of per-model values).
- "not_found": the text does not talk about it at all.

Return ONLY JSON:
{"fields": [
  {"field": "<column name exactly as requested>",
   "status": "reported" | "mentioned_only" | "not_found",
   "explanation": "<one short, specific sentence about THIS paper, e.g. 'The paper never states training time for any model.'>",
   "example": "<a real value from the text if reported, else null>",
   "page": <page number of the example or mention, else null>,
   "closest": "<a column this paper DOES report that is closest in meaning, else null>"}
]}
"""


def _mention_pages(field: str, pages: list[Page]) -> list[int]:
    """Pages whose text contains the field name as a phrase (plain search, no AI)."""
    phrase = re.sub(r"\s+", " ", field.strip().lower())
    if len(phrase) < 3:
        return []
    return [p.number for p in pages if phrase in re.sub(r"\s+", " ", p.text.lower())]


def diagnose_fields(pages: list[Page], fields: list[str], max_chars: int = 40_000) -> list[dict]:
    """Explain, per requested field, whether this paper reports it. Used when extraction found nothing.

    Returns [{"field", "status", "explanation", "example", "page", "closest", "mentioned_on"}].
    "mentioned_on" comes from a plain text search, so it's a hard fact rather than the AI's opinion."""
    mentions = {f: _mention_pages(f, pages) for f in fields}
    # show the model the table-heavy pages plus any page that mentions a requested field
    wanted = {n for pg in mentions.values() for n in pg[:3]}
    dense = sorted(pages_with_results(pages), key=_decimals, reverse=True)[:5]
    chosen = sorted({p.number: p for p in dense + [p for p in pages if p.number in wanted]}.values(),
                    key=lambda p: p.number)
    text = pages_to_prompt_text(chosen, max_chars=max_chars)
    raw = complete(DIAGNOSE_SYSTEM,
                   f"Requested columns: {json.dumps(fields)}\n\nPaper text:\n{text}\n\nReturn the JSON now.")

    by_name = {}
    for value in _all_json_values(raw or ""):
        if isinstance(value, dict) and isinstance(_lower_keys(value).get("fields"), list):
            for item in _lower_keys(value)["fields"]:
                if isinstance(item, dict):
                    item = _lower_keys(item)
                    by_name[_clean(item.get("field")).lower()] = item
            break

    out = []
    for f in fields:
        item = by_name.get(f.lower(), {})
        status = item.get("status") if item.get("status") in ("reported", "mentioned_only", "not_found") else None
        if status is None:  # AI gave nothing usable: fall back to the plain search alone
            status = "mentioned_only" if mentions[f] else "not_found"
        if status == "not_found" and mentions[f]:
            status = "mentioned_only"  # the words ARE in the paper, so "not found" would be wrong
        page = item.get("page") if isinstance(item.get("page"), int) else None
        closest = _clean(item.get("closest")) or None
        if closest and (closest.lower() == f.lower() or _looks_like_value(closest)):
            closest = None
        out.append({"field": f, "status": status,
                    "explanation": _clean(item.get("explanation")) or None,
                    "example": _clean(item.get("example")) or None,
                    "page": page, "closest": closest, "mentioned_on": mentions[f][:5]})
    return out


def suggest_fields(topic: str) -> list[str]:
    """Optional helper: propose fields from a topic string."""
    raw = complete(SUGGEST_SYSTEM, f"Topic or paper: {topic}\nReturn the JSON array of field names.")
    data = _decode_first_array(raw or "") or []
    return [str(x) for x in data if isinstance(x, str)]
