"""Check, with plain code (no AI), that each extracted value is backed by the paper text.

Statuses:
  verified   - quote found in the paper AND every number in the values appears in the quote
  mismatch   - quote found, but a value's number is not in the quote (AI may have altered it)
  unverified - quote not found in the paper (AI may have made it up)
"""
import re
import unicodedata

from rapidfuzz import fuzz

from .numbers import looks_numeric, signed_numbers
from .pdf_reader import Page

# Tuned on real paper extracts: 85 catches most true quotes even when PDF table
# extraction inserts extra whitespace or ligatures; 90 was too strict for some tables.
QUOTE_MATCH_THRESHOLD = 85


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # fixes ligatures like "ﬁ"
    text = re.sub(r"-\s*\n\s*", "", text)        # re-join words hyphenated across lines
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def find_quote(quote: str, pages: list[Page], hint_page: int | None) -> tuple[int | None, float]:
    """Return (page_number, score) of the best match; the hinted page is checked first."""
    q = normalize(quote)
    if not q:
        return None, 0.0
    ordered = sorted(pages, key=lambda p: p.number != hint_page)
    best_page, best_score = None, 0.0
    for p in ordered:
        score = fuzz.partial_ratio(q, normalize(p.text))
        if score > best_score:
            best_page, best_score = p.number, score
        if score >= 99:
            break
    return best_page, best_score


def numbers_in(text: str) -> set[str]:
    """Numbers with their signs, canonicalized so '6.43%', '6.430' and '6.43' match,
    and '−0.021' (typographic minus) matches '-0.021' but NOT '0.021'."""
    return set(signed_numbers(text))


def verify_row(row: dict, fields: list[str], pages: list[Page]) -> dict:
    quote = row.get("quote") or ""
    hint = row.get("page") if isinstance(row.get("page"), int) else None
    page, score = find_quote(quote, pages, hint)

    if score < QUOTE_MATCH_THRESHOLD:
        status = "unverified"
    else:
        # Only check values that ARE numbers ("79.8", "6.43% error"). Numbers inside names
        # ("DeepSeek-R1", "AIME 2024") are labels and usually aren't in a table-row quote.
        quote_nums = numbers_in(normalize(quote))
        numeric_values = [str(row.get(f)) for f in fields if looks_numeric(row.get(f))]
        value_nums = set().union(*(numbers_in(v) for v in numeric_values)) if numeric_values else set()
        status = "verified" if value_nums <= quote_nums else "mismatch"
        if status == "verified" and row.get("alignment") == "uncertain":
            status = "mismatch"  # number is real, but the table's column count didn't line up

    return {**row, "found_page": page, "match_score": round(score, 1), "status": status}
