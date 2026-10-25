"""Read a PDF into a list of pages (page numbers start at 1)."""
from dataclasses import dataclass
import re
from pathlib import Path

import pymupdf as fitz


@dataclass
class Page:
    number: int
    text: str


def read_pdf(path: str | Path) -> list[Page]:
    doc = fitz.open(path)
    pages = [Page(number=i + 1, text=page.get_text()) for i, page in enumerate(doc)]
    doc.close()
    return pages


def paper_title(path: str | Path) -> str:
    """PDF metadata title if present, else the file name."""
    doc = fitz.open(path)
    title = ((doc.metadata or {}).get("title") or "").strip()
    doc.close()
    junk = {"", "untitled", "title", "none", "pdf"}
    if title.lower() in junk or title.lower().startswith(("microsoft word", "untitled")) or len(title) < 4:
        return Path(path).stem
    return title


def pages_to_prompt_text(pages: list[Page], max_chars: int = 300_000) -> str:
    """Join pages with markers so the LLM can report page numbers."""
    parts, total = [], 0
    for p in pages:
        block = f"\n=== PAGE {p.number} ===\n{p.text}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "".join(parts)


# Keywords that usually mark the experimental results sections
_RESULT_KEYWORDS = re.compile(
    r"\b(results?|experiments?|evaluation|benchmark|table\s+\d|ablation|performance)\b",
    re.I,
)


def select_relevant_pages(pages: list[Page], keep_abstract: bool = True) -> list[Page]:
    """Prefer pages that look like results/experiments; always keep the first 2 pages."""
    if not pages:
        return pages
    selected = []
    # Always keep early pages (abstract / intro often states headline numbers)
    for p in pages[:2]:
        selected.append(p)
    for p in pages[2:]:
        if _RESULT_KEYWORDS.search(p.text):
            selected.append(p)
    # If we found almost nothing, fall back to the whole paper
    return selected if len(selected) > 2 else pages
