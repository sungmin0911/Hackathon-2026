"""Render a PDF page as an image with the evidence quote highlighted."""
import re

import pymupdf as fitz


def _candidates(quote: str) -> list[str]:
    """Full quote first, then shorter pieces (PDF line breaks often split long quotes)."""
    q = re.sub(r"\s+", " ", quote or "").strip()
    if not q:
        return []
    out = [q]
    words = q.split()
    for size in (8, 5):  # sliding windows of words
        out += [" ".join(words[i:i + size]) for i in range(0, max(len(words) - size + 1, 1), size // 2 or 1)]
    out += [n for n in re.findall(r"\d+(?:\.\d+)?%?", q) if len(n) >= 3]  # numbers as a last resort
    seen, uniq = set(), []
    for c in out:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def render_page(pdf_path: str, page_number: int, quote: str, zoom: float = 2.0) -> tuple[bytes, bool]:
    """Return (png_bytes, highlighted?). page_number starts at 1."""
    doc = fitz.open(pdf_path)
    page = doc[max(0, min(page_number - 1, len(doc) - 1))]
    found = False
    for cand in _candidates(quote):
        rects = page.search_for(cand)
        if rects:
            for r in rects:
                page.add_highlight_annot(r)
            found = True
            if len(cand) > 12:  # a real phrase matched; stop before numbers-only fallbacks
                break
    png = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png")
    doc.close()
    return png, found
