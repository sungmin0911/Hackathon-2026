"""Detect inconsistencies in extracted results.

Intra-paper: same identifying fields (e.g. Model + Dataset) reported with different numbers
             inside one paper (e.g. abstract vs. table).
Cross-paper: different papers report different numbers for the same identifying fields.

Works with any user-defined fields: columns whose values mostly start with a number
("89.5", "6.43% error") are treated as scores; every other column identifies the result.
So "Model, Benchmark, Score" and "Drug, Dose, Outcome" both work without configuration.

These flags mean "reported differently, worth a look", not "error": papers often test
different versions or settings of the same model.
"""
import re
from collections import defaultdict

from .numbers import looks_numeric, signed_numbers
from .verify import normalize



def _looks_numeric(value) -> bool:
    return looks_numeric(value)


def split_fields(rows: list[dict], fields: list[str]) -> tuple[list[str], list[str]]:
    """Return (key_fields, score_fields).

    The score is the LAST column whose values mostly start with a number; every other column
    (including earlier numeric ones like Dose or Parameters) identifies the result.
    If no column looks numeric, the last field is used as the score."""
    def numeric_share(f):
        vals = [r.get(f) for r in rows if r.get(f) not in (None, "")]
        return sum(_looks_numeric(v) for v in vals) / len(vals) if vals else 0

    numeric = [f for f in fields if numeric_share(f) > 0.5]
    score = numeric[-1] if numeric else (fields[-1] if len(fields) > 1 else None)
    if score is None:
        return list(fields), []
    return [f for f in fields if f != score], [score]


def first_number(value) -> str | None:
    """First number in order of appearance, sign kept ('88.3 (5-shot)' -> '88.3', '−0.021' -> '-0.021')."""
    nums = signed_numbers(value)
    return nums[0] if nums else None


def _key(row: dict, key_fields: list[str]) -> tuple[str, ...] | None:
    parts = tuple(normalize(str(row.get(f) or "")) for f in key_fields)
    return parts if key_fields and all(parts) else None


def _score(row: dict, score_fields: list[str]) -> str | None:
    for f in score_fields:
        n = first_number(row.get(f))
        if n is not None:
            return n
    return None


def _conflict_base(key: tuple[str, ...], key_fields: list[str]) -> dict:
    return {"key": dict(zip(key_fields, key)), "label": " / ".join(key)}


def detect_intra_paper_inconsistencies(rows: list[dict], fields: list[str]) -> list[dict]:
    """Same identifying fields, different numbers, within ONE paper (grouped per paper)."""
    key_fields, score_fields = split_fields(rows, fields)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        k = _key(r, key_fields)
        if k:
            groups[(r.get("file") or r.get("paper") or "", k)].append(r)

    conflicts = []
    for (_, k), group in groups.items():
        numbers = {_score(r, score_fields) for r in group} - {None}
        if len(numbers) > 1:
            conflicts.append({"type": "intra", **_conflict_base(k, key_fields),
                              "values": sorted(numbers), "rows": group, "status": "inconsistent"})
    return conflicts


def detect_cross_paper_conflicts(all_rows: list[dict], fields: list[str]) -> list[dict]:
    """Different papers report different numbers for the same identifying fields."""
    key_fields, score_fields = split_fields(all_rows, fields)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_rows:
        k = _key(r, key_fields)
        if k:
            groups[k].append(r)

    conflicts = []
    for k, group in groups.items():
        by_paper: dict[str, set[str]] = defaultdict(set)
        for r in group:
            n = _score(r, score_fields)
            if n is not None:
                by_paper[r.get("paper") or r.get("file") or "unknown"].add(n)
        if len(by_paper) > 1 and len(set().union(*by_paper.values())) > 1:
            conflicts.append({"type": "cross", **_conflict_base(k, key_fields),
                              "values_by_paper": {p: sorted(v) for p, v in by_paper.items()},
                              "status": "conflict"})
    return conflicts


def annotate_rows_with_consistency(rows: list[dict], fields: list[str]) -> list[dict]:
    """Add an 'inconsistency' key to rows that take part in an intra-paper conflict."""
    flagged = {id(r) for c in detect_intra_paper_inconsistencies(rows, fields) for r in c["rows"]}
    return [{**r, "inconsistency": "intra-paper"} if id(r) in flagged else dict(r) for r in rows]
