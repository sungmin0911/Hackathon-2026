"""Short written takeaway, built ONLY from verified rows and organized by the user's own fields.

How it works:
  1. Code computes the facts from the rows, using the user's columns: the score column is the last
     numeric field (e.g. Score), the entity is the first field (e.g. Model), and results are grouped
     by the field closest to the score (e.g. Benchmark). "Better" respects lower-is-better metrics
     such as error, loss or latency.
  2. The AI only turns those facts into a few sentences. It is told not to compute anything new.
  3. Code checks that every number in the summary appears in the facts or rows; any that don't
     are reported back so the UI can flag them.
"""
import json
import re
from collections import defaultdict

from .consistency import detect_cross_paper_conflicts, split_fields
from .llm import complete
from .numbers import first_number

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
LOWER_IS_BETTER = re.compile(r"error|loss|wer|cer|perplexity|latency|time|cost|rmse|mae|mse|p-?\s?value|aic|bic",
                             re.I)
MAX_GROUPS = 12

SYSTEM = """You write a short takeaway (3-5 sentences) of a results table for a researcher.
You get FACTS that were computed by code from verified rows, plus the user's column names.
Rules:
- Use ONLY the facts. Do not add outside knowledge, and do not compute new numbers
  (no averages, differences or percentages that are not in the facts).
- Talk in terms of the user's columns (e.g. "Model", "Benchmark", "Score"), using their names.
- Respect "better_is": if it says "lower", the lowest value is the best one.
- Mention: the overall picture, which items lead most often, notable gaps or spreads, and
  any cross-paper conflicts. If a focus is given, center the summary on it.
- Plain prose, no bullet points, no headings."""


def to_number(value) -> float | None:
    """First number in a value like '93.57%', '6.43% error' or '−0.021', sign included."""
    return first_number(value)


def _fmt(x: float) -> str:
    return f"{x:g}"


def lower_is_better(rows: list[dict], score_field: str) -> bool:
    """True if the score column looks like error/loss/latency (by its name or most of its values)."""
    if LOWER_IS_BETTER.search(score_field):
        return True
    vals = [str(r.get(score_field) or "") for r in rows]
    return bool(vals) and sum(bool(LOWER_IS_BETTER.search(v)) for v in vals) / len(vals) > 0.5


def filter_focus(rows: list[dict], fields: list[str], focus: str | None) -> list[dict]:
    """Rows whose field values mention the focus text (case-insensitive)."""
    if not focus or not focus.strip():
        return rows
    f = focus.strip().lower()
    return [r for r in rows if any(f in str(r.get(k) or "").lower() for k in fields)]


def build_facts(rows: list[dict], fields: list[str], focus: str | None = None) -> dict:
    """Compute the summary facts in code, organized by the user's fields."""
    key_fields, score_fields = split_fields(rows, fields)
    score_f = score_fields[0] if score_fields else None
    entity_f = key_fields[0] if key_fields else None
    group_f = key_fields[-1] if len(key_fields) > 1 else None   # e.g. Benchmark in Model, Benchmark, Score
    lower = lower_is_better(rows, score_f) if score_f else False

    facts = {
        "columns": fields,
        "score_column": score_f, "entity_column": entity_f, "group_column": group_f,
        "better_is": "lower" if lower else "higher",
        "focus": focus or None,
        "verified_rows": len(rows),
        "papers": sorted({str(r.get("paper") or r.get("file") or "") for r in rows} - {""}),
        "distinct_values": {k: len({str(r.get(k)) for r in rows if r.get(k) not in (None, "")}) for k in key_fields},
    }
    if not (score_f and entity_f):
        return facts

    scored = [(r, to_number(r.get(score_f))) for r in rows]
    scored = [(r, v) for r, v in scored if v is not None]
    pick_best = min if lower else max
    pick_worst = max if lower else min

    groups = defaultdict(list)
    for r, v in scored:
        groups[str(r.get(group_f)) if group_f else "all"].append((r, v))
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:MAX_GROUPS]

    leaders = defaultdict(int)
    per_group = []
    for g, items in ranked:
        best_r, best_v = pick_best(items, key=lambda t: t[1])
        worst_r, worst_v = pick_worst(items, key=lambda t: t[1])
        if len(items) > 1:
            leaders[str(best_r.get(entity_f))] += 1
        per_group.append({
            (group_f or "group"): g, "results": len(items),
            "best": {entity_f: best_r.get(entity_f), score_f: best_r.get(score_f)},
            "worst": {entity_f: worst_r.get(entity_f), score_f: worst_r.get(score_f)},
            "range": f"{_fmt(min(v for _, v in items))} to {_fmt(max(v for _, v in items))}",
        })
    facts["by_group"] = per_group
    facts["groups_shown"] = f"{len(per_group)} of {len(groups)}"
    facts["most_often_best"] = [{entity_f: k, "times_best": n}
                                for k, n in sorted(leaders.items(), key=lambda kv: -kv[1])[:5]]
    facts["cross_paper_conflicts"] = [
        {"result": c["label"], "values_by_paper": c["values_by_paper"]}
        for c in detect_cross_paper_conflicts(rows, fields)[:5]]
    return facts


MEASURE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:%|pp\b|percentage points?\b|points?\b)", re.I)


def unsupported_numbers(text: str, facts: dict, rows: list[dict]) -> list[str]:
    """Numbers in the summary that the data doesn't support (should be none).

    - A number used as a measurement ("6 points", "12.4%", "3 pp") must equal an actual score
      in the rows; a computed difference like "6 points behind" is not in the data, so it's flagged.
    - Any other number must appear somewhere in the facts or rows (counts, names like "Opus 4")."""
    score_f = facts.get("score_column")
    scores = {_fmt(v) for v in (to_number(r.get(score_f)) for r in rows) if v is not None} if score_f else set()
    allowed = set()
    for blob in (json.dumps(facts, default=str), json.dumps(rows, default=str)):
        allowed |= {_fmt(float(n)) for n in NUM_RE.findall(blob.replace(",", ""))}

    clean = text.replace(",", "")
    bad = []
    measured_spans = []
    for m in MEASURE_RE.finditer(clean):
        measured_spans.append(m.span(1))
        if _fmt(float(m.group(1))) not in scores:
            bad.append(m.group(1))
    for m in NUM_RE.finditer(clean):
        if m.span() in measured_spans:
            continue
        if _fmt(float(m.group(0))) not in allowed:
            bad.append(m.group(0))
    seen, out = set(), []
    for n in bad:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def summarize(rows: list[dict], fields: list[str], excluded: int, focus: str | None = None) -> dict:
    """Return {"text", "facts", "unsupported", "rows_used"}; "text" is "" if no rows match the focus."""
    rows = filter_focus(rows, fields, focus)
    if not rows:
        return {"text": "", "facts": {}, "unsupported": [], "rows_used": 0}
    facts = build_facts(rows, fields, focus)
    facts["excluded_unverified_rows"] = excluded
    text = complete(SYSTEM, f"FACTS:\n{json.dumps(facts, indent=1, default=str)}\n\nWrite the takeaway now.").strip()
    return {"text": text, "facts": facts, "unsupported": unsupported_numbers(text, facts, rows),
            "rows_used": len(rows)}
