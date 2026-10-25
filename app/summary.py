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
from .numbers import first_number, signed_numbers

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
- Respect "better_is" (per group if given): "lower" = lowest is best, "closer to 0" = smallest absolute
  value is best (e.g. bias), "none" = no best/worst (e.g. mean, coverage): just describe the range.
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


_GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ζ": "zeta", "η": "eta",
          "θ": "theta", "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi",
          "ρ": "rho", "σ": "sigma", "ς": "sigma", "τ": "tau", "υ": "upsilon", "φ": "phi", "χ": "chi",
          "ψ": "psi", "ω": "omega"}


def canon(text) -> str:
    """Text for matching, ignoring symbol style: 'µ₁' (micro sign), 'μ1' and 'mu 1' all become 'mu1';
    'σ²₁' -> 'sigma21'; case, spaces, '_' and '^' are ignored."""
    import unicodedata
    s = unicodedata.normalize("NFKC", str(text or "")).lower()   # micro sign -> μ, ₁ -> 1, ² -> 2
    s = "".join(_GREEK.get(ch, ch) for ch in s)
    return re.sub(r"[\s_^{}\\]+", "", s)


def focus_column(fields: list[str], focus: str | None) -> str | None:
    """The column the focus names, if it is a column name (e.g. 'parameter' -> 'Parameter')."""
    if not focus:
        return None
    f = canon(focus)
    for col in fields:
        if canon(col) == f or canon(col).rstrip("s") == f.rstrip("s"):
            return col
    return None


def filter_focus(rows: list[dict], fields: list[str], focus: str | None) -> list[dict]:
    """Rows whose values (or table setting, or paper) mention the focus, ignoring symbol style."""
    if not focus or not focus.strip():
        return rows
    f = canon(focus)
    return [r for r in rows
            if any(f in canon(r.get(k)) for k in [*fields, "setting", "paper"])]


def focus_examples(rows: list[dict], fields: list[str], per_field: int = 4) -> dict[str, list[str]]:
    """A few example values per identifying column, to suggest what the focus can be."""
    key_fields, _ = split_fields(rows, fields)
    out = {}
    for k in key_fields:
        seen = []
        for r in rows:
            v = str(r.get(k) or "").strip()
            if v and v not in seen:
                seen.append(v)
            if len(seen) >= per_field:
                break
        if seen:
            out[k] = seen
    return out


METRIC_COLUMN = re.compile(r"measure|metric|statistic|criterion|medida", re.I)
METRIC_WORDS = re.compile(r"^\s*(bias|abs(olute)? bias|rmse|mse|mae|mape|mean|median|sd|std|se|standard error|"
                          r"variance|coverage|accuracy|acc|f1|auc|precision|recall|power|type i error|"
                          r"error|loss|r2|r²|correlation|width|length|ci width)\b", re.I)
CLOSER_TO_ZERO = re.compile(r"bias", re.I)
NO_DIRECTION = re.compile(r"^\s*(mean|median|coverage|correlation|estimate)\b", re.I)


def metric_column(rows: list[dict], key_fields: list[str], exclude: str | None) -> str | None:
    """A column whose values are measure names (Bias, RMSE, Coverage...). Each measure must be
    compared on its own, with its own direction, instead of being mixed into one score."""
    for k in key_fields:
        if k == exclude:
            continue
        vals = [str(r.get(k) or "") for r in rows if r.get(k)]
        if METRIC_COLUMN.search(k) or (vals and sum(bool(METRIC_WORDS.match(v)) for v in vals) / len(vals) > 0.6):
            return k
    return None


def direction_for(metric_name: str, rows: list[dict], score_f: str) -> str:
    """'higher', 'lower', 'closer to 0' or 'none' (e.g. Mean or Coverage have no simple 'better')."""
    if CLOSER_TO_ZERO.search(metric_name):
        return "closer to 0"
    if NO_DIRECTION.match(metric_name):
        return "none"
    if LOWER_IS_BETTER.search(metric_name):
        return "lower"
    return "lower" if lower_is_better(rows, score_f) else "higher"


def build_facts(rows: list[dict], fields: list[str], focus: str | None = None,
                group_by: str | None = None) -> dict:
    """Compute the summary facts in code, organized by the user's fields.
    group_by: a column to group results by (when the focus names a column).
    If a column holds measure names (Bias, RMSE...), each measure is compared separately."""
    key_fields, score_fields = split_fields(rows, fields)
    score_f = score_fields[0] if score_fields else None
    group_f = group_by if group_by in key_fields else (key_fields[-1] if len(key_fields) > 1 else None)
    metric_f = metric_column(rows, key_fields, exclude=group_f)
    if metric_f and group_f is None:
        group_f = None
    others = [k for k in key_fields if k not in (group_f, metric_f)]
    entity_f = others[0] if others else (key_fields[0] if key_fields else None)

    facts = {
        "columns": fields,
        "score_column": score_f, "entity_column": entity_f, "group_column": group_f, "measure_column": metric_f,
        "focus": focus or None,
        "verified_rows": len(rows),
        "papers": sorted({str(r.get("paper") or r.get("file") or "") for r in rows} - {""}),
        "distinct_values": {k: len({str(r.get(k)) for r in rows if r.get(k) not in (None, "")}) for k in key_fields},
    }
    if not metric_f:
        facts["better_is"] = "lower" if (score_f and lower_is_better(rows, score_f)) else "higher"
    if not (score_f and entity_f):
        return facts

    scored = [(r, to_number(r.get(score_f))) for r in rows]
    scored = [(r, v) for r, v in scored if v is not None]

    groups = defaultdict(list)
    for r, v in scored:
        key = (str(r.get(metric_f)) if metric_f else None, str(r.get(group_f)) if group_f else "all")
        groups[key].append((r, v))
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:MAX_GROUPS]

    leaders = defaultdict(int)
    per_group = []
    for (metric, g), items in ranked:
        direction = direction_for(metric, rows, score_f) if metric_f else facts["better_is"]
        entry = {}
        if metric_f:
            entry[metric_f] = metric
        if group_f:
            entry[group_f] = g
        entry.update({"results": len(items), "better_is": direction,
                      "range": f"{_fmt(min(v for _, v in items))} to {_fmt(max(v for _, v in items))}"})
        if direction != "none" and len(items) > 1:
            keyf = {"higher": lambda t: -t[1], "lower": lambda t: t[1], "closer to 0": lambda t: abs(t[1])}[direction]
            ordered = sorted(items, key=keyf)
            best_r, worst_r = ordered[0][0], ordered[-1][0]
            entry["best"] = {entity_f: best_r.get(entity_f), score_f: best_r.get(score_f)}
            entry["worst"] = {entity_f: worst_r.get(entity_f), score_f: worst_r.get(score_f)}
            leaders[str(best_r.get(entity_f))] += 1
        per_group.append(entry)
    facts["by_group"] = per_group
    facts["groups_shown"] = f"{len(per_group)} of {len(groups)}"
    facts["most_often_best"] = [{entity_f: k, "times_best": n}
                                for k, n in sorted(leaders.items(), key=lambda kv: -kv[1])[:5]]
    facts["cross_paper_conflicts"] = [
        {"result": c["label"], "values_by_paper": c["values_by_paper"]}
        for c in detect_cross_paper_conflicts(rows, fields)[:5]]
    return facts


MEASURE_RE = re.compile(r"((?:(?<![\w.])[-−‒–])?\d+(?:[.,]\d+)*|\.\d+)\s*(?:%|pp\b|percentage points?\b|points?\b)",
                        re.I)


def unsupported_numbers(text: str, facts: dict, rows: list[dict]) -> list[str]:
    """Numbers in the summary that the data doesn't support (should be none).

    - A number used as a measurement ("6 points", "12.4%", "3 pp") must equal an actual score in
      the rows; a computed difference like "6 points behind" is not in the data, so it's flagged.
    - Any other number must appear in the facts or rows (counts, names like "Opus 4").
    Signs and formats are read the same way as everywhere else ("−0.150" == "-0.15")."""
    score_f = facts.get("score_column")
    scores = {_fmt(v) for v in (to_number(r.get(score_f)) for r in rows) if v is not None} if score_f else set()
    scores |= {s.lstrip("-") for s in scores}   # "0.15 lower" may be written without its sign
    allowed = set()
    for blob in (json.dumps(facts, default=str, ensure_ascii=False), json.dumps(rows, default=str, ensure_ascii=False)):
        nums = signed_numbers(blob)
        allowed |= set(nums) | {n.lstrip("-") for n in nums}

    bad, measured = [], []
    for m in MEASURE_RE.finditer(text):
        measured.append(m.span(1))
        n = signed_numbers(m.group(1))
        if n and _fmt(float(n[0])) not in scores and n[0] not in scores:
            bad.append(m.group(1))
    for m in re.finditer(r"(?:(?<![\w.])[-−‒–])?\d+(?:[.,]\d+)*|\.\d+", text):
        if any(s <= m.start() < e for s, e in measured):
            continue
        n = signed_numbers(m.group(0))
        if n and n[0] not in allowed:
            bad.append(m.group(0))
    seen, out = set(), []
    for n in bad:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def summarize(rows: list[dict], fields: list[str], excluded: int, focus: str | None = None) -> dict:
    """Return {"text", "facts", "unsupported", "rows_used", "mode", "examples"}.

    The focus can be a column name ('Parameter' -> grouped by that column) or a value
    ('mu1', 'Bias', 'Claude', 'Table 3' -> only matching rows). "text" is "" if nothing matches,
    and "examples" then lists values the user could focus on instead."""
    col = focus_column(fields, focus)
    if col:
        mode, used, label = "column", rows, None
    else:
        mode, used, label = ("value" if focus else "all"), filter_focus(rows, fields, focus), focus or None
    if not used:
        return {"text": "", "facts": {"focus": focus}, "unsupported": [], "rows_used": 0, "mode": mode,
                "examples": focus_examples(rows, fields)}
    facts = build_facts(used, fields, label, group_by=col)
    if col:
        facts["focus"] = f"grouped by the {col} column"
    facts["excluded_unverified_rows"] = excluded
    text = complete(SYSTEM, f"FACTS:\n{json.dumps(facts, indent=1, default=str, ensure_ascii=False)}"
                            f"\n\nWrite the takeaway now.").strip()
    return {"text": text, "facts": facts, "unsupported": unsupported_numbers(text, facts, used),
            "rows_used": len(used), "mode": mode, "examples": {}}
