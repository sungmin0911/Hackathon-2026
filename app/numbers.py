"""One place that reads numbers out of text, used by verification, conflicts, leaderboard and summary.

Handles the formats research papers actually use:
  - signs, including the typographic minus "−" (U+2212) and en dash "–" used as a minus
  - no leading zero (".003"), thousands separators ("2,431"), percent signs ("6.43%")
  - standard errors in parentheses ("(0.012)"), significance stars ("0.083***"), "<0.001"
A dash is only a minus sign when it isn't glued to a word or number before it, so
"DeepSeek-R1", "MMLU-Pro", ranges like "10–20" and "1.9e−2" don't produce negative numbers.
"""
import re
from decimal import Decimal, InvalidOperation

MINUS_CHARS = "-−‒–"
_SIGNED = re.compile(r"(?:(?<![\w.])([" + MINUS_CHARS + r"]))?(\d+(?:[.,]\d+)*|\.\d+)")
_LOOKS_NUMERIC = re.compile(r"^\s*[<>≤≥~≈±+(\[]?\s*[" + MINUS_CHARS + r"]?\s*\.?\d")


def _canonical(sign: str, digits: str) -> str:
    """'−', '.50' -> '-0.5'; '', '2,431' -> '2431'. Same value, same string."""
    d = digits.replace(",", "")
    if d.startswith("."):
        d = "0" + d
    try:
        d = format(Decimal(d).normalize(), "f")
    except InvalidOperation:
        pass
    return ("-" + d) if sign and d.strip("0.") else d


def signed_numbers(text) -> list[str]:
    """All numbers in order, as canonical strings with their sign: '−0.021*** (0.010)' -> ['-0.021', '0.01']."""
    return [_canonical(sign, digits) for sign, digits in _SIGNED.findall(str(text or ""))]


def first_number(text) -> float | None:
    """The first number as a float, sign included, or None."""
    nums = signed_numbers(text)
    return float(nums[0]) if nums else None


def looks_numeric(value) -> bool:
    """True for values that ARE numbers ('79.8', '−0.021', '(0.012)', '.003', '<0.001'),
    False for names that merely contain digits ('DeepSeek-R1', 'AIME 2024', 'GPT-4')."""
    return bool(_LOOKS_NUMERIC.match(str(value or "")))
