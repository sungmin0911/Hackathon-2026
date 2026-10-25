"""Measure extraction accuracy against a hand-checked answer key.

1. Run the extractor:
     python cli.py papers/ --fields "Model,Benchmark,Score" --out out/results.csv
2. Score it:
     python eval/score.py out/results.csv eval/gold.csv

Only papers that appear in gold.csv are scored. A gold row counts as "correct" if the
extractor produced a row for the same paper with a matching model + benchmark and the same number.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):  # Windows consoles (e.g. Korean cp949) can't print emoji
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import re
import sys

import pandas as pd


def norm(s) -> str:
    s = re.sub(r"\(.*?\)", " ", str(s or "").lower())  # drop "(EM)", "(Pass@1)"
    s = re.sub(r"^(openai|anthropic|google)\s+", "", s.strip())
    return re.sub(r"[^a-z0-9.]+", " ", s).strip()


def number(s) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", str(s or "").replace(",", ""))
    return float(m.group(0)) if m else None


def same_benchmark(a, b) -> bool:
    # strict: "GPQA Diamond" == "GPQA-Diamond (Pass@1)", but "MMLU" != "MMLU-Pro"
    return bool(norm(a)) and norm(a).replace(" ", "") == norm(b).replace(" ", "")


def same_model(a, b) -> bool:
    # equal, or equal except a trailing date/version code ("Claude 3.5 Sonnet" ~ "Claude-3.5-Sonnet-1022")
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    rest = long_[len(short):].strip()
    return long_.startswith(short + " ") and bool(re.fullmatch(r"\d{3,4}", rest))


def main(results_csv: str, gold_csv: str) -> None:
    res = pd.read_csv(results_csv)
    gold = pd.read_csv(gold_csv)
    res = res[res["file"].isin(gold["file"])]

    outcomes = []
    for _, g in gold.iterrows():
        cands = res[(res["file"] == g["file"])
                    & res["Model"].map(lambda m: same_model(m, g["Model"]))
                    & res["Benchmark"].map(lambda b: same_benchmark(b, g["Benchmark"]))]
        if cands.empty:
            outcomes.append(("missing", None))
            continue
        hit = cands[cands["Score"].map(number) == number(g["Score"])]
        if not hit.empty:
            order = hit["status"].map({"verified": 0, "mismatch": 1, "unverified": 2}).fillna(3)
            outcomes.append(("correct", hit.loc[order.idxmin(), "status"]))
        else:
            outcomes.append(("wrong_value", cands.iloc[0]["status"]))

    gold["outcome"] = [o for o, _ in outcomes]
    gold["status"] = [s for _, s in outcomes]
    n = len(gold)
    c = (gold["outcome"] == "correct").sum()
    w = (gold["outcome"] == "wrong_value").sum()
    m = (gold["outcome"] == "missing").sum()

    print(f"Gold rows: {n}")
    print(f"  correct      {c:3d}  ({c / n:.0%})")
    print(f"  wrong value  {w:3d}  ({w / n:.0%})")
    print(f"  missing      {m:3d}  ({m / n:.0%})")

    found = gold[gold["outcome"] != "missing"]
    if not found.empty:
        print("\nDoes the ✅ status mean the value is right?")
        print(pd.crosstab(found["status"], found["outcome"]).to_string())

    bad = gold[gold["outcome"] != "correct"]
    if not bad.empty:
        print("\nRows to look at:")
        print(bad[["file", "Model", "Benchmark", "Score", "outcome", "status"]].to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python eval/score.py out/results.csv eval/gold.csv")
    main(sys.argv[1], sys.argv[2])
