"""Run the extractor on a folder of PDFs.

Usage:
  python cli.py papers/ --fields "Model,Dataset,Accuracy"
"""
import sys
if hasattr(sys.stdout, "reconfigure"):  # Windows consoles (e.g. Korean cp949) can't print emoji
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from app import db
from app.consistency import detect_cross_paper_conflicts, detect_intra_paper_inconsistencies
from app.pipeline import process_paper

ICON = {"verified": "✅", "mismatch": "⚠️", "unverified": "❌"}


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--fields", required=True, help='Comma-separated, e.g. "Model,Dataset,Accuracy"')
    ap.add_argument("--out", default="out/results.csv")
    ap.add_argument("--no-cache", action="store_true", help="Re-extract even if Tiger Cloud has saved results")
    ap.add_argument("--check-db", action="store_true", help="Only test the Tiger Cloud connection and exit")
    args = ap.parse_args()

    if args.check_db:
        c = db.check()
        print(f"Tiger Cloud: connected. {c['papers']} papers, {c['rows']} results stored." if c["ok"]
              else f"Tiger Cloud: NOT connected ({c['error']})")
        return

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    pdfs = sorted(Path(args.folder).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {args.folder}")

    if db.enabled():
        c = db.check()
        print(f"Tiger Cloud: connected ({c['papers']} papers saved). Cache {'off' if args.no_cache else 'on'}."
              if c["ok"] else f"Tiger Cloud: NOT connected, continuing without it ({c['error']})")
    all_rows, start = [], time.time()
    for pdf in pdfs:
        print(f"Reading {pdf.name} ...")
        try:
            rows = process_paper(pdf, fields, use_cache=not args.no_cache,
                                 on_progress=lambda d, n, pg: print(f"  ... {d}/{n} parts done ({pg})", flush=True))
        except Exception as e:  # keep going if one paper fails
            print(f"  failed: {e}")
            continue
        if rows and rows[0].get("cached"):
            print(f"  (loaded {len(rows)} saved results from Tiger Cloud, no AI call)")
        for r in rows:
            vals = " | ".join(f"{f}: {r.get(f)}" for f in fields)
            extra = f" | {r.get('inconsistency')}" if r.get("inconsistency") else ""
            print(f"  {ICON.get(r['status'], '?')} {vals} | p.{r['found_page']}{extra}")
        all_rows.extend(rows)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).drop(columns=["path", "cached"], errors="ignore").to_csv(args.out, index=False)
    counts = pd.Series([r["status"] for r in all_rows]).value_counts().to_dict()
    print(f"\n{len(all_rows)} rows from {len(pdfs)} papers in {time.time() - start:.0f}s -> {args.out}")
    print(f"Status: {counts}")

    # Consistency summary
    intra = detect_intra_paper_inconsistencies(all_rows, fields)
    cross = detect_cross_paper_conflicts(all_rows, fields)
    if intra or cross:
        print(f"\nInconsistencies: {len(intra)} intra-paper, {len(cross)} cross-paper")
        for c in intra[:5]:
            print(f"  intra  {c['label']}: {c['values']}")
        for c in cross[:5]:
            print(f"  cross  {c['label']}: {c['values_by_paper']}")


if __name__ == "__main__":
    main()
