# Notes for extraction (A) and verification (B)

Findings from inspecting the real test papers in `papers/`.

## 1. Tables come out of the PDF as one number per line

Example, `kimi-k2.pdf` page 16, as PyMuPDF extracts it:

```
MMLU (EM)
89.5
89.4
87.0
91.5
92.9
90.4
90.1
```

The model names are in the table header, many lines above. So:

- **A:** the LLM has to work out which number belongs to which column by counting. This is where
  wrong-model errors will come from. Test it hard on the tables in `eval/gold.csv`.
- **B:** a quote like `"MMLU (EM) 89.5 89.4 87.0 91.5"` will match the page (good), but it only proves
  the number is in the MMLU row, **not that it belongs to the right model**. Be honest about this in
  the UI and pitch: ✅ means "number found in the paper," not "attribution guaranteed."

## 2. Structured table extraction exists, but the text is mangled

`page.find_tables()` (PyMuPDF) finds the table and keeps columns, but text gets scrambled:

```
['Benchmark', 'Kimi -K2\n-\nInstruct', ..., 'Claude\nSonnet4', 'Claude\nOpus4', ...]
['MMLU(EM)', '895\n.', '894\n.', ...]      <- 89.5 became "895\n."
```

Possible improvement for A (only if time): use `find_tables()` to get column structure, clean the
cells, and send the LLM a tidy markdown table. Or send the page **image** to a vision-capable model
for table pages.

## 3. Papers are long

`minimax-01.pdf` is 68 pages and `deepseek-v2.pdf` is 52. Full text may hit cost/time limits. Results
tables are usually in an "Evaluation" / "Experiments" section. That's a use for C's vector search,
or a simple filter: only send pages containing digits + benchmark words.

## 4. Same competitor, different numbers is often legitimate

e.g. MMLU-Pro for **DeepSeek-V3** is 75.9 in the R1 report, while Kimi-K2 reports 81.2 for
**DeepSeek-V3-0324**, a *different version*. Cross-paper conflict flags (B) should:
- compare exact model names (including version/date suffixes), and
- be worded as "reported differently, check settings," not "error."

## 5. Measuring accuracy

`eval/gold.csv` has 38 hand-checked rows (DeepSeek-R1 p.13, Kimi-K2 p.16; MMLU, MMLU-Pro, GPQA Diamond).

```bash
python cli.py papers/ --fields "Model,Benchmark,Score" --out out/results.csv
python eval/score.py out/results.csv eval/gold.csv
```

It reports correct / wrong value / missing, and whether ✅ actually means correct.
Re-run after every prompt change. Add more gold rows if you can.

## 6. Merge review (after A/B update)

Fixed while merging, each with a regression test in `tests/test_consistency.py`:

1. **JSON parser dropped whole papers** when a quote contained citation brackets like `[24]`
   (non-greedy `\[.*?\]` cut the array at the first `]`). It now uses `json.JSONDecoder.raw_decode`.
2. **Detector only worked with fields named exactly `Model` and `Dataset`.** Our demo uses
   `Model, Benchmark, Score`, so it never fired. It now treats the last numeric-looking column as the
   score and all other columns as identifiers (works for `Drug, Dose, Outcome` too).
3. **Score picked randomly** from a set of numbers ("88.3 (5-shot)" → 88.3 or 5 depending on the run),
   which caused random false conflicts. It now uses the first number in order.
4. Intra-paper check now groups by paper, so it's safe on combined rows too.

Measured: `select_relevant_pages` only kicks in above 180k chars (only `minimax-01`: 68 → 50 pages).
On the other reports it would keep ~90% of pages, since "results/performance/benchmark" appear everywhere.
