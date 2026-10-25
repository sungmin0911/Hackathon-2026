# Manual baseline + demo video

## Manual baseline (15 min, do this BEFORE seeing the tool's output)

This is the "before" number for the pitch. Use a stopwatch.

1. Open `papers/deepseek-r1.pdf`, `kimi-k2.pdf`, `qwen3.pdf`, `minimax-01.pdf`, `kimi-k1.5.pdf`.
2. For each paper, find every model's **MMLU** score and type `Model, Benchmark, Score` into a spreadsheet.
3. Stop the timer when all 5 are done.

| Papers | Rows typed | Time | Copy errors found later |
|---|---|---|---|
| 5 | | | |

Pitch line: *"5 papers took me [X] minutes, about [X × 8] for a 40-paper review."*
Only claim what you measured. Say "about" when extrapolating.

## Accuracy check

```bash
python cli.py papers/ --fields "Model,Benchmark,Score" --out out/results.csv
python eval/score.py out/results.csv eval/gold.csv
```

| Run | Prompt change | Correct | Wrong value | Missing | ✅ but wrong |
|---|---|---|---|---|---|
| 1 | baseline | | | | |
| 2 | | | | | |

The last column matters most: it's how often the tool says ✅ when the value is wrong.

## Demo video (2 min max, record Sunday ~11 AM after the feature freeze)

| # | Shot | ~Sec |
|---|---|---|
| 1 | Title card + one-line problem | 10 |
| 2 | Scrolling a long PDF table: "imagine doing this for 40 papers" | 10 |
| 3 | Upload 7 papers, type `Model, Benchmark, Score`, press Extract | 15 |
| 4 | Results table with ✅/⚠️/❌ counts | 15 |
| 5 | Click ✅ row → highlighted source page | 15 |
| 6 | Click ⚠️/❌ row → explain why it's flagged | 15 |
| 7 | Leaderboard by benchmark | 15 |
| 8 | Switch to non-CS papers with different fields (if ready) | 15 |
| 9 | Accuracy number + "built with Tiger Data" + close | 10 |

Tools: screen recording (macOS: Cmd+Shift+5; Windows: Win+G). Upload unlisted to YouTube and link it on Devpost.
