# 2-minute pitch

Fill every `[ ]` with **real numbers from your own runs**. Don't present numbers you didn't measure.

---

**0:00 Problem (20s)**
> Before you can do research, you have to know what's already been done. That means opening dozens
> of papers, finding the results tables, and copying numbers into a spreadsheet by hand. It's slow,
> and one copy mistake breaks your whole comparison.
>
> I timed it: copying model, benchmark, and score from [5] papers took me [ ] minutes. That's about
> [ ] hours for a 40-paper literature review.

**0:20 Live demo: extract (35s)**
> Here are [7] public AI technical reports: DeepSeek, Qwen, Kimi, MiniMax. I type the columns I want:
> *Model, Benchmark, Score*.
*(Press Extract; while it runs:)*
> It works in any field: you define the columns, and it reads the papers.
*(Results appear.)*
> [ ] results from [7] papers in [ ] seconds.

**0:55 Live demo: verification, the key part (35s)**
> AI tools make up numbers. So every value comes with an exact quote, and plain code, not AI, checks
> that quote really exists in the paper.
*(Click a ✅ row: highlighted PDF page appears.)*
> Here's the source, highlighted.
*(Click a ⚠️ or ❌ row.)*
> And here the AI's number didn't match the paper, so it's flagged, not trusted.
>
> We checked [38] values by hand: [ ]% were correct, and [the ✅ status was right [ ]% of the time].

**1:30 Leaderboard (15s)**
*(Leaderboard tab, Separate by: Benchmark.)*
> Because the data is verified, we can rank models across papers. Only ✅ rows count.
> *(If you found one:)* And notice: two reports give different scores for [model] on [benchmark].
> That's something you'd never catch reading papers one at a time.

**1:45 Close (15s)**
> Built on Tiger Data for storage and search. Papers in, verified comparison table out: days of
> manual work in minutes, and every number traceable to its source.

---

## Likely judge questions

| Question | Answer |
|---|---|
| How is this different from Elicit / ChatGPT? | Code-level verification of every value against the source, cross-paper ranking, and conflict flags. ChatGPT gives you numbers with no proof. |
| What does ✅ actually guarantee? | That the quoted text and the number exist in the paper. It doesn't fully guarantee the number is attributed to the right model column; that's why we show the source page in one click. *(Honest answers win trust.)* |
| What's your accuracy? | Give the `eval/score.py` numbers and how many rows you checked. |
| What about scanned PDFs? | Not supported yet (needs OCR). Most modern papers are text PDFs. |
| How does Tiger Data help? | *(Fill in after C is done: e.g. vector search picks only the relevant pages of a 68-page report.)* |
| Who would use this? | Grad students doing literature reviews, meta-analysis researchers, anyone writing a related-work section. |

## Before demo time

- [ ] Run the demo end to end 3 times, timing it under 2 minutes
- [ ] **Pre-run the extraction and keep the results**: if venue Wi-Fi or the API is slow, show saved results instead of waiting
- [ ] Pick the exact ✅ and ⚠️ rows you'll click in advance
- [ ] Laptop charged, browser zoom up so judges can read the table
