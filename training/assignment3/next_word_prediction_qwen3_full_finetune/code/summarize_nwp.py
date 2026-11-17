import os
import sys

import pandas as pd

RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
path = os.path.join(RESULTS_DIR, "all_results.csv")
if not os.path.exists(path):
    sys.exit(f"No results yet: {path} not found")

df = pd.read_csv(path).drop_duplicates(subset=["run", "persona_mode", "test_set"], keep="last")
cols = ["run", "persona_mode", "perplexity", "top1", "top3", "word_top1", "word_top3", "word_mrr",
        "keystroke_savings", "tokens", "words"]
lines = ["# Next-word prediction results", "",
         "Token-level: perplexity and next-token top-1/top-3 accuracy (%). Word-level: the next whole word is among "
         "the model's suggestions (%), mean reciprocal rank, and the share of characters saved by accepting a correct "
         "suggestion at the start of each word (%).", ""]
for test_set, part in df.groupby("test_set", sort=False):
    part = part.sort_values(["run", "persona_mode"])
    lines += [f"## {test_set}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in part.iterrows():
        cells = []
        for c in cols:
            v = row.get(c)
            cells.append("" if pd.isna(v) else (f"{v:.3f}" if c == "word_mrr" else
                                               f"{v:.2f}" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

out = os.path.join(RESULTS_DIR, "results_table.md")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
print(f"Saved {out}")
