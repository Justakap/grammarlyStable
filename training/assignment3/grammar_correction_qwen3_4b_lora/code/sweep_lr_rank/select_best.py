import os
import re
import sys

import pandas as pd

RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
SETS = ["val_indictalk", "val_a2"]
TIE = float(os.getenv("TIE", "1.0"))

path = os.path.join(RESULTS_DIR, "all_results.csv")
if not os.path.exists(path):
    sys.exit(f"No results yet: {path} not found")

df = pd.read_csv(path)
df = df[(df["system"] == "model") & (df["test_set"].isin(SETS))]
df = df.drop_duplicates(subset=["run", "test_set"], keep="last")

rows = []
for run, part in df.groupby("run"):
    if set(part["test_set"]) != set(SETS):
        continue
    by = part.set_index("test_set")
    m = re.match(r"^lr([0-9.eE+-]+)_r(\d+)$", str(run))
    rows.append({
        "run": run,
        "lr": m.group(1) if m else "",
        "rank": int(m.group(2)) if m else 0,
        "f05_indictalk": by.loc["val_indictalk", "edit_f0_5"],
        "f05_a2": by.loc["val_a2", "edit_f0_5"],
        "mean_f05": (by.loc["val_indictalk", "edit_f0_5"] + by.loc["val_a2", "edit_f0_5"]) / 2,
        "mean_over_correction": (by.loc["val_indictalk", "over_correction_rate"]
                                 + by.loc["val_a2", "over_correction_rate"]) / 2,
        "mean_gleu": (by.loc["val_indictalk", "gleu"] + by.loc["val_a2", "gleu"]) / 2,
        "mean_exact_match": (by.loc["val_indictalk", "exact_match"] + by.loc["val_a2", "exact_match"]) / 2,
    })

if not rows:
    sys.exit("No run has been evaluated on both validation sets yet")

table = pd.DataFrame(rows).sort_values("mean_f05", ascending=False).reset_index(drop=True)
best = table["mean_f05"].max()
tied = table[table["mean_f05"] >= best - TIE].sort_values(["mean_over_correction", "rank"])
winner = tied.iloc[0]

cols = ["run", "lr", "rank", "f05_indictalk", "f05_a2", "mean_f05", "mean_over_correction", "mean_gleu",
        "mean_exact_match"]
lines = ["# Learning rate and LoRA rank sweep (validation sets)", "",
         "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
for _, r in table.iterrows():
    lines.append("| " + " | ".join(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
lines += ["",
          f"Rule: highest mean edit F0.5 over val_indictalk and val_a2; runs within {TIE} point of the best count as "
          "tied, and among those the lowest over-correction wins, then the smaller rank.", ""]
if len(tied) > 1:
    lines += [f"Tied with the best: {', '.join(tied['run'])}.", ""]
lines += [f"**Selected: {winner['run']}** (learning rate {winner['lr']}, rank {winner['rank']}, "
          f"mean F0.5 {winner['mean_f05']:.2f}, over-correction {winner['mean_over_correction']:.2f}%).", "",
          "Train it on the full Data Mix D and score it on the test sets with:", "",
          f"    bash submit_final.sh {winner['lr']} {winner['rank']}", ""]

out = os.path.join(RESULTS_DIR, "selection.md")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
print(f"Saved {out}")
