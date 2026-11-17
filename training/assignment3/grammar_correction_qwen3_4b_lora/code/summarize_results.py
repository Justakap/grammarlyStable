import os
import sys

import pandas as pd

RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
path = os.path.join(RESULTS_DIR, "all_results.csv")
if not os.path.exists(path):
    sys.exit(f"No results yet: {path} not found")

df = pd.read_csv(path)
df = df.drop_duplicates(subset=["run", "system", "test_set"], keep="last")
df = df[(df["system"] == "model") | ~df.duplicated(subset=["system", "test_set"], keep="first")]
df.loc[df["system"] == "copy_input", "run"] = "copy input"

cols = ["run", "n", "exact_match", "over_correction_rate", "chrf", "chrf_pp", "gleu", "edit_f0_5", "bleu"]
if "similarity_prediction_vs_reference" in df.columns and df["similarity_prediction_vs_reference"].notna().any():
    cols.append("similarity_prediction_vs_reference")

lines = ["# Grammar correction results", ""]
for test_set, part in df.groupby("test_set", sort=False):
    part = part.sort_values("run")
    lines += [f"## {test_set}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in part.iterrows():
        cells = []
        for c in cols:
            v = row.get(c)
            cells.append("" if pd.isna(v) else (f"{v:.2f}" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

out = os.path.join(RESULTS_DIR, "results_table.md")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
print(f"Saved {out}")
