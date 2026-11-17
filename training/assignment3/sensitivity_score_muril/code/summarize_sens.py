import os
import json
import glob

RESULTS_DIR = os.getenv("RESULTS_DIR", "results")


def fmt(v, pct=False):
    if v is None:
        return "–"
    return f"{100 * v:.2f}" if pct else (f"{v:.2f}" if isinstance(v, float) else str(v))


runs = []
for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*", "metrics.json"))):
    runs.append(json.load(open(path, encoding="utf-8")))

lines = ["# Sensitivity score results", "",
         "Score = 100 x calibrated sigmoid. Bands: under 30 safe, 30-70 sensitive, 70 and over offensive.", "",
         "## TRAC-1 Hindi test (Romanized Facebook comments, official dev set)", "",
         "| run | n | AUROC NAG vs OAG | AUROC NAG vs aggressive | mean NAG / CAG / OAG | band acc | band macro-F1 | OAG F1 @50 | ECE before / after |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in runs:
    t = r["tests"].get("test_trac")
    if not t:
        continue
    mb = t["mean_score_by_label"]
    lines.append(f"| {r['run']} | {t['n']} | {fmt(t['auroc_NAG_vs_OAG'], True)} | {fmt(t['auroc_NAG_vs_aggressive'], True)} | "
                 f"{mb['NAG']} / {mb['CAG']} / {mb['OAG']} | {t['band_accuracy']} | {t['band_macro_f1']} | "
                 f"{t['OAG_at_50']['f1']} | {fmt(t['ece_NAG_vs_OAG_before'], True)} / {fmt(t['ece_NAG_vs_OAG_after'], True)} |")

if any("test_kaggle" in r["tests"] for r in runs):
    lines += ["", "## Kaggle Hinglish hate-speech test", "",
              "| run | n | AUROC | precision @50 | recall @50 | F1 @50 | ECE before / after |", "|---|---|---|---|---|---|---|"]
    for r in runs:
        t = r["tests"].get("test_kaggle")
        if t:
            lines.append(f"| {r['run']} | {t['n']} | {fmt(t['auroc'], True)} | {t['at_50']['precision']} | "
                         f"{t['at_50']['recall']} | {t['at_50']['f1']} | {fmt(t['ece_before'], True)} / {fmt(t['ece_after'], True)} |")

lines += ["", "## Everyday chat (false alarms) and profanity check", "",
          "| run | test | n | mean score | share >= 30 (%) | share >= 70 (%) |", "|---|---|---|---|---|---|"]
for r in runs:
    for name in ("test_safe_indictalk", "test_safe_a2", "test_a2_lexicon"):
        t = r["tests"].get(name)
        if t:
            lines.append(f"| {r['run']} | {name} | {t['n']} | {t['mean_score']} | {t['share_ge_30']} | {t['share_ge_70']} |")
lines += ["", "For the safe-chat sets lower is better (false alarms); for test_a2_lexicon (messages with a strong "
          "profanity word) higher is better.", ""]

lines += ["## Sample messages", "", "| run | score | band | text |", "|---|---|---|---|"]
for r in runs:
    for s in r.get("sample_texts", []):
        lines.append(f"| {r['run']} | {s['score']} | {s['band']} | {s['text']} |")

path = os.path.join(RESULTS_DIR, "results_table.md")
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nWrote {path}")
