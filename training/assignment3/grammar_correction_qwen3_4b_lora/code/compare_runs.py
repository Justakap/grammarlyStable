import os
import sys
import difflib
import itertools

import numpy as np
import pandas as pd

RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
RUNS = os.getenv("RUNS", "D G1 G2").split()
TEST_SETS = os.getenv("TEST_SETS", "test_indictalk test_a2").split()
N_BOOT = int(os.getenv("N_BOOT", "5000"))
SEED = int(os.getenv("SEED", "42"))


def ws(text):
    return " ".join(str(text).split())


def edit_set(src_tokens, tgt_tokens):
    sm = difflib.SequenceMatcher(a=src_tokens, b=tgt_tokens, autojunk=False)
    return {(i1, i2, tuple(tgt_tokens[j1:j2])) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"}


def per_sentence(df):
    src = [ws(x) for x in df["incorrect_text"]]
    hyp = [ws(x) for x in df["prediction"]]
    ref = [ws(x) for x in df["correct_text"]]
    tp, fp, fn = [], [], []
    for s, h, r in zip(src, hyp, ref):
        s_t = s.split()
        h_e, r_e = edit_set(s_t, h.split()), edit_set(s_t, r.split())
        tp.append(len(h_e & r_e))
        fp.append(len(h_e - r_e))
        fn.append(len(r_e - h_e))
    clean = np.array([s == r for s, r in zip(src, ref)])
    changed = np.array([h != s for s, h in zip(src, hyp)])
    return np.array(tp), np.array(fp), np.array(fn), clean, changed


def f05(tp, fp, fn):
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)
    r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 1.0)
    d = 0.25 * p + r
    return 100 * np.where(d > 0, 1.25 * p * r / np.where(d > 0, d, 1), 0.0)


def load(run, test):
    path = os.path.join(RESULTS_DIR, run, f"predictions_{test}.csv")
    if not os.path.exists(path):
        sys.exit(f"{path} not found")
    return pd.read_csv(path, keep_default_na=False)


rng = np.random.default_rng(SEED)
point, boot_f, boot_oc = {}, {}, {}
lines = [f"# Paired bootstrap comparison ({N_BOOT:,} resamples of the test sentences)", ""]

for test in TEST_SETS:
    frames = {run: load(run, test) for run in RUNS}
    ref = frames[RUNS[0]]
    for run, df in frames.items():
        if len(df) != len(ref) or not (df["incorrect_text"].values == ref["incorrect_text"].values).all():
            sys.exit(f"{run} and {RUNS[0]} were not scored on the same {test} rows")
    n = len(ref)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    for run, df in frames.items():
        tp, fp, fn, clean, changed = per_sentence(df)
        point[(run, test)] = (float(f05(tp.sum(), fp.sum(), fn.sum())),
                              100 * changed[clean].mean(), int(changed[clean].sum()), int(clean.sum()))
        boot_f[(run, test)] = f05(tp[idx].sum(1), fp[idx].sum(1), fn[idx].sum(1))
        c = clean[idx]
        boot_oc[(run, test)] = 100 * (changed[idx] & c).sum(1) / np.maximum(c.sum(1), 1)

for run in RUNS:
    boot_f[(run, "mean")] = np.mean([boot_f[(run, t)] for t in TEST_SETS], axis=0)
    point[(run, "mean")] = (float(np.mean([point[(run, t)][0] for t in TEST_SETS])), None, None, None)


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return f"{lo:.2f} to {hi:.2f}"


def p_value(d):
    return min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))


sets = TEST_SETS + ["mean"]
lines += ["## Edit F0.5 with 95% confidence interval", "",
          "| run | " + " | ".join(sets) + " |", "|" + "---|" * (len(sets) + 1)]
for run in RUNS:
    lines.append(f"| {run} | " + " | ".join(f"{point[(run, s)][0]:.2f} ({ci(boot_f[(run, s)])})" for s in sets) + " |")

lines += ["", "## Differences in edit F0.5 (first run minus second)", "",
          "| pair | set | difference | 95% interval | p-value | verdict |", "|---|---|---|---|---|---|"]
for a, b in itertools.combinations(RUNS, 2):
    for s in sets:
        d = boot_f[(a, s)] - boot_f[(b, s)]
        diff = point[(a, s)][0] - point[(b, s)][0]
        p = p_value(d)
        verdict = "real difference" if p < 0.05 else "within noise"
        lines.append(f"| {a} vs {b} | {s} | {diff:+.2f} | {ci(d)} | {p:.3f} | {verdict} |")

lines += ["", "## Over-correction (already-correct inputs the model changed)", "",
          "| pair | set | first run | second run | p-value | verdict |", "|---|---|---|---|---|---|"]
for a, b in itertools.combinations(RUNS, 2):
    for s in TEST_SETS:
        pa, pb = point[(a, s)], point[(b, s)]
        d = boot_oc[(a, s)] - boot_oc[(b, s)]
        p = p_value(d) if (pa[2] != pb[2]) else 1.0
        verdict = "real difference" if p < 0.05 else "within noise"
        lines.append(f"| {a} vs {b} | {s} | {pa[1]:.2f}% ({pa[2]}/{pa[3]}) | {pb[1]:.2f}% ({pb[2]}/{pb[3]}) | "
                     f"{p:.3f} | {verdict} |")

lines += ["", "p < 0.05 means the difference held in at least 97.5% of resamples in the same direction. "
          "The point scores should match results_table.md."]
out = os.path.join(RESULTS_DIR, "significance_" + "_".join(RUNS) + ".md")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nSaved {out}")
