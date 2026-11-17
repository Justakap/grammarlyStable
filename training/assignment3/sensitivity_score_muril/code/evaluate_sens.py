import os
import sys
import csv
import json
import glob
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sens_common import sigmoid, auroc, ece, prf, macro_f1, band

RUN_NAME = os.getenv("RUN_NAME", "muril")
MODEL_PATH = os.getenv("MODEL_PATH", f"runs/run_{RUN_NAME}/best")
DATA_DIR = os.getenv("DATA_DIR", "data_sens")
TEST_FILES = [f for f in os.getenv("TEST_FILES", "").split(",") if f] or sorted(glob.glob(os.path.join(DATA_DIR, "test_*.jsonl")))
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
EVAL_BATCH = int(os.getenv("EVAL_BATCH_SIZE", "64"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
N_EXAMPLES = int(os.getenv("N_EXAMPLES", "15"))
OUT_DIR = os.path.join(RESULTS_DIR, RUN_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(OUT_DIR, "evaluate.log"))])
logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEVICE == "cuda" and torch.cuda.is_bf16_supported()

SAMPLE_TEXTS = [
    "kal milte hai yaar, movie ka plan pakka",
    "bhai tu bahut accha kaam kar raha hai, keep it up",
    "ye government kuch nahi karti, sab chor hai",
    "tu pagal hai kya, dimag kharab hai tera",
    "chup kar bewakoof, teri aukaat kya hai",
    "in logo ko desh se bahar nikal do, ye sab gaddar hai",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE).eval()
model.config.pad_token_id = tokenizer.pad_token_id
calib_path = os.path.join(MODEL_PATH, "calibration.json")
TEMPERATURE = json.load(open(calib_path))["temperature"] if os.path.exists(calib_path) else 1.0
logger.info(f"Run {RUN_NAME} | model {MODEL_PATH} | temperature {TEMPERATURE} | tests {TEST_FILES}")


def logits_for(texts):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [0.0] * len(texts)
    with torch.no_grad():
        for s in range(0, len(order), EVAL_BATCH):
            idx = order[s:s + EVAL_BATCH]
            enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True, max_length=MAX_LENGTH,
                            return_tensors="pt").to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=AMP):
                z = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            for i, v in zip(idx, z.float().view(-1).tolist()):
                out[i] = v
    return out


def scores_from(logits, temperature):
    return [100.0 * sigmoid(z / temperature) for z in logits]


def metrics_for(name, rows, logits):
    raw = scores_from(logits, 1.0)
    cal = scores_from(logits, TEMPERATURE)
    m = {"test": name, "n": len(rows), "mean_score": round(sum(cal) / len(cal), 2),
         "share_ge_30": round(100 * sum(s >= 30 for s in cal) / len(cal), 2),
         "share_ge_70": round(100 * sum(s >= 70 for s in cal) / len(cal), 2)}
    labels = [r["label"] for r in rows]
    if "trac" in name:
        m["mean_score_by_label"] = {l: round(sum(s for s, x in zip(cal, labels) if x == l) /
                                            max(1, labels.count(l)), 2) for l in ("NAG", "CAG", "OAG")}
        m["count_by_label"] = {l: labels.count(l) for l in ("NAG", "CAG", "OAG")}
        pairs = [(1 if l == "OAG" else 0, s) for l, s in zip(labels, cal) if l in ("NAG", "OAG")]
        m["auroc_NAG_vs_OAG"] = auroc([p for p, _ in pairs], [s for _, s in pairs])
        m["auroc_NAG_vs_aggressive"] = auroc([0 if l == "NAG" else 1 for l in labels], cal)
        raw_pairs = [s for l, s in zip(labels, raw) if l in ("NAG", "OAG")]
        m["ece_NAG_vs_OAG_before"] = ece([p for p, _ in pairs], [s / 100 for s in raw_pairs])
        m["ece_NAG_vs_OAG_after"] = ece([p for p, _ in pairs], [s / 100 for _, s in pairs])
        gold = [{"NAG": "safe", "CAG": "sensitive", "OAG": "offensive"}[l] for l in labels]
        pred = [band(s) for s in cal]
        m["band_accuracy"] = round(100 * sum(g == p for g, p in zip(gold, pred)) / len(gold), 2)
        m["band_macro_f1"] = round(100 * macro_f1(gold, pred, ["safe", "sensitive", "offensive"]), 2)
        p, r, f1 = prf([l == "OAG" for l in labels], [s >= 50 for s in cal])
        m["OAG_at_50"] = {"precision": round(100 * p, 2), "recall": round(100 * r, 2), "f1": round(100 * f1, 2)}
    elif "kaggle" in name:
        y = [1 if r["target"] >= 0.5 else 0 for r in rows]
        m["auroc"] = auroc(y, cal)
        p, r, f1 = prf(y, [s >= 50 for s in cal])
        m["at_50"] = {"precision": round(100 * p, 2), "recall": round(100 * r, 2), "f1": round(100 * f1, 2)}
        m["ece_before"] = ece(y, [s / 100 for s in raw])
        m["ece_after"] = ece(y, [s / 100 for s in cal])
    for k, v in list(m.items()):
        if isinstance(v, float):
            m[k] = round(v, 4)
    return m, cal


all_metrics = {"run": RUN_NAME, "model_path": MODEL_PATH, "temperature": TEMPERATURE, "tests": {}}
for path in TEST_FILES:
    name = os.path.basename(path).replace(".jsonl", "")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not rows:
        continue
    logits = logits_for([r["text"] for r in rows])
    m, scores = metrics_for(name, rows, logits)
    all_metrics["tests"][name] = m
    logger.info(f"{name}: {json.dumps(m)}")
    with open(os.path.join(OUT_DIR, f"predictions_{name}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "target", "source", "score", "band"])
        for r, s in zip(rows, scores):
            w.writerow([r["text"], r["label"], r["target"], r["source"], round(s, 2), band(s)])
    ranked = sorted(zip(scores, rows), key=lambda x: x[0])
    with open(os.path.join(OUT_DIR, f"examples_{name}.md"), "w", encoding="utf-8") as f:
        f.write(f"# {name}: lowest and highest scores\n\n| score | label | text |\n|---|---|---|\n")
        for s, r in ranked[:N_EXAMPLES] + ranked[-N_EXAMPLES:]:
            f.write(f"| {s:.1f} | {r['label']} | {r['text'][:160].replace('|', '/')} |\n")

sample_scores = scores_from(logits_for(SAMPLE_TEXTS), TEMPERATURE)
all_metrics["sample_texts"] = [{"text": t, "score": round(s, 1), "band": band(s)} for t, s in zip(SAMPLE_TEXTS, sample_scores)]
for item in all_metrics["sample_texts"]:
    logger.info(f"  {item['score']:5.1f}%  {item['band']:9}  {item['text']}")
with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(all_metrics, f, indent=2, ensure_ascii=False)
logger.info(f"Done. Results in {OUT_DIR}/")
