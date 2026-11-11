import os
import re
import sys
import random
import logging

import pandas as pd

from hinglish_noise import noisy_message, coverage_report

SEED = int(os.getenv("SEED", "42"))
DATA_DIR = os.getenv("DATA_DIR", "data_a3")
A2_NAME = os.getenv("A2_NAME", "theguywithblacktie/hinglish-conversations")
A2_CONFIG = os.getenv("A2_CONFIG", "large")
A2_TRAIN_CSV = os.getenv("A2_TRAIN_CSV", "grammar_dataset_train.csv")
MAX_A2_TRAIN = int(os.getenv("MAX_A2_TRAIN", "150000"))
OUT_FILE = os.getenv("OUT_FILE", "train_a2_full.csv")
STEM = os.path.splitext(OUT_FILE)[0]
LOG_NAME = "prepare_a2_full_a3.log" if OUT_FILE == "train_a2_full.csv" else f"prepare_{STEM}.log"
COVERAGE_NAME = "coverage_report_a2_full.txt" if OUT_FILE == "train_a2_full.csv" else f"coverage_report_{STEM}.txt"
P_CLEAN = float(os.getenv("P_CLEAN", "0.15"))
MIN_WORDS = int(os.getenv("MIN_WORDS", "3"))
MAX_WORDS = int(os.getenv("MAX_WORDS", "60"))
EVAL_FILES = ["val.csv", "test_indictalk.csv", "test_a2.csv", "test_a2_original.csv"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(os.path.join(DATA_DIR, LOG_NAME))],
)
logger = logging.getLogger(__name__)

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
INVISIBLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u2028\u2029\ufeff]")


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", INVISIBLE.sub("", text)).strip()


def text_key(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def keep_text(text):
    n = len(text.split())
    if not MIN_WORDS <= n <= MAX_WORDS or DEVANAGARI.search(text):
        return False
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) >= 0.9


def load_a2_train_texts():
    if A2_TRAIN_CSV and os.path.exists(A2_TRAIN_CSV):
        df = pd.read_csv(A2_TRAIN_CSV, keep_default_na=False)
        texts = [normalize_text(t) for t in df["correct_text"].tolist()]
        logger.info(f"A2 train: {len(texts):,} clean messages from {A2_TRAIN_CSV}")
        return texts
    from datasets import load_dataset
    ds = load_dataset(A2_NAME, A2_CONFIG, split="train")
    texts = []
    for row in ds:
        for col in ("input", "output"):
            for msg in row[col] or []:
                try:
                    if "content" in msg:
                        texts.append(normalize_text(str(msg["content"])))
                except Exception:
                    pass
    logger.info(f"A2 train: {len(texts):,} clean messages from {A2_NAME}")
    return texts


if not os.path.exists(os.path.join(DATA_DIR, "test_a2.csv")):
    sys.exit(f"{DATA_DIR}/test_a2.csv not found. Run prepare_data_a3.sh first.")

eval_keys = set()
for name in EVAL_FILES:
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        eval_keys |= {text_key(t) for t in pd.read_csv(path, keep_default_na=False)["correct_text"].astype(str)}
logger.info(f"Sentences reserved for validation/test: {len(eval_keys):,}")

seen, pool = set(), []
for t in load_a2_train_texts():
    k = text_key(t)
    if t and keep_text(t) and k and k not in seen and k not in eval_keys:
        seen.add(k)
        pool.append(t)
random.Random(SEED + 7).shuffle(pool)
if MAX_A2_TRAIN > 0:
    pool = pool[:MAX_A2_TRAIN]

random.seed(SEED + 8)
rows = []
for text in pool:
    noisy, tags = noisy_message(text, P_CLEAN)
    rows.append({"incorrect_text": noisy, "correct_text": text, "persona": "none",
                 "source": "a2", "tags": ";".join(tags)})
df = pd.DataFrame(rows, columns=["incorrect_text", "correct_text", "persona", "source", "tags"])
out = os.path.join(DATA_DIR, OUT_FILE)
df.to_csv(out, index=False)

lines = coverage_report(df["tags"].tolist(), OUT_FILE)
for line in lines:
    logger.info(line)
with open(os.path.join(DATA_DIR, COVERAGE_NAME), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
logger.info(f"Wrote {out}: {len(df):,} rows | unchanged inputs {(df['incorrect_text'] == df['correct_text']).mean():.1%}")
