import os
import re
import sys
import json
import random
import difflib
import logging

import pandas as pd

from hinglish_noise import noisy_message, coverage_report

SEED = int(os.getenv("SEED", "42"))
DATA_DIR = os.getenv("DATA_DIR", "data_a3")

INDICTALK_NAME = os.getenv("INDICTALK_NAME", "LingoIITGN/IndicTalk")
INDICTALK_CONFIG = os.getenv("INDICTALK_CONFIG", "hindi")
INDICTALK_JSONL = os.getenv("INDICTALK_JSONL", "")

A2_NAME = os.getenv("A2_NAME", "theguywithblacktie/hinglish-conversations")
A2_CONFIG = os.getenv("A2_CONFIG", "large")
A2_TRAIN_CSV = os.getenv("A2_TRAIN_CSV", "grammar_dataset_train.csv")
A2_TEST_CSV = os.getenv("A2_TEST_CSV", "grammar_dataset_test.csv")

MAX_INDICTALK_TRAIN = int(os.getenv("MAX_INDICTALK_TRAIN", "150000"))
REPLAY_RATIO = float(os.getenv("REPLAY_RATIO", "0.25"))
VAL_CONV_FRAC = float(os.getenv("VAL_CONV_FRAC", "0.02"))
TEST_CONV_FRAC = float(os.getenv("TEST_CONV_FRAC", "0.04"))
N_VAL_INDICTALK = int(os.getenv("N_VAL_INDICTALK", "1500"))
N_VAL_A2 = int(os.getenv("N_VAL_A2", "500"))
N_TEST = int(os.getenv("N_TEST", "2000"))
P_CLEAN = float(os.getenv("P_CLEAN", "0.15"))
MIN_WORDS = int(os.getenv("MIN_WORDS", "3"))
MAX_WORDS = int(os.getenv("MAX_WORDS", "60"))
KEEP_SPLIT_FROM = os.getenv("KEEP_SPLIT_FROM", "")
if KEEP_SPLIT_FROM and os.path.abspath(KEEP_SPLIT_FROM) == os.path.abspath(DATA_DIR):
    sys.exit("KEEP_SPLIT_FROM must be a different folder from DATA_DIR")

os.makedirs(DATA_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(os.path.join(DATA_DIR, "prepare_data_a3.log"))],
)
logger = logging.getLogger(__name__)
logger.info(f"Settings: DATA_DIR={DATA_DIR} | words per message {MIN_WORDS} to {MAX_WORDS if MAX_WORDS > 0 else 'no limit'} | "
            f"MAX_INDICTALK_TRAIN={MAX_INDICTALK_TRAIN or 'no cap'} | "
            f"GRAMMAR_SCALE_WORDS={os.getenv('GRAMMAR_SCALE_WORDS', '0')} | KEEP_SPLIT_FROM={KEEP_SPLIT_FROM or '-'}")

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
INVISIBLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u2028\u2029\ufeff]")


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", INVISIBLE.sub("", text)).strip()


def text_key(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def is_romanized(text):
    if DEVANAGARI.search(text):
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(c.isascii() for c in letters) / len(letters) >= 0.9


def keep_text(text):
    n = len(text.split())
    return n >= MIN_WORDS and (MAX_WORDS <= 0 or n <= MAX_WORDS) and is_romanized(text)


def turn_text(turn):
    if isinstance(turn, dict):
        for k in ("text", "content", "utterance", "message"):
            if turn.get(k):
                return normalize_text(turn[k])
        return ""
    return normalize_text(turn)


def iter_indictalk_rows():
    if INDICTALK_JSONL:
        with open(INDICTALK_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    from datasets import load_dataset
    ds = load_dataset(INDICTALK_NAME, INDICTALK_CONFIG, split="train")
    for row in ds:
        yield row


def load_indictalk():
    convs, seen_rows, roman_rows = [], 0, 0
    word_counts, too_short, too_long, not_roman = [], 0, 0, 0
    for idx, row in enumerate(iter_indictalk_rows()):
        seen_rows += 1
        if not str(row.get("Script", "")).strip().lower().startswith("roman"):
            continue
        roman_rows += 1
        conv = row.get("Conversation") or []
        if isinstance(conv, str):
            conv = json.loads(conv)
        texts = []
        for t in (turn_text(x) for x in conv):
            if not t:
                continue
            n = len(t.split())
            word_counts.append(n)
            if n < MIN_WORDS:
                too_short += 1
            elif MAX_WORDS > 0 and n > MAX_WORDS:
                too_long += 1
            elif not is_romanized(t):
                not_roman += 1
            else:
                texts.append(t)
        if texts:
            convs.append((idx, str(row.get("Persona") or "none").strip().lower(), texts))
    logger.info(f"IndicTalk rows read: {seen_rows:,} | Romanized: {roman_rows:,} | "
                f"Romanized conversations kept: {len(convs):,}")
    if word_counts:
        wc = sorted(word_counts)
        pick = lambda q: wc[min(len(wc) - 1, int(q * len(wc)))]
        logger.info(f"IndicTalk turns: {len(wc):,} | words per turn p50 {pick(0.5)} | p90 {pick(0.9)} | "
                    f"p99 {pick(0.99)} | max {wc[-1]}")
        limit = f"over {MAX_WORDS} words {too_long:,} ({too_long / len(wc):.1%})" if MAX_WORDS > 0 else "no upper limit"
        logger.info(f"IndicTalk turns dropped: under {MIN_WORDS} words {too_short:,} | {limit} | "
                    f"not Romanized {not_roman:,}")
    return convs


def extract_contents(msg_list):
    out = []
    if msg_list is None:
        return out
    for msg in msg_list:
        try:
            if "content" in msg:
                out.append(normalize_text(str(msg["content"])))
        except Exception:
            pass
    return out


def load_a2_texts(csv_path, split):
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, keep_default_na=False)
        texts = [normalize_text(t) for t in df["correct_text"].tolist()]
        logger.info(f"A2 {split}: {len(texts):,} clean messages from {csv_path}")
    else:
        from datasets import load_dataset
        ds = load_dataset(A2_NAME, A2_CONFIG, split=split)
        texts = []
        for row in ds:
            texts += extract_contents(row["input"]) + extract_contents(row["output"])
        logger.info(f"A2 {split}: {len(texts):,} clean messages from {A2_NAME}")
    return [t for t in texts if t and keep_text(t)]


def dedup(texts, banned=None):
    banned = banned or set()
    seen, out = set(), []
    for t in texts:
        k = text_key(t)
        if k and k not in seen and k not in banned:
            seen.add(k)
            out.append(t)
    return out


def make_pairs(items, source, seed):
    random.seed(seed)
    rows = []
    for text, persona in items:
        noisy, tags = noisy_message(text, P_CLEAN)
        rows.append({"incorrect_text": noisy, "correct_text": text, "persona": persona,
                     "source": source, "tags": ";".join(tags)})
    return pd.DataFrame(rows, columns=["incorrect_text", "correct_text", "persona", "source", "tags"])


def sample(items, n, rng):
    items = list(items)
    rng.shuffle(items)
    return items if n <= 0 else items[:n]


rng = random.Random(SEED)

convs = load_indictalk()
rng.shuffle(convs)
n_test = max(1, round(len(convs) * TEST_CONV_FRAC))
n_val = max(1, round(len(convs) * VAL_CONV_FRAC))
it_test_convs, it_val_convs, it_train_convs = convs[:n_test], convs[n_test:n_test + n_val], convs[n_test + n_val:]


def read_keys(folder, names):
    keys = set()
    for name in names:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            keys |= {text_key(t) for t in pd.read_csv(path, keep_default_na=False)["correct_text"].astype(str)}
    return keys


old_train_keys, old_eval_keys = set(), set()
if KEEP_SPLIT_FROM:
    old_train_keys = read_keys(KEEP_SPLIT_FROM, ["train_indictalk.csv", "train_a2_replay.csv", "train_a2_full.csv",
                                                 "train_a2_all.csv"])
    old_test_keys = read_keys(KEEP_SPLIT_FROM, ["test_indictalk.csv", "test_a2.csv", "test_a2_original.csv"])
    old_val_keys = read_keys(KEEP_SPLIT_FROM, ["val.csv"])
    old_it_test = read_keys(KEEP_SPLIT_FROM, ["test_indictalk.csv"])
    if not old_train_keys or not old_it_test:
        sys.exit(f"KEEP_SPLIT_FROM={KEEP_SPLIT_FROM}: train_indictalk.csv or test_indictalk.csv not found there")
    old_eval_keys = old_test_keys | old_val_keys
    new_keys = {text_key(t) for _, _, texts in convs for t in texts}
    found = len(old_it_test & new_keys) / len(old_it_test)
    logger.info(f"Old IndicTalk test sentences found in this data: {found:.1%}")
    if found < 0.99:
        logger.warning(f"Only {found:.1%} of the old test sentences were found, so the old split cannot be fully "
                       f"reused. Use the same IndicTalk source as for {KEEP_SPLIT_FROM}/.")
    parts = {"test": [], "val": [], "train": []}
    moved, dropped = 0, 0
    for name, part in (("test", it_test_convs), ("val", it_val_convs), ("train", it_train_convs)):
        for conv in part:
            keys = {text_key(t) for t in conv[2]}
            in_train, in_eval = bool(keys & old_train_keys), bool(keys & old_eval_keys)
            if in_train and in_eval:
                dropped += 1
                continue
            target = ("train" if in_train else "test" if keys & old_test_keys
                      else "val" if keys & old_val_keys else name)
            moved += target != name
            parts[target].append(conv)
    it_test_convs, it_val_convs, it_train_convs = parts["test"], parts["val"], parts["train"]
    logger.info(f"Kept the conversation split of {KEEP_SPLIT_FROM}/: {moved:,} conversations moved to their old "
                f"split, {dropped:,} dropped (sentences in both old training and old evaluation)")

logger.info(f"IndicTalk conversation split: train {len(it_train_convs):,} | val {len(it_val_convs):,} | "
            f"test {len(it_test_convs):,}")


def flatten(conv_list):
    return [(t, persona) for _, persona, texts in conv_list for t in texts]


def dedup_pairs(items, banned=None):
    banned = banned or set()
    seen, out = set(), []
    for t, p in items:
        k = text_key(t)
        if k and k not in seen and k not in banned:
            seen.add(k)
            out.append((t, p))
    return out


it_test = dedup_pairs(flatten(it_test_convs), old_train_keys)
it_val = dedup_pairs(flatten(it_val_convs), {text_key(t) for t, _ in it_test} | old_train_keys)

a2_test_texts = dedup(load_a2_texts(A2_TEST_CSV, "test"), old_train_keys)
a2_train_texts = dedup(load_a2_texts(A2_TRAIN_CSV, "train"))

it_test_s = sample(it_test, N_TEST, rng)
it_val_s = sample(it_val, N_VAL_INDICTALK, rng)
if KEEP_SPLIT_FROM:
    old_a2_test = pd.read_csv(os.path.join(KEEP_SPLIT_FROM, "test_a2.csv"), keep_default_na=False)
    old_val = pd.read_csv(os.path.join(KEEP_SPLIT_FROM, "val.csv"), keep_default_na=False)
    old_a2_val = old_val[old_val["source"] == "a2"].reset_index(drop=True)
    a2_test_s = []
    logger.info(f"Reusing {KEEP_SPLIT_FROM}/test_a2.csv ({len(old_a2_test):,} rows) and its {len(old_a2_val):,} "
                f"Assignment 2 validation rows unchanged")
else:
    a2_test_s = [(t, "none") for t in sample(a2_test_texts, N_TEST, rng)]

eval_keys = {text_key(t) for t, _ in it_test_s + it_val_s + a2_test_s} | old_eval_keys
a2_pool = [t for t in a2_train_texts if text_key(t) not in eval_keys]
rng.shuffle(a2_pool)
n_a2_val = 0 if KEEP_SPLIT_FROM else N_VAL_A2
a2_val_s = [(t, "none") for t in a2_pool[:n_a2_val]]
a2_pool = a2_pool[n_a2_val:]
eval_keys |= {text_key(t) for t, _ in a2_val_s}

it_train = dedup_pairs(flatten(it_train_convs), eval_keys)
it_train_s = sample(it_train, MAX_INDICTALK_TRAIN, rng)
n_replay = round(len(it_train_s) * REPLAY_RATIO / max(1e-9, 1 - REPLAY_RATIO)) if REPLAY_RATIO > 0 else 0
a2_replay_s = [(t, "none") for t in a2_pool[:n_replay]]
logger.info(f"IndicTalk train messages available: {len(it_train):,} | used: {len(it_train_s):,} | "
            f"A2 replay: {len(a2_replay_s):,} (ratio {REPLAY_RATIO})")

outputs = {
    "train_indictalk.csv": make_pairs(it_train_s, "indictalk", SEED + 1),
    "train_a2_replay.csv": make_pairs(a2_replay_s, "a2", SEED + 2),
    "val.csv": pd.concat([make_pairs(it_val_s, "indictalk", SEED + 3),
                          make_pairs(a2_val_s, "a2", SEED + 4)], ignore_index=True),
    "test_indictalk.csv": make_pairs(it_test_s, "indictalk", SEED + 5),
    "test_a2.csv": make_pairs(a2_test_s, "a2", SEED + 6),
}

if KEEP_SPLIT_FROM:
    outputs["val.csv"] = pd.concat([outputs["val.csv"], old_a2_val], ignore_index=True)
    outputs["test_a2.csv"] = old_a2_test

if A2_TEST_CSV and os.path.exists(A2_TEST_CSV):
    orig = pd.read_csv(A2_TEST_CSV, keep_default_na=False)
    orig = orig[(orig["incorrect_text"].str.len() > 0) & (orig["correct_text"].str.len() > 0)]
    orig = orig.sample(n=min(N_TEST, len(orig)), random_state=SEED)
    outputs["test_a2_original.csv"] = pd.DataFrame({
        "incorrect_text": orig["incorrect_text"].map(normalize_text),
        "correct_text": orig["correct_text"].map(normalize_text),
        "persona": "none", "source": "a2_original", "tags": "a2_noise",
    })

def edit_density(df, limit=20000):
    if not len(df):
        return 0.0, 0.0
    part = df if len(df) <= limit else df.sample(n=limit, random_state=SEED)
    edits = words = 0
    for a, b in zip(part["incorrect_text"].astype(str), part["correct_text"].astype(str)):
        x, y = a.split(), b.split()
        words += len(y)
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=x, b=y, autojunk=False).get_opcodes():
            if op != "equal":
                edits += max(i2 - i1, j2 - j1)
    return round(100 * edits / max(words, 1), 2), round(words / len(part), 1)


stats = {"seed": SEED, "replay_ratio": REPLAY_RATIO, "p_clean": P_CLEAN, "max_words": MAX_WORDS,
         "grammar_scale_words": int(os.getenv("GRAMMAR_SCALE_WORDS", "0")), "files": {}}
report = []
if KEEP_SPLIT_FROM:
    for name in ("train_indictalk.csv", "train_a2_replay.csv"):
        path = os.path.join(KEEP_SPLIT_FROM, name)
        if os.path.exists(path):
            density, mean_words = edit_density(pd.read_csv(path, keep_default_na=False))
            logger.info(f"For comparison, {path}: {density} word edits per 100 words | mean {mean_words} words")
for name, df in outputs.items():
    df.to_csv(os.path.join(DATA_DIR, name), index=False)
    identity = float((df["incorrect_text"] == df["correct_text"]).mean()) if len(df) else 0.0
    density, mean_words = edit_density(df)
    stats["files"][name] = {
        "rows": int(len(df)),
        "identity_share": round(identity, 4),
        "word_edits_per_100_words": density,
        "mean_words": mean_words,
        "persona_counts": {k: int(v) for k, v in df["persona"].value_counts().items()},
    }
    logger.info(f"Wrote {name}: {len(df):,} rows | unchanged inputs {identity:.1%} | "
                f"{density} word edits per 100 words | mean {mean_words} words")
    if name != "test_a2_original.csv":
        lines = coverage_report(df["tags"].tolist(), name)
        report += lines + [""]
        for line in lines:
            logger.info(line)

with open(os.path.join(DATA_DIR, "coverage_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report))
with open(os.path.join(DATA_DIR, "data_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)
logger.info(f"Done. Files are in {DATA_DIR}/")
