import os
import sys
import csv
import json
import random
import logging
from collections import Counter, defaultdict

from sens_common import clean_text, is_romanized, load_lexicon, has_lexicon_word, TRAC_LABELS

SEED = int(os.getenv("SEED", "42"))
DATA_DIR = os.getenv("DATA_DIR", "data_sens")
RAW_DIR = os.getenv("RAW_DIR", "raw")
TRAC_DIR = os.getenv("TRAC_DIR", os.path.join(RAW_DIR, "trac-1"))
LEXICON = os.getenv("LEXICON", os.path.join(RAW_DIR, "Hinglish-Offensive-Text-Classification",
                                            "Hinglish_Profanity_List.csv"))
KAGGLE_CSV = os.getenv("KAGGLE_CSV", "")
CAG_TARGET = float(os.getenv("CAG_TARGET", "0.5"))
TRAC_VAL_FRAC = float(os.getenv("TRAC_VAL_FRAC", "0.1"))
KAGGLE_VAL_FRAC = float(os.getenv("KAGGLE_VAL_FRAC", "0.1"))
KAGGLE_TEST_FRAC = float(os.getenv("KAGGLE_TEST_FRAC", "0.1"))
MIN_WORDS = int(os.getenv("MIN_WORDS", "2"))
MAX_SAFE_WORDS = int(os.getenv("MAX_SAFE_WORDS", "60"))
N_SAFE_IT = [int(x) for x in os.getenv("N_SAFE_IT", "2500,300,1000").split(",")]   # train,val,test
N_SAFE_A2 = [int(x) for x in os.getenv("N_SAFE_A2", "2500,300,1000").split(",")]   # train,val,test
N_LEXICON_TEST = int(os.getenv("N_LEXICON_TEST", "1000"))
LEXICON_TEST_MIN_SEVERITY = int(os.getenv("LEXICON_TEST_MIN_SEVERITY", "5"))
USE_SAFE_CHAT = os.getenv("USE_SAFE_CHAT", "1") == "1"
INDICTALK_NAME = os.getenv("INDICTALK_NAME", "LingoIITGN/IndicTalk")
INDICTALK_CONFIG = os.getenv("INDICTALK_CONFIG", "hindi")
INDICTALK_JSONL = os.getenv("INDICTALK_JSONL", "")
A2_NAME = os.getenv("A2_NAME", "theguywithblacktie/hinglish-conversations")
A2_CONFIG = os.getenv("A2_CONFIG", "large")
A2_JSONL_TRAIN = os.getenv("A2_JSONL_TRAIN", "")
A2_JSONL_TEST = os.getenv("A2_JSONL_TEST", "")

os.makedirs(DATA_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(os.path.join(DATA_DIR, "prepare_data_sens.log"))],
)
logger = logging.getLogger(__name__)
rng = random.Random(SEED)
seen_texts = set()


def keep(text):
    key = text.lower()
    if len(text.split()) < MIN_WORDS or key in seen_texts:
        return False
    seen_texts.add(key)
    return True


def stratified_split(rows, fracs, key="label"):
    """fracs: fractions of the later parts; the first part gets the rest."""
    by = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    parts = [[] for _ in range(len(fracs) + 1)]
    for items in by.values():
        rng.shuffle(items)
        start = len(items)
        for k in range(len(fracs), 0, -1):
            n = int(round(len(items) * fracs[k - 1]))
            parts[k].extend(items[start - n:start])
            start -= n
        parts[0].extend(items[:start])
    for p in parts:
        rng.shuffle(p)
    return parts


# ---------- TRAC-1 Hindi (Facebook comments, aggression) ----------
def read_trac(path):
    rows, counts = [], Counter()
    with open(path, encoding="utf-8") as f:
        for rec in csv.reader(f):
            if len(rec) < 3:
                counts["bad_row"] += 1
                continue
            label = rec[-1].strip().upper()
            text = clean_text(",".join(rec[1:-1]))
            if label not in TRAC_LABELS:
                counts["bad_label"] += 1
                continue
            if not is_romanized(text):
                counts["devanagari"] += 1
                continue
            if not keep(text):
                counts["short_or_duplicate"] += 1
                continue
            target = {"NAG": 0.0, "CAG": CAG_TARGET, "OAG": 1.0}[label]
            rows.append({"text": text, "target": target, "label": label, "source": "trac"})
            counts[label] += 1
    logger.info(f"{path}: {dict(counts)}")
    return rows


trac_train_file = os.path.join(TRAC_DIR, "hindi", "agr_hi_train.csv")
trac_dev_file = os.path.join(TRAC_DIR, "hindi", "agr_hi_dev.csv")
for f in (trac_train_file, trac_dev_file):
    if not os.path.exists(f):
        logger.error(f"{f} not found. Clone it first: git clone https://github.com/kmi-linguistics/trac-1 {TRAC_DIR}")
        sys.exit(1)
trac_test = read_trac(trac_dev_file)          # official dev set = our test set (never trained on)
trac_all = read_trac(trac_train_file)
trac_train, trac_val = stratified_split(trac_all, [TRAC_VAL_FRAC])


# ---------- Optional Kaggle Hinglish hate-speech CSV ----------
def pick_column(columns, candidates):
    low = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def to_binary(value):
    v = str(value).strip().lower()
    try:
        return 1 if float(v) > 0 else 0
    except ValueError:
        pass
    if any(w in v for w in ("non", "not", "normal", "neutral", "no hate", "none", "clean")):
        return 0
    if any(w in v for w in ("hate", "offens", "abus", "toxic", "yes", "true")):
        return 1
    return None


kaggle_train, kaggle_val, kaggle_test = [], [], []
if KAGGLE_CSV:
    if not os.path.exists(KAGGLE_CSV):
        logger.error(f"KAGGLE_CSV={KAGGLE_CSV} not found")
        sys.exit(1)
    with open(KAGGLE_CSV, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        text_col = os.getenv("KAGGLE_TEXT_COL") or pick_column(
            cols, ["text", "clean_text", "tweet", "comment", "sentence", "content", "message", "post"])
        label_col = os.getenv("KAGGLE_LABEL_COL") or pick_column(
            cols, ["label", "hate", "is_hate", "hate_label", "class", "target", "hate_speech", "category"])
        lang_col = os.getenv("KAGGLE_LANG_COL") or pick_column(cols, ["language", "lang", "language_type"])
        logger.info(f"Kaggle CSV columns: {cols} | text: {text_col} | label: {label_col} | language: {lang_col}")
        if not text_col or not label_col:
            logger.error("Could not find the text or label column. Set KAGGLE_TEXT_COL and KAGGLE_LABEL_COL.")
            sys.exit(1)
        rows, counts = [], Counter()
        for r in reader:
            if lang_col:
                lang = str(r.get(lang_col, "")).lower()
                if not ("hinglish" in lang or "code" in lang or "mix" in lang):
                    counts["other_language"] += 1
                    continue
            text = clean_text(r.get(text_col, ""))
            y = to_binary(r.get(label_col, ""))
            if y is None:
                counts["unknown_label"] += 1
                continue
            if not is_romanized(text):
                counts["devanagari_or_empty"] += 1
                continue
            if not keep(text):
                counts["short_or_duplicate"] += 1
                continue
            rows.append({"text": text, "target": float(y), "label": "hate" if y else "non_hate", "source": "kaggle"})
            counts["hate" if y else "non_hate"] += 1
        logger.info(f"Kaggle rows kept: {dict(counts)}")
        kaggle_train, kaggle_val, kaggle_test = stratified_split(rows, [KAGGLE_VAL_FRAC, KAGGLE_TEST_FRAC])


# ---------- Safe chat: IndicTalk and Assignment 2 messages ----------
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def indictalk_conversations():
    rows = read_jsonl(INDICTALK_JSONL) if INDICTALK_JSONL else None
    if rows is None:
        from datasets import load_dataset
        rows = load_dataset(INDICTALK_NAME, INDICTALK_CONFIG, split="train")
    for row in rows:
        if not str(row.get("Script", "")).strip().lower().startswith("roman"):
            continue
        conv = row.get("Conversation") or []
        if isinstance(conv, str):
            conv = json.loads(conv)
        turns = [clean_text(t.get("text", "") if isinstance(t, dict) else str(t)) for t in conv]
        yield [t for t in turns if t]


def a2_messages(split):
    path = A2_JSONL_TRAIN if split == "train" else A2_JSONL_TEST
    rows = read_jsonl(path) if path else None
    if rows is None:
        from datasets import load_dataset
        rows = load_dataset(A2_NAME, A2_CONFIG, split=split)
    for row in rows:
        for m in list(row.get("input") or []) + list(row.get("output") or []):
            text = clean_text(m.get("content", "") if isinstance(m, dict) else str(m))
            if text:
                yield text


def take(texts, n, source, label="safe_chat"):
    out = []
    for t in texts:
        if len(out) >= n:
            break
        if keep(t):
            out.append({"text": t, "target": 0.0, "label": label, "source": source})
    return out


lexicon, lexicon_strong = set(), set()
if os.path.exists(LEXICON):
    lexicon = load_lexicon(LEXICON, 1)
    lexicon_strong = load_lexicon(LEXICON, LEXICON_TEST_MIN_SEVERITY)
    logger.info(f"Profanity lexicon: {len(lexicon)} words ({len(lexicon_strong)} with severity >= "
                f"{LEXICON_TEST_MIN_SEVERITY})")
else:
    logger.warning(f"{LEXICON} not found: Assignment 2 safe messages are not filtered for profanity, "
                   "and there is no lexicon test set")

safe = {"train": [], "val": [], "test_safe_indictalk": [], "test_safe_a2": []}
lexicon_test = []
if USE_SAFE_CHAT:
    convs = [c for c in indictalk_conversations() if c]
    rng.shuffle(convs)
    n_test_conv = max(1, len(convs) // 10)
    test_turns = [t for c in convs[:n_test_conv] for t in c]
    train_turns = [t for c in convs[n_test_conv:] for t in c]
    fits = lambda t: is_romanized(t) and len(t.split()) <= MAX_SAFE_WORDS
    rng.shuffle(train_turns)
    rng.shuffle(test_turns)
    train_turns = [t for t in train_turns if fits(t)]
    safe["train"] += take(train_turns, N_SAFE_IT[0], "indictalk")
    safe["val"] += take(train_turns[N_SAFE_IT[0] * 2:], N_SAFE_IT[1], "indictalk")
    safe["test_safe_indictalk"] = take([t for t in test_turns if fits(t)], N_SAFE_IT[2], "indictalk")

    a2_train = [t for t in a2_messages("train") if fits(t)]
    rng.shuffle(a2_train)
    clean_a2 = [t for t in a2_train if not has_lexicon_word(t, lexicon)]
    safe["train"] += take(clean_a2, N_SAFE_A2[0], "a2")
    safe["val"] += take(clean_a2[N_SAFE_A2[0] * 2:], N_SAFE_A2[1], "a2")

    a2_test = [t for t in a2_messages("test") if fits(t)]
    rng.shuffle(a2_test)
    safe["test_safe_a2"] = take([t for t in a2_test if not has_lexicon_word(t, lexicon)], N_SAFE_A2[2], "a2")
    if lexicon_strong:
        # Weak positives for evaluation only: Assignment 2 test messages with a strong profanity word.
        lexicon_test = take([t for t in a2_test if has_lexicon_word(t, lexicon_strong)], N_LEXICON_TEST,
                            "a2_lexicon", label="profanity_lexicon")
        for r in lexicon_test:
            r["target"] = 1.0


# ---------- Write ----------
def write(name, rows):
    path = os.path.join(DATA_DIR, name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    stat = {"rows": len(rows), "by_source_label": dict(Counter(f"{r['source']}:{r['label']}" for r in rows))}
    logger.info(f"Wrote {name}: {stat}")
    return stat


train = trac_train + kaggle_train + safe["train"]
val = trac_val + kaggle_val + safe["val"]
rng.shuffle(train)
rng.shuffle(val)
stats = {"settings": {"CAG_TARGET": CAG_TARGET, "KAGGLE_CSV": KAGGLE_CSV or None, "USE_SAFE_CHAT": USE_SAFE_CHAT,
                      "N_SAFE_IT": N_SAFE_IT, "N_SAFE_A2": N_SAFE_A2, "SEED": SEED}}
stats["train.jsonl"] = write("train.jsonl", train)
stats["val.jsonl"] = write("val.jsonl", val)
stats["test_trac.jsonl"] = write("test_trac.jsonl", trac_test)
if kaggle_test:
    stats["test_kaggle.jsonl"] = write("test_kaggle.jsonl", kaggle_test)
for name in ("test_safe_indictalk", "test_safe_a2"):
    if safe[name]:
        stats[f"{name}.jsonl"] = write(f"{name}.jsonl", safe[name])
if lexicon_test:
    stats["test_a2_lexicon.jsonl"] = write("test_a2_lexicon.jsonl", lexicon_test)
with open(os.path.join(DATA_DIR, "data_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)
logger.info("Done.")
