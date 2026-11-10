import os
import sys
import json
import random
import logging
from collections import Counter

from nwp_common import clean_text, clean_persona, chunk_turns

SEED = int(os.getenv("SEED", "42"))
DATA_DIR = os.getenv("DATA_DIR", "data_nwp")
INDICTALK_NAME = os.getenv("INDICTALK_NAME", "LingoIITGN/IndicTalk")
INDICTALK_CONFIG = os.getenv("INDICTALK_CONFIG", "hindi")
INDICTALK_JSONL = os.getenv("INDICTALK_JSONL", "")
A2_NAME = os.getenv("A2_NAME", "theguywithblacktie/hinglish-conversations")
A2_CONFIG = os.getenv("A2_CONFIG", "large")
A2_JSONL_TRAIN = os.getenv("A2_JSONL_TRAIN", "")
A2_JSONL_TEST = os.getenv("A2_JSONL_TEST", "")
VAL_CONV_FRAC = float(os.getenv("VAL_CONV_FRAC", "0.02"))
TEST_CONV_FRAC = float(os.getenv("TEST_CONV_FRAC", "0.04"))
N_VAL_A2 = int(os.getenv("N_VAL_A2", "2000"))
MAX_A2_TRAIN = int(os.getenv("MAX_A2_TRAIN", "0"))
MAX_A2_TEST = int(os.getenv("MAX_A2_TEST", "0"))
MAX_CHUNK_WORDS = int(os.getenv("MAX_CHUNK_WORDS", "450"))

os.makedirs(DATA_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(os.path.join(DATA_DIR, "prepare_data_nwp.log"))],
)
logger = logging.getLogger(__name__)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def indictalk_rows():
    if INDICTALK_JSONL:
        yield from read_jsonl(INDICTALK_JSONL)
        return
    from datasets import load_dataset
    yield from load_dataset(INDICTALK_NAME, INDICTALK_CONFIG, split="train")


def a2_rows(split):
    path = A2_JSONL_TRAIN if split == "train" else A2_JSONL_TEST
    if path:
        yield from read_jsonl(path)
        return
    from datasets import load_dataset
    yield from load_dataset(A2_NAME, A2_CONFIG, split=split)


def message_texts(messages):
    out = []
    for m in messages or []:
        try:
            text = clean_text(str(m.get("content", "") if isinstance(m, dict) else m))
        except Exception:
            text = ""
        if text:
            out.append(text)
    return out


def load_indictalk():
    convs, seen, roman = [], 0, 0
    for idx, row in enumerate(indictalk_rows()):
        seen += 1
        if not str(row.get("Script", "")).strip().lower().startswith("roman"):
            continue
        roman += 1
        conv = row.get("Conversation") or []
        if isinstance(conv, str):
            conv = json.loads(conv)
        turns = [clean_text(t.get("text", "") if isinstance(t, dict) else t) for t in conv]
        turns = [t for t in turns if t]
        if turns:
            convs.append({"id": f"it{idx}", "source": "indictalk",
                          "persona": clean_persona(row.get("Persona")), "turns": turns})
    logger.info(f"IndicTalk rows read: {seen:,} | Romanized: {roman:,} | conversations kept: {len(convs):,}")
    return convs


def load_a2(split):
    convs = []
    for idx, row in enumerate(a2_rows(split)):
        turns = message_texts(row.get("input")) + message_texts(row.get("output"))
        if turns:
            convs.append({"id": f"a2{split[:2]}{idx}", "source": "a2", "persona": "none", "turns": turns})
    logger.info(f"Assignment 2 {split}: {len(convs):,} conversations")
    return convs


def chunked(convs):
    out = []
    for c in convs:
        pieces = chunk_turns(c["turns"], MAX_CHUNK_WORDS)
        for k, turns in enumerate(pieces):
            out.append({**c, "id": f"{c['id']}_{k}" if len(pieces) > 1 else c["id"], "turns": turns})
    return out


def write(name, rows):
    path = os.path.join(DATA_DIR, name)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    words = [sum(len(t.split()) for t in r["turns"]) for r in rows]
    stats = {
        "rows": len(rows),
        "sources": dict(Counter(r["source"] for r in rows)),
        "personas": dict(Counter(r["persona"] for r in rows)),
        "mean_words": round(sum(words) / max(len(words), 1), 1),
        "max_words": max(words) if words else 0,
        "total_words": sum(words),
    }
    logger.info(f"Wrote {name}: {stats}")
    return stats


rng = random.Random(SEED)

it = load_indictalk()
rng.shuffle(it)
n_test = max(1, round(len(it) * TEST_CONV_FRAC))
n_val = max(1, round(len(it) * VAL_CONV_FRAC))
it_test, it_val, it_train = it[:n_test], it[n_test:n_test + n_val], it[n_test + n_val:]
logger.info(f"IndicTalk conversation split: train {len(it_train):,} | val {len(it_val):,} | test {len(it_test):,}")

a2_train = load_a2("train")
rng.shuffle(a2_train)
a2_val, a2_train = a2_train[:N_VAL_A2], a2_train[N_VAL_A2:]
if MAX_A2_TRAIN > 0:
    a2_train = a2_train[:MAX_A2_TRAIN]
a2_test = load_a2("test")
if MAX_A2_TEST > 0:
    a2_test = rng.sample(a2_test, min(MAX_A2_TEST, len(a2_test)))

train = chunked(it_train) + chunked(a2_train)
rng.shuffle(train)
stats = {"seed": SEED, "max_chunk_words": MAX_CHUNK_WORDS, "files": {}}
stats["files"]["train.jsonl"] = write("train.jsonl", train)
stats["files"]["val.jsonl"] = write("val.jsonl", chunked(it_val) + chunked(a2_val))
stats["files"]["test_indictalk.jsonl"] = write("test_indictalk.jsonl", chunked(it_test))
stats["files"]["test_a2.jsonl"] = write("test_a2.jsonl", chunked(a2_test))
with open(os.path.join(DATA_DIR, "data_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)
logger.info(f"Done. Files are in {DATA_DIR}/")
