import os
import csv
import sys
import json
import math
import time
import random
import logging
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from nwp_common import (PERSONA_MODES, persona_for_mode, persona_prefix, join_turns, word_positions,
                        is_word_start, build_word)

RUN_NAME = os.getenv("RUN_NAME", "0.6B")
MODEL_PATH = os.getenv("MODEL_PATH", f"runs/run_{RUN_NAME}/best")
PERSONA_MODE = os.getenv("PERSONA_MODE", "true")
TURN_SEPARATOR = os.getenv("TURN_SEPARATOR", "newline")
DATA_DIR = os.getenv("DATA_DIR", "data_nwp")
TEST_FILES = [f.strip() for f in os.getenv(
    "TEST_FILES", f"{DATA_DIR}/test_indictalk.jsonl,{DATA_DIR}/test_a2.jsonl").split(",") if f.strip()]
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "1024"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
WORD_BATCH_SIZE = int(os.getenv("WORD_BATCH_SIZE", "16"))
N_WORD_POSITIONS = int(os.getenv("N_WORD_POSITIONS", "3000"))
TOP_K = int(os.getenv("TOP_K", "3"))
CANDIDATE_POOL = int(os.getenv("CANDIDATE_POOL", "50"))
MAX_WORD_TOKENS = int(os.getenv("MAX_WORD_TOKENS", "6"))
MAX_TOKEN_EXAMPLES = int(os.getenv("MAX_TOKEN_EXAMPLES", "0"))
SEED = int(os.getenv("SEED", "42"))

if PERSONA_MODE not in PERSONA_MODES:
    sys.exit(f"PERSONA_MODE must be one of {PERSONA_MODES}")
LABEL = f"{RUN_NAME}__persona_{PERSONA_MODE}"
OUT_DIR = os.path.join(RESULTS_DIR, LABEL)
os.makedirs(OUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(OUT_DIR, "evaluation.log"))],
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
SUMMARY_FIELDS = ["run", "persona_mode", "test_set", "tokens", "loss", "perplexity", "top1", "top3",
                  "words", "word_top1", "word_top3", "word_mrr", "keystroke_savings"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=DTYPE)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=DTYPE)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()
prefix_cache = {}


def prefix_ids(persona):
    text = persona_prefix(persona)
    if text not in prefix_cache:
        prefix_cache[text] = tokenizer(text, add_special_tokens=False)["input_ids"] if text else []
    return prefix_cache[text]


def piece(token_id):
    return tokenizer.decode([token_id])


@torch.no_grad()
def token_metrics(rows):
    samples = []
    for start in range(0, len(rows), 2000):
        part = rows[start:start + 2000]
        texts = [join_turns(r["turns"], TURN_SEPARATOR) + tokenizer.eos_token for r in part]
        for r, t in zip(part, tokenizer(texts, add_special_tokens=False)["input_ids"]):
            p = prefix_ids(persona_for_mode(r["persona"], PERSONA_MODE))
            samples.append(((p + t)[:MAX_LENGTH], len(p), r["persona"]))
    order = sorted(range(len(samples)), key=lambda i: len(samples[i][0]), reverse=True)
    totals = defaultdict(lambda: [0.0, 0, 0, 0])
    for s in range(0, len(order), BATCH_SIZE):
        batch = [samples[i] for i in order[s:s + BATCH_SIZE]]
        width = max(len(ids) for ids, _, _ in batch)
        input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        for i, (ids, plen, _) in enumerate(batch):
            t = torch.tensor(ids, dtype=torch.long)
            input_ids[i, :len(ids)] = t
            mask[i, :len(ids)] = 1
            labels[i, plen:len(ids)] = t[plen:]
        input_ids, mask, labels = input_ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
        logits = model(input_ids=input_ids, attention_mask=mask).logits[:, :-1, :].float()
        target = labels[:, 1:]
        valid = target != -100
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-100,
                             reduction="none").view(target.shape)
        top3 = logits.topk(3, dim=-1).indices
        hit1 = (top3[..., 0] == target) & valid
        hit3 = (top3 == target.unsqueeze(-1)).any(-1) & valid
        for i, (_, _, persona) in enumerate(batch):
            for key in ("all", persona):
                totals[key][0] += float(ce[i][valid[i]].sum())
                totals[key][1] += int(valid[i].sum())
                totals[key][2] += int(hit1[i].sum())
                totals[key][3] += int(hit3[i].sum())
        del logits, ce, top3
    out = {}
    for key, (loss, n, h1, h3) in totals.items():
        avg = loss / max(n, 1)
        out[key] = {"tokens": n, "loss": round(avg, 5), "perplexity": round(math.exp(min(avg, 20)), 4),
                    "top1": round(100 * h1 / max(n, 1), 3), "top3": round(100 * h3 / max(n, 1), 3)}
    return out


def left_pad(seqs):
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, width - len(s):] = torch.tensor(s, dtype=torch.long)
        mask[i, width - len(s):] = 1
    return ids.to(DEVICE), mask.to(DEVICE)


def last_logits(ids, mask):
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    try:
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=1)
    except TypeError:
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos)
    return out.logits[:, -1, :].float()


@torch.no_grad()
def predict_words(contexts):
    ids, mask = left_pad(contexts)
    logits = last_logits(ids, mask)
    pool = logits.topk(CANDIDATE_POOL, dim=-1).indices.tolist()
    firsts = []
    for row in pool:
        chosen = [t for t in row if is_word_start(piece(t))][:TOP_K]
        firsts.append(chosen)
    seqs, owners = [], []
    for b, chosen in enumerate(firsts):
        for t in chosen:
            seqs.append(contexts[b] + [t])
            owners.append((b, t))
    words = [[] for _ in contexts]
    if not seqs:
        return words
    if MAX_WORD_TOKENS > 1:
        g_ids, g_mask = left_pad(seqs)
        gen = model.generate(input_ids=g_ids, attention_mask=g_mask, max_new_tokens=MAX_WORD_TOKENS - 1,
                             do_sample=False, temperature=None, top_p=None, top_k=None,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        new = gen[:, g_ids.size(1):].tolist()
    else:
        new = [[] for _ in seqs]
    for (b, t), cont in zip(owners, new):
        pieces = [piece(t)]
        for tok in cont:
            if tok in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                break
            pieces.append(piece(tok))
        word = build_word(pieces)
        if word and word not in words[b]:
            words[b].append(word)
    return words


def word_metrics(rows, name):
    candidates = []
    for i, r in enumerate(rows):
        for end, target in word_positions(r["turns"], TURN_SEPARATOR):
            candidates.append((i, end, target))
    rng = random.Random(SEED)
    picked = rng.sample(candidates, min(N_WORD_POSITIONS, len(candidates)))
    logger.info(f"{name}: {len(candidates):,} word positions, scoring {len(picked):,}")
    contexts, meta = [], []
    for i, end, target in picked:
        r = rows[i]
        p = prefix_ids(persona_for_mode(r["persona"], PERSONA_MODE))
        c = tokenizer(join_turns(r["turns"], TURN_SEPARATOR)[:end], add_special_tokens=False)["input_ids"]
        room = MAX_LENGTH - MAX_WORD_TOKENS - len(p)
        contexts.append(p + c[-room:])
        meta.append((r["persona"], target, join_turns(r["turns"], TURN_SEPARATOR)[max(0, end - 80):end]))
    order = sorted(range(len(contexts)), key=lambda k: len(contexts[k]), reverse=True)
    predictions = [None] * len(contexts)
    for s in range(0, len(order), WORD_BATCH_SIZE):
        idx = order[s:s + WORD_BATCH_SIZE]
        for k, words in zip(idx, predict_words([contexts[k] for k in idx])):
            predictions[k] = words
    totals = defaultdict(lambda: [0, 0, 0, 0.0, 0, 0])
    sample_rows = []
    for (persona, target, tail), words in zip(meta, predictions):
        rank = words.index(target) + 1 if target in words else 0
        chars = len(target) + 1
        for key in ("all", persona):
            t = totals[key]
            t[0] += 1
            t[1] += int(rank == 1)
            t[2] += int(rank > 0)
            t[3] += 1 / rank if rank else 0.0
            t[4] += chars if rank else 0
            t[5] += chars
        sample_rows.append({"persona": persona, "context_end": tail, "target": target,
                            "suggestions": " | ".join(words), "rank": rank})
    with open(os.path.join(OUT_DIR, f"word_predictions_{name}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["persona", "context_end", "target", "suggestions", "rank"])
        w.writeheader()
        w.writerows(sample_rows)
    out = {}
    for key, (n, h1, hk, rr, saved, chars) in totals.items():
        out[key] = {"words": n, "word_top1": round(100 * h1 / max(n, 1), 3),
                    f"word_top{TOP_K}": round(100 * hk / max(n, 1), 3),
                    "word_mrr": round(rr / max(n, 1), 4),
                    "keystroke_savings": round(100 * saved / max(chars, 1), 3)}
    return out


def append_summary(row):
    path = os.path.join(RESULTS_DIR, "all_results.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


logger.info(f"Evaluating {MODEL_PATH} | persona mode: {PERSONA_MODE} | turn separator: {TURN_SEPARATOR}")
for path in TEST_FILES:
    if not os.path.exists(path):
        logger.warning(f"{path} not found, skipped")
        continue
    name = os.path.splitext(os.path.basename(path))[0]
    rows = read_jsonl(path)
    t0 = time.perf_counter()
    token_rows = rows if MAX_TOKEN_EXAMPLES <= 0 or len(rows) <= MAX_TOKEN_EXAMPLES else \
        random.Random(SEED).sample(rows, MAX_TOKEN_EXAMPLES)
    tok = token_metrics(token_rows)
    word = word_metrics(rows, name)
    by_persona = {k: {**tok.get(k, {}), **word.get(k, {})} for k in sorted(set(tok) | set(word)) if k != "all"}
    report = {"run": RUN_NAME, "model_path": MODEL_PATH, "persona_mode": PERSONA_MODE, "turn_separator": TURN_SEPARATOR,
              "test_set": name, "minutes": round((time.perf_counter() - t0) / 60, 2),
              "token": tok["all"], "word": word.get("all", {}), "by_persona": by_persona}
    with open(os.path.join(OUT_DIR, f"metrics_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    w = word.get("all", {})
    append_summary({"run": RUN_NAME, "persona_mode": PERSONA_MODE, "test_set": name, **tok["all"],
                    **{k: w.get(k) for k in ("words", "word_top1", "word_mrr", "keystroke_savings")},
                    "word_top3": w.get(f"word_top{TOP_K}")})
    logger.info(f"{name} | tokens: {json.dumps(tok['all'])}")
    logger.info(f"{name} | words: {json.dumps(w)}")
