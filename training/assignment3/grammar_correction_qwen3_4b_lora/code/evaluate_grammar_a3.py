import os
import sys
import csv
import json
import math
import time
import difflib
import logging
from collections import Counter

import pandas as pd

from prompt_a3 import build_prompt

RUN_NAME = os.getenv("RUN_NAME", "D")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-4B")
ADAPTER = os.getenv("ADAPTER", f"runs/run_{RUN_NAME}/best")
USE_PERSONA = os.getenv("USE_PERSONA", "1") == "1"
TEST_FILES = [f.strip() for f in os.getenv(
    "TEST_FILES", "data_a3/test_indictalk.csv,data_a3/test_a2.csv,data_a3/test_a2_original.csv").split(",")
    if f.strip()]
LIMIT = int(os.getenv("LIMIT", "0"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
USE_SIMILARITY = os.getenv("USE_SIMILARITY", "0") == "1"
SIM_MODEL = os.getenv("SIM_MODEL", "sentence-transformers/LaBSE")
MIN_GROUP = int(os.getenv("MIN_GROUP", "20"))
SEED = int(os.getenv("SEED", "42"))

OUT_DIR = os.path.join(RESULTS_DIR, RUN_NAME)
os.makedirs(OUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(OUT_DIR, "evaluation.log"))],
)
logger = logging.getLogger(__name__)


def ws(text):
    return " ".join(str(text).split())


def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def gleu(srcs, hyps, refs, order=4):
    stats = [0] * (2 + 2 * order)
    for s, h, r in zip(srcs, hyps, refs):
        s, h, r = s.split(), h.split(), r.split()
        stats[0] += len(h)
        stats[1] += len(r)
        for n in range(1, order + 1):
            h_n, s_n, r_n = ngram_counts(h, n), ngram_counts(s, n), ngram_counts(r, n)
            s_not_r = Counter({k: v for k, v in s_n.items() if k not in r_n})
            stats[2 * n] += max(sum((h_n & r_n).values()) - sum((h_n & s_not_r).values()), 0)
            stats[2 * n + 1] += max(len(h) + 1 - n, 0)
    if any(x == 0 for x in stats):
        return 0.0
    c, r = stats[0], stats[1]
    log_prec = sum(math.log(stats[2 * n] / stats[2 * n + 1]) for n in range(1, order + 1)) / order
    return 100 * math.exp(min(0.0, 1 - r / c) + log_prec)


def edit_set(src_tokens, tgt_tokens):
    sm = difflib.SequenceMatcher(a=src_tokens, b=tgt_tokens, autojunk=False)
    return {(i1, i2, tuple(tgt_tokens[j1:j2])) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"}


def edit_prf(srcs, hyps, refs, beta=0.5):
    tp = fp = fn = 0
    for s, h, r in zip(srcs, hyps, refs):
        s_t = s.split()
        h_e, r_e = edit_set(s_t, h.split()), edit_set(s_t, r.split())
        tp += len(h_e & r_e)
        fp += len(h_e - r_e)
        fn += len(r_e - h_e)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    b2 = beta * beta
    f = (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) else 0.0
    return 100 * p, 100 * r, 100 * f


def text_metrics(hyps, refs):
    try:
        import sacrebleu
        return {
            "chrf": sacrebleu.corpus_chrf(hyps, [refs]).score,
            "chrf_pp": sacrebleu.metrics.CHRF(word_order=2).corpus_score(hyps, [refs]).score,
            "bleu": sacrebleu.corpus_bleu(hyps, [refs]).score,
        }
    except ImportError:
        logger.warning("sacrebleu is not installed: chrF, chrF++ and BLEU are skipped (pip install sacrebleu)")
        return {"chrf": None, "chrf_pp": None, "bleu": None}


def compute_metrics(srcs, hyps, refs):
    srcs, hyps, refs = [ws(x) for x in srcs], [ws(x) for x in hyps], [ws(x) for x in refs]
    n = len(refs)
    clean = [i for i in range(n) if srcs[i] == refs[i]]
    noisy = [i for i in range(n) if srcs[i] != refs[i]]
    p, r, f = edit_prf(srcs, hyps, refs)
    m = {
        "n": n,
        "n_noisy_inputs": len(noisy),
        "n_clean_inputs": len(clean),
        "exact_match": 100 * sum(h == t for h, t in zip(hyps, refs)) / max(n, 1),
        "exact_match_noisy": 100 * sum(hyps[i] == refs[i] for i in noisy) / max(len(noisy), 1) if noisy else None,
        "over_correction_rate": 100 * sum(hyps[i] != srcs[i] for i in clean) / len(clean) if clean else None,
        "gleu": gleu(srcs, hyps, refs),
        "edit_precision": p,
        "edit_recall": r,
        "edit_f0_5": f,
    }
    m.update(text_metrics(hyps, refs))
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}


def group_metrics(df, key):
    out = {}
    if key == "tags":
        groups = Counter(t for tags in df["tags"] for t in str(tags).split(";") if t)
        names = [g for g, c in groups.items() if c >= MIN_GROUP]
        for g in sorted(names):
            part = df[df["tags"].astype(str).str.split(";").apply(lambda ts: g in ts)]
            out[g] = compute_metrics(part["incorrect_text"].tolist(), part["prediction"].tolist(),
                                     part["correct_text"].tolist())
    else:
        for g, part in df.groupby(key):
            if len(part) >= MIN_GROUP:
                out[str(g)] = compute_metrics(part["incorrect_text"].tolist(), part["prediction"].tolist(),
                                              part["correct_text"].tolist())
    return out


def load_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (
        torch.float16 if device == "cuda" else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=dtype)
    if ADAPTER:
        from peft import PeftModel
        logger.info(f"Loading adapter {ADAPTER}")
        model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload()
    else:
        logger.info("No adapter: evaluating the untuned base model")
    model.to(device).eval()
    eos_ids = {tokenizer.eos_token_id}
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            eos_ids.add(tid)
    return model, tokenizer, device, sorted(i for i in eos_ids if i is not None)


def generate(model, tokenizer, device, eos_ids, prompts):
    import torch
    order = sorted(range(len(prompts)), key=lambda k: len(prompts[k]))
    raw = [""] * len(prompts)
    hit_limit = [False] * len(prompts)
    for b in range(0, len(order), BATCH_SIZE):
        idx = order[b:b + BATCH_SIZE]
        enc = tokenizer([prompts[k] for k in idx], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        limit = min(MAX_NEW_TOKENS, int(enc["attention_mask"].sum(1).max().item()) + 32)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=limit, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id)
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for row, (k, t) in enumerate(zip(idx, texts)):
            raw[k] = t
            toks = new_tokens[row].tolist()
            hit_limit[k] = len(toks) >= limit and not any(x in eos_ids for x in toks)
        if (b // BATCH_SIZE) % 10 == 0:
            logger.info(f"  generated {min(b + BATCH_SIZE, len(order)):,}/{len(order):,}")
    return raw, hit_limit


def first_line(text):
    line = text.strip().split("\n")[0].strip()
    return line.strip('"').strip("'").strip()


def similarity(hyps, refs, srcs):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers is not installed: similarity skipped")
        return {}
    m = SentenceTransformer(SIM_MODEL)
    e_h = m.encode(hyps, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    e_r = m.encode(refs, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    e_s = m.encode(srcs, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    return {"similarity_prediction_vs_reference": round(float((e_h * e_r).sum(1).mean()), 4),
            "similarity_input_vs_reference": round(float((e_s * e_r).sum(1).mean()), 4)}


SUMMARY_FIELDS = ["run", "system", "test_set", "n", "exact_match", "exact_match_noisy", "over_correction_rate",
                  "chrf", "chrf_pp", "gleu", "edit_precision", "edit_recall", "edit_f0_5", "bleu",
                  "similarity_prediction_vs_reference"]


def append_summary(row):
    path = os.path.join(RESULTS_DIR, "all_results.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


logger.info(f"Run {RUN_NAME} | adapter: {ADAPTER or 'none'} | persona in prompt: {USE_PERSONA}")
model, tokenizer, device, eos_ids = load_model()

for path in TEST_FILES:
    if not os.path.exists(path):
        logger.warning(f"Skipping missing test file {path}")
        continue
    name = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path, keep_default_na=False)
    for col, default in (("persona", "none"), ("source", name), ("tags", "")):
        if col not in df.columns:
            df[col] = default
    if LIMIT > 0 and len(df) > LIMIT:
        df = df.sample(n=LIMIT, random_state=SEED).reset_index(drop=True)
    logger.info(f"Evaluating {name}: {len(df):,} examples")

    t0 = time.perf_counter()
    prompts = [build_prompt(i, p, USE_PERSONA) for i, p in zip(df["incorrect_text"], df["persona"])]
    df["raw_output"], df["hit_token_limit"] = generate(model, tokenizer, device, eos_ids, prompts)
    df["prediction"] = df["raw_output"].map(first_line)
    n_hit = int(df["hit_token_limit"].sum())
    if n_hit:
        logger.warning(f"{n_hit:,} of {len(df):,} outputs stopped at the token limit without an end token")
    gen_minutes = (time.perf_counter() - t0) / 60

    srcs, hyps, refs = df["incorrect_text"].tolist(), df["prediction"].tolist(), df["correct_text"].tolist()
    model_m = compute_metrics(srcs, hyps, refs)
    copy_m = compute_metrics(srcs, srcs, refs)
    if USE_SIMILARITY:
        model_m.update(similarity(hyps, refs, srcs))
        copy_m["similarity_prediction_vs_reference"] = model_m.get("similarity_input_vs_reference")

    df["exact"] = [ws(h) == ws(r) for h, r in zip(hyps, refs)]
    df.to_csv(os.path.join(OUT_DIR, f"predictions_{name}.csv"), index=False)
    report = {
        "run": RUN_NAME, "test_set": name, "adapter": ADAPTER or None, "base_model": BASE_MODEL,
        "persona_in_prompt": USE_PERSONA, "generation_minutes": round(gen_minutes, 2),
        "outputs_hit_token_limit": n_hit,
        "model": model_m, "copy_input_baseline": copy_m,
        "by_persona": group_metrics(df, "persona"), "by_error_type": group_metrics(df, "tags"),
    }
    with open(os.path.join(OUT_DIR, f"metrics_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    append_summary({"run": RUN_NAME, "system": "model", "test_set": name, **model_m})
    append_summary({"run": RUN_NAME, "system": "copy_input", "test_set": name, **copy_m})
    logger.info(f"{name} | model: " + json.dumps(model_m))
    logger.info(f"{name} | copy-input baseline: " + json.dumps(copy_m))

logger.info(f"Done. Results in {OUT_DIR}/ and {RESULTS_DIR}/all_results.csv")
