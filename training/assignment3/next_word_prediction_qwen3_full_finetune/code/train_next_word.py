import os
import sys
import json
import math
import time
import random
import datetime
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

from nwp_common import persona_prefix, join_turns

RUN_NAME = os.getenv("RUN_NAME", "0.6B")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B-Base")
DATA_DIR = os.getenv("DATA_DIR", "data_nwp")
TRAIN_FILE = os.getenv("TRAIN_FILE", os.path.join(DATA_DIR, "train.jsonl"))
VAL_FILE = os.getenv("VAL_FILE", os.path.join(DATA_DIR, "val.jsonl"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", f"runs/run_{RUN_NAME}")
USE_PERSONA = os.getenv("USE_PERSONA", "1") == "1"
PERSONA_DROPOUT = float(os.getenv("PERSONA_DROPOUT", "0.15"))
TURN_SEPARATOR = os.getenv("TURN_SEPARATOR", "newline")
MAX_TRAIN_EXAMPLES = int(os.getenv("MAX_TRAIN_EXAMPLES", "0"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "1024"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "8"))
GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", "4"))
EPOCHS = int(os.getenv("EPOCHS", "1"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-5"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.03"))
MAX_GRAD_NORM = float(os.getenv("MAX_GRAD_NORM", "1.0"))
SEED = int(os.getenv("SEED", "42"))
LOG_EVERY = int(os.getenv("LOG_EVERY", "20"))
EVAL_EVERY = int(os.getenv("EVAL_EVERY", "1000"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(os.path.join(OUTPUT_DIR, "training.log"))],
)
logger = logging.getLogger(__name__)

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else (
    torch.float16 if DEVICE == "cuda" else torch.float32
)

config = {k: v for k, v in globals().items()
          if k.isupper() and isinstance(v, (str, int, float, bool, list))}
config["AMP_DTYPE"] = str(AMP_DTYPE)
logger.info(f"Run {RUN_NAME} | full fine-tuning of {MODEL_NAME} | device {DEVICE} ({AMP_DTYPE} autocast)")
logger.info(json.dumps(config, indent=2))


def read_jsonl(path, limit=0):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if limit > 0 and len(rows) > limit:
        rows = random.Random(SEED).sample(rows, limit)
    logger.info(f"Loaded {len(rows):,} examples from {path}")
    return rows


class TextDataset(Dataset):
    def __init__(self, rows, tokenizer, dropout=0.0):
        rng = random.Random(SEED + 1)
        prefixes, texts = [], []
        self.personas, self.dropped_persona = [], 0
        for r in rows:
            persona = r["persona"]
            if dropout > 0 and persona != "none" and rng.random() < dropout:
                persona = "none"
                self.dropped_persona += 1
            prefixes.append(persona_prefix(persona) if USE_PERSONA else "")
            texts.append(join_turns(r["turns"], TURN_SEPARATOR) + tokenizer.eos_token)
            self.personas.append(persona)
        self.samples, self.truncated, self.all_lengths = [], 0, []
        cache = {}
        for start in range(0, len(texts), 10000):
            t_ids = tokenizer(texts[start:start + 10000], add_special_tokens=False)["input_ids"]
            for p, t in zip(prefixes[start:start + 10000], t_ids):
                if p not in cache:
                    cache[p] = tokenizer(p, add_special_tokens=False)["input_ids"] if p else []
                ids = cache[p] + t
                self.all_lengths.append(len(ids))
                if len(ids) > MAX_LENGTH:
                    ids = ids[:MAX_LENGTH]
                    self.truncated += 1
                self.samples.append((ids, len(cache[p])))
        self.lengths = [len(ids) for ids, _ in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_collate(pad_id):
    def collate(batch):
        max_len = int(math.ceil(max(len(ids) for ids, _ in batch) / 8) * 8)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, (ids, prefix_len) in enumerate(batch):
            t = torch.tensor(ids, dtype=torch.long)
            input_ids[i, :len(ids)] = t
            attention_mask[i, :len(ids)] = 1
            labels[i, prefix_len:len(ids)] = t[prefix_len:]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return collate


def length_grouped_batches(lengths, batch_size, seed, shuffle=True):
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    if shuffle:
        rng.shuffle(idx)
    mega = batch_size * 50
    batches = []
    for s in range(0, len(idx), mega):
        chunk = sorted(idx[s:s + mega], key=lambda k: lengths[k], reverse=True)
        batches += [chunk[b:b + batch_size] for b in range(0, len(chunk), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def length_report(name, ds):
    lengths = sorted(ds.all_lengths)
    if not lengths:
        return
    pick = lambda q: lengths[min(len(lengths) - 1, int(q * len(lengths)))]
    logger.info(f"{name} token lengths: p50 {pick(0.5)} | p90 {pick(0.9)} | p99 {pick(0.99)} | max {lengths[-1]} | "
                f"truncated to {MAX_LENGTH}: {ds.truncated:,} ({ds.truncated / len(lengths):.2%}) | "
                f"total tokens {sum(ds.lengths):,}")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
collate = make_collate(tokenizer.pad_token_id)

train_rows = read_jsonl(TRAIN_FILE, MAX_TRAIN_EXAMPLES)
train_ds = TextDataset(train_rows, tokenizer, PERSONA_DROPOUT if USE_PERSONA else 0.0)
sources = {}
for r in train_rows:
    sources[r["source"]] = sources.get(r["source"], 0) + 1
logger.info(f"Training examples: {len(train_ds):,} | sources: {sources} | "
            f"persona replaced by 'none' (dropout): {train_ds.dropped_persona:,}")
length_report("Training", train_ds)

val_rows = read_jsonl(VAL_FILE)
val_sets = {}
for src in sorted({r["source"] for r in val_rows}):
    val_sets[src] = TextDataset([r for r in val_rows if r["source"] == src], tokenizer)
logger.info("Validation sets: " + ", ".join(f"{k}={len(v):,}" for k, v in val_sets.items()))

try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.to(DEVICE)
model.train()
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Parameters: {total_params:,} | trainable: {trainable_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)")

no_decay = ["bias", "norm"]
optimizer = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
     "weight_decay": WEIGHT_DECAY},
    {"params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)],
     "weight_decay": 0.0},
], lr=LEARNING_RATE, foreach=False)

batches_per_epoch = math.ceil(len(train_ds) / BATCH_SIZE)
steps_per_epoch = math.ceil(batches_per_epoch / GRAD_ACCUM_STEPS)
total_steps = steps_per_epoch * EPOCHS
warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scaler = torch.amp.GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16))
logger.info(f"Batches per epoch: {batches_per_epoch:,} | optimizer steps: {total_steps:,} | warmup: {warmup_steps:,} | "
            f"effective batch {BATCH_SIZE * GRAD_ACCUM_STEPS}")


def autocast():
    return torch.amp.autocast(device_type="cuda" if DEVICE == "cuda" else "cpu", dtype=AMP_DTYPE,
                              enabled=AMP_DTYPE != torch.float32)


@torch.no_grad()
def validation_loss(ds):
    model.eval()
    loader = DataLoader(ds, batch_sampler=length_grouped_batches(ds.lengths, EVAL_BATCH_SIZE, SEED, shuffle=False),
                        collate_fn=collate, num_workers=NUM_WORKERS)
    total, tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with autocast():
            out = model(**batch)
        n = int((batch["labels"][:, 1:] != -100).sum().item())
        total += float(out.loss.float().item()) * n
        tokens += n
    model.train()
    return total, tokens


def run_validation(step):
    result = {"step": step}
    grand, grand_tokens = 0.0, 0
    for name, ds in val_sets.items():
        if len(ds):
            total, tokens = validation_loss(ds)
            grand, grand_tokens = grand + total, grand_tokens + tokens
            result[name] = round(total / max(tokens, 1), 5)
            result[f"{name}_ppl"] = round(math.exp(min(result[name], 20)), 3)
    result["all"] = round(grand / max(grand_tokens, 1), 5)
    logger.info("VALIDATION | " + " | ".join(f"{k}: {v}" for k, v in result.items()))
    return result


def save_model(path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def peak_vram_gb():
    return torch.cuda.max_memory_allocated() / 1024 ** 3 if DEVICE == "cuda" else 0.0


history = [run_validation(0)]
best_loss = float("inf")

global_step = 0
window_loss, window_batches, window_tokens = 0.0, 0, 0
start_time = time.perf_counter()
last_log_time = start_time
optimizer.zero_grad(set_to_none=True)

for epoch in range(EPOCHS):
    batches = length_grouped_batches(train_ds.lengths, BATCH_SIZE, SEED + epoch, shuffle=True)
    loader = DataLoader(train_ds, batch_sampler=batches, collate_fn=collate, num_workers=NUM_WORKERS,
                        pin_memory=DEVICE == "cuda")
    for step, batch in enumerate(loader):
        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
        with autocast():
            out = model(**batch)
            loss = out.loss / GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()
        window_loss += float(out.loss.detach().float().item())
        window_batches += 1
        window_tokens += int(batch["attention_mask"].sum().item())

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(batches):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % LOG_EVERY == 0:
                now = time.perf_counter()
                elapsed = now - start_time
                eta = datetime.timedelta(seconds=int((total_steps - global_step) * elapsed / global_step))
                logger.info(f"Epoch {epoch + 1}/{EPOCHS} | Step {global_step}/{total_steps} | "
                            f"Loss: {window_loss / max(window_batches, 1):.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
                            f"Tokens/s: {window_tokens / max(now - last_log_time, 1e-9):.0f} | "
                            f"Peak VRAM: {peak_vram_gb():.1f} GB | ETA: {eta}")
                last_log_time = now
                window_loss, window_batches, window_tokens = 0.0, 0, 0

            if global_step % EVAL_EVERY == 0:
                history.append(run_validation(global_step))
                if history[-1]["all"] < best_loss:
                    best_loss = history[-1]["all"]
                    save_model(os.path.join(OUTPUT_DIR, "best"))
                    logger.info(f"New best validation loss {best_loss:.5f}, saved to {OUTPUT_DIR}/best")

train_minutes = (time.perf_counter() - start_time) / 60
if history[-1]["step"] != global_step:
    history.append(run_validation(global_step))
if history[-1]["all"] < best_loss:
    best_loss = history[-1]["all"]
    save_model(os.path.join(OUTPUT_DIR, "best"))
save_model(os.path.join(OUTPUT_DIR, "final"))

summary = {
    "run": RUN_NAME,
    "model": MODEL_NAME,
    "train_examples": len(train_ds),
    "train_tokens": sum(train_ds.lengths),
    "truncated_examples": train_ds.truncated,
    "persona_dropout_examples": train_ds.dropped_persona,
    "optimizer_steps": global_step,
    "total_params": total_params,
    "trainable_params": trainable_params,
    "training_minutes": round(train_minutes, 2),
    "peak_vram_gb": round(peak_vram_gb(), 2),
    "best_val_loss": best_loss,
    "val_history": history,
    "config": config,
}
with open(os.path.join(OUTPUT_DIR, "train_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
logger.info(f"Training finished in {train_minutes:.2f} minutes | peak VRAM {peak_vram_gb():.1f} GB | "
            f"best validation loss {best_loss:.5f}")
