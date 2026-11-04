import os
import sys
import json
import math
import time
import random
import datetime
import logging

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from prompt_a3 import build_prompt

RUN_NAME = os.getenv("RUN_NAME", "D")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-4B")
INIT_ADAPTER = os.getenv("INIT_ADAPTER", "")
TRAIN_FILES = [f.strip() for f in os.getenv(
    "TRAIN_FILES", "data_a3/train_indictalk.csv,data_a3/train_a2_replay.csv").split(",") if f.strip()]
VAL_FILE = os.getenv("VAL_FILE", "data_a3/val.csv")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", f"runs/run_{RUN_NAME}")
USE_PERSONA = os.getenv("USE_PERSONA", "1") == "1"
MAX_TRAIN_EXAMPLES = int(os.getenv("MAX_TRAIN_EXAMPLES", "0"))

MAX_LENGTH = int(os.getenv("MAX_LENGTH", "320"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "32"))
GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", "4"))
EPOCHS = int(os.getenv("EPOCHS", "1"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.05"))
MAX_GRAD_NORM = float(os.getenv("MAX_GRAD_NORM", "1.0"))
LORA_R = int(os.getenv("LORA_R", "32"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "64"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
SEED = int(os.getenv("SEED", "42"))
LOG_EVERY = int(os.getenv("LOG_EVERY", "10"))
EVAL_EVERY = int(os.getenv("EVAL_EVERY", "500"))
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else (
    torch.float16 if DEVICE == "cuda" else torch.float32
)

config = {k: v for k, v in globals().items()
          if k.isupper() and isinstance(v, (str, int, float, bool, list))}
config["DTYPE"] = str(DTYPE)
logger.info(f"Run {RUN_NAME} | device {DEVICE} ({DTYPE})")
logger.info(json.dumps(config, indent=2))


def read_pairs(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p, keep_default_na=False)
        if "persona" not in df.columns:
            df["persona"] = "none"
        if "source" not in df.columns:
            df["source"] = os.path.basename(p)
        frames.append(df)
        logger.info(f"Loaded {len(df):,} pairs from {p}")
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["incorrect_text"].astype(str).str.len() > 0) & (df["correct_text"].astype(str).str.len() > 0)]
    return df.reset_index(drop=True)


class PairDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        prompts = [build_prompt(i, p, USE_PERSONA) for i, p in zip(df["incorrect_text"], df["persona"])]
        responses = [f"{c}{tokenizer.eos_token}" for c in df["correct_text"]]
        self.samples, self.dropped, self.all_lengths = [], 0, []
        for start in range(0, len(prompts), 10000):
            p_ids = tokenizer(prompts[start:start + 10000], add_special_tokens=False)["input_ids"]
            r_ids = tokenizer(responses[start:start + 10000], add_special_tokens=False)["input_ids"]
            for p, r in zip(p_ids, r_ids):
                self.all_lengths.append(len(p) + len(r))
                if len(p) + len(r) > max_length:
                    self.dropped += 1
                    continue
                self.samples.append((p + r, len(p)))
        self.lengths = [len(ids) for ids, _ in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_collate(pad_id):
    def collate(batch):
        max_len = max(len(ids) for ids, _ in batch)
        max_len = int(math.ceil(max_len / 8) * 8)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, (ids, prompt_len) in enumerate(batch):
            n = len(ids)
            t = torch.tensor(ids, dtype=torch.long)
            input_ids[i, :n] = t
            attention_mask[i, :n] = 1
            labels[i, prompt_len:n] = t[prompt_len:]
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
    share = ds.dropped / len(lengths)
    logger.info(f"{name} token lengths (prompt + answer): p50 {pick(0.5)} | p90 {pick(0.9)} | p99 {pick(0.99)} | "
                f"max {lengths[-1]} | over {MAX_LENGTH}: {ds.dropped:,} ({share:.2%})")
    if share > 0.02:
        logger.warning(f"More than 2% of {name} pairs are longer than MAX_LENGTH={MAX_LENGTH} and were dropped. "
                       f"Consider MAX_LENGTH={int(pick(0.995) / 8 + 1) * 8}.")


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
collate = make_collate(tokenizer.pad_token_id)

train_df = read_pairs(TRAIN_FILES)
if MAX_TRAIN_EXAMPLES > 0 and len(train_df) > MAX_TRAIN_EXAMPLES:
    train_df = train_df.sample(n=MAX_TRAIN_EXAMPLES, random_state=SEED).reset_index(drop=True)
train_ds = PairDataset(train_df, tokenizer, MAX_LENGTH)
logger.info(f"Training pairs kept: {len(train_ds):,} | dropped (> {MAX_LENGTH} tokens): {train_ds.dropped:,} | "
            f"sources: {train_df['source'].value_counts().to_dict()} | "
            f"unchanged inputs: {(train_df['incorrect_text'] == train_df['correct_text']).mean():.1%}")

length_report("Training", train_ds)

val_df = read_pairs([VAL_FILE])
val_sets = {"all": PairDataset(val_df, tokenizer, MAX_LENGTH)}
for src, part in val_df.groupby("source"):
    val_sets[src] = PairDataset(part.reset_index(drop=True), tokenizer, MAX_LENGTH)
length_report("Validation", val_sets["all"])
logger.info("Validation sets: " + ", ".join(f"{k}={len(v):,}" for k, v in val_sets.items()))

try:
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=DTYPE)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=DTYPE)
model.to(DEVICE)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()

if INIT_ADAPTER:
    logger.info(f"Continuing training from adapter: {INIT_ADAPTER}")
    model = PeftModel.from_pretrained(model, INIT_ADAPTER, is_trainable=True)
else:
    logger.info("Starting a new LoRA adapter from the base model")
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
trainable_params, total_params = model.get_nb_trainable_parameters()
logger.info(f"Trainable params: {trainable_params:,} || all params: {total_params:,} || "
            f"trainable%: {100 * trainable_params / total_params:.4f}")
model.train()

no_decay = ["bias", "layer_norm", "layernorm", "rmsnorm", "norm.weight"]
optimizer = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n.lower() for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
    {"params": [p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n.lower() for nd in no_decay)], "weight_decay": 0.0},
], lr=LEARNING_RATE)

batches_per_epoch = math.ceil(len(train_ds) / BATCH_SIZE)
steps_per_epoch = math.ceil(batches_per_epoch / GRAD_ACCUM_STEPS)
total_steps = steps_per_epoch * EPOCHS
warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scaler = torch.amp.GradScaler("cuda", enabled=(DTYPE == torch.float16))
logger.info(f"Batches per epoch: {batches_per_epoch:,} | optimizer steps: {total_steps:,} | warmup: {warmup_steps:,}")


def autocast():
    return torch.amp.autocast(device_type="cuda" if DEVICE == "cuda" else "cpu", dtype=DTYPE,
                              enabled=DTYPE != torch.float32)


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
    return total / max(tokens, 1)


def run_validation(step):
    result = {"step": step}
    for name, ds in val_sets.items():
        if len(ds):
            result[name] = round(validation_loss(ds), 5)
    logger.info("VALIDATION | " + " | ".join(f"{k}: {v}" for k, v in result.items()))
    return result


def save_adapter(path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def peak_vram_gb():
    return torch.cuda.max_memory_allocated() / 1024 ** 3 if DEVICE == "cuda" else 0.0


history = [run_validation(0)]
best_loss = history[0]["all"]
save_adapter(os.path.join(OUTPUT_DIR, "best"))

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
                    save_adapter(os.path.join(OUTPUT_DIR, "best"))
                    logger.info(f"New best validation loss {best_loss:.5f}, saved to {OUTPUT_DIR}/best")

train_minutes = (time.perf_counter() - start_time) / 60
if not history or history[-1]["step"] != global_step:
    history.append(run_validation(global_step))
    if history[-1]["all"] < best_loss:
        best_loss = history[-1]["all"]
        save_adapter(os.path.join(OUTPUT_DIR, "best"))
save_adapter(os.path.join(OUTPUT_DIR, "final"))

summary = {
    "run": RUN_NAME,
    "init_adapter": INIT_ADAPTER or None,
    "train_files": TRAIN_FILES,
    "train_pairs": len(train_ds),
    "dropped_long_pairs": train_ds.dropped,
    "optimizer_steps": global_step,
    "trainable_params": trainable_params,
    "total_params": total_params,
    "training_minutes": round(train_minutes, 2),
    "peak_vram_gb": round(peak_vram_gb(), 2),
    "best_val_loss": best_loss,
    "val_history": history,
    "config": config,
}
with open(os.path.join(OUTPUT_DIR, "train_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
logger.info(f"Training finished in {train_minutes:.2f} minutes | peak VRAM {peak_vram_gb():.1f} GB | "
            f"best val loss {best_loss:.5f}")
logger.info(f"Best adapter: {OUTPUT_DIR}/best | final adapter: {OUTPUT_DIR}/final")
