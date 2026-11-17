import os
import sys
import json
import math
import time
import random
import logging

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from sens_common import auroc

MODEL_NAME = os.getenv("MODEL_NAME", "google/muril-base-cased")
RUN_NAME = os.getenv("RUN_NAME", "muril")
DATA_DIR = os.getenv("DATA_DIR", "data_sens")
TRAIN_FILE = os.getenv("TRAIN_FILE", os.path.join(DATA_DIR, "train.jsonl"))
VAL_FILE = os.getenv("VAL_FILE", os.path.join(DATA_DIR, "val.jsonl"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", f"runs/run_{RUN_NAME}")
SEED = int(os.getenv("SEED", "42"))
LR = float(os.getenv("LEARNING_RATE", "2e-5"))
EPOCHS = int(os.getenv("EPOCHS", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM_STEPS", "1"))
EVAL_BATCH = int(os.getenv("EVAL_BATCH_SIZE", "64"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
WARMUP = float(os.getenv("WARMUP_RATIO", "0.06"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
EVAL_EVERY = int(os.getenv("EVAL_EVERY", "0"))          # 0 = once per epoch
USE_BF16 = os.getenv("USE_BF16", "1") == "1"
GRAD_CKPT = os.getenv("GRADIENT_CHECKPOINTING", "0") == "1"
MAX_TRAIN_EXAMPLES = int(os.getenv("MAX_TRAIN_EXAMPLES", "0"))

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(OUTPUT_DIR, "train.log"))],
)
logger = logging.getLogger(__name__)
random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEVICE == "cuda" and USE_BF16 and torch.cuda.is_bf16_supported()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


train_rows = read_jsonl(TRAIN_FILE)
if MAX_TRAIN_EXAMPLES:
    train_rows = train_rows[:MAX_TRAIN_EXAMPLES]
val_rows = read_jsonl(VAL_FILE)
logger.info(f"Model {MODEL_NAME} | run {RUN_NAME} | device {DEVICE} | bf16 autocast {AMP}")
logger.info(f"Train {len(train_rows):,} | val {len(val_rows):,}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
try:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1, dtype=torch.float32)
except TypeError:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1, torch_dtype=torch.float32)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.problem_type = "regression"   # one logit; we apply our own BCE loss
if GRAD_CKPT:
    model.gradient_checkpointing_enable()
model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
logger.info(f"Parameters: {n_params:,} (all trainable, full fine-tuning) | weights fp32")


def collate(batch):
    enc = tokenizer([r["text"] for r in batch], padding=True, truncation=True, max_length=MAX_LENGTH,
                    return_tensors="pt")
    enc["targets"] = torch.tensor([float(r["target"]) for r in batch], dtype=torch.float32)
    return enc


def logits_for(rows):
    model.eval()
    out = []
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["text"]))
    loader = DataLoader([rows[i] for i in order], batch_size=EVAL_BATCH, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=AMP):
                logit = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            out.extend(logit.float().view(-1).tolist())
    res = [0.0] * len(rows)
    for pos, i in enumerate(order):
        res[i] = out[pos]
    model.train()
    return res


def bce(logits, targets, temperature=1.0):
    total = 0.0
    for z, t in zip(logits, targets):
        z = z / temperature
        total += max(z, 0) - z * t + math.log1p(math.exp(-abs(z)))
    return total / max(len(logits), 1)


def evaluate(rows):
    logits = logits_for(rows)
    targets = [r["target"] for r in rows]
    hard = [(1 if r["target"] >= 0.75 else 0, z) for r, z in zip(rows, logits) if r["target"] <= 0.25 or r["target"] >= 0.75]
    by_source = {}
    for src in sorted({r["source"] for r in rows}):
        idx = [k for k, r in enumerate(rows) if r["source"] == src]
        by_source[src] = round(bce([logits[k] for k in idx], [targets[k] for k in idx]), 4)
    return {
        "loss": bce(logits, targets),
        "auroc_safe_vs_offensive": auroc([h for h, _ in hard], [z for _, z in hard]),
        "loss_by_source": by_source,
    }, logits


loader = DataLoader(train_rows, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, drop_last=False)
no_decay = ("bias", "LayerNorm.weight", "layer_norm", "norm.weight")
params = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     "weight_decay": WEIGHT_DECAY},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(params, lr=LR)
total_steps = math.ceil(len(loader) / GRAD_ACCUM) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP), total_steps)
loss_fn = torch.nn.BCEWithLogitsLoss()
logger.info(f"Optimizer steps: {total_steps:,} | LR {LR} | batch {BATCH_SIZE} x {GRAD_ACCUM} | epochs {EPOCHS}")

history, best_loss, best_logits = [], float("inf"), None
step, start = 0, time.time()
if DEVICE == "cuda":
    torch.cuda.reset_peak_memory_stats()


def run_eval(epoch):
    global best_loss, best_logits
    metrics, logits = evaluate(val_rows)
    entry = {"step": step, "epoch": epoch, **metrics}
    history.append(entry)
    logger.info(f"Validation at step {step}: loss {metrics['loss']:.4f} | AUROC {metrics['auroc_safe_vs_offensive']} | "
                f"by source {metrics['loss_by_source']}")
    if metrics["loss"] < best_loss:
        best_loss, best_logits = metrics["loss"], logits
        model.save_pretrained(os.path.join(OUTPUT_DIR, "best"))
        tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "best"))
        logger.info("  new best model saved")


model.train()
for epoch in range(1, EPOCHS + 1):
    running, n = 0.0, 0
    for i, batch in enumerate(loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=AMP):
            logit = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.view(-1)
        loss = loss_fn(logit.float(), batch["targets"]) / GRAD_ACCUM
        loss.backward()
        running += loss.item() * GRAD_ACCUM
        n += 1
        if (i + 1) % GRAD_ACCUM == 0 or i + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                logger.info(f"epoch {epoch} step {step}/{total_steps} | train loss {running / n:.4f} | "
                            f"lr {scheduler.get_last_lr()[0]:.2e}")
                running, n = 0.0, 0
            if EVAL_EVERY and step % EVAL_EVERY == 0:
                run_eval(epoch)
    if not EVAL_EVERY:
        run_eval(epoch)

model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

# Temperature scaling on the validation set, so a score of 80% means about 80% offensive.
val_targets = [r["target"] for r in val_rows]
grid = [round(0.5 + 0.05 * k, 2) for k in range(71)]          # 0.5 .. 4.0
temperature = min(grid, key=lambda t: bce(best_logits, val_targets, t))
calib = {"temperature": temperature, "val_loss_before": round(bce(best_logits, val_targets), 4),
         "val_loss_after": round(bce(best_logits, val_targets, temperature), 4)}
with open(os.path.join(OUTPUT_DIR, "best", "calibration.json"), "w") as f:
    json.dump(calib, f, indent=2)
logger.info(f"Calibration: {calib}")

summary = {
    "run": RUN_NAME, "model": MODEL_NAME, "parameters": n_params, "train_examples": len(train_rows),
    "val_examples": len(val_rows), "learning_rate": LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
    "grad_accum_steps": GRAD_ACCUM, "max_length": MAX_LENGTH, "bf16_autocast": AMP, "weights": "fp32",
    "optimizer_steps": step, "training_minutes": round((time.time() - start) / 60, 1),
    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if DEVICE == "cuda" else None,
    "best_val_loss": round(best_loss, 4), "calibration": calib, "history": history,
}
with open(os.path.join(OUTPUT_DIR, "train_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
logger.info(f"Done in {summary['training_minutes']} min. Best model: {OUTPUT_DIR}/best")
