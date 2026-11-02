import gc
import json
import math
import os
import shutil
import time
from datetime import timedelta
from pathlib import Path

import psutil
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)


MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "Qwen/Qwen3-0.6B-Base",
)

DATASET_NAME = os.getenv(
    "DATASET_NAME",
    "theguywithblacktie/hinglish-conversations",
)

DATASET_CONFIG = os.getenv(
    "DATASET_CONFIG",
    "large",
)

MAX_LENGTH = int(os.getenv("MAX_LENGTH", "1024"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

GRAD_ACCUM_STEPS = int(
    os.getenv("GRAD_ACCUM_STEPS", "4")
)

EPOCHS = int(os.getenv("EPOCHS", "1"))

LEARNING_RATE = float(
    os.getenv("LEARNING_RATE", "1e-5")
)

WEIGHT_DECAY = float(
    os.getenv("WEIGHT_DECAY", "0.01")
)

WARMUP_RATIO = float(
    os.getenv("WARMUP_RATIO", "0.03")
)

MAX_GRAD_NORM = float(
    os.getenv("MAX_GRAD_NORM", "1.0")
)

VALIDATION_RATIO = float(
    os.getenv("VALIDATION_RATIO", "0.02")
)

SEED = int(
    os.getenv("SEED", "42")
)

SAVE_STEPS = int(
    os.getenv("SAVE_STEPS", "2000")
)

HEARTBEAT_SECONDS = int(
    os.getenv("HEARTBEAT_SECONDS", "60")
)

NUM_WORKERS = int(
    os.getenv("NUM_WORKERS", "4")
)

OUTPUT_DIR = Path(
    os.getenv(
        "OUTPUT_DIR",
        "outputs/qwen3-0.6b-hinglish-next-token",
    )
)

PROCESSED_DIR = Path(
    os.getenv(
        "PROCESSED_DIR",
        "processed_dataset",
    )
)

TOKENIZED_DIR = Path(
    os.getenv(
        "TOKENIZED_DIR",
        "tokenized_dataset",
    )
)

CHECKPOINT_DIR = (
    OUTPUT_DIR / "checkpoints"
)

FINAL_MODEL_DIR = (
    OUTPUT_DIR / "final_model"
)

LOG_FILE = (
    OUTPUT_DIR / "training.log"
)


def ensure_directories():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    TOKENIZED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def now_string():
    return time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def log(message):
    ensure_directories()

    line = (
        f"[{now_string()}] {message}"
    )

    print(
        line,
        flush=True,
    )

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            line + "\n"
        )


def format_duration(seconds):
    seconds = max(
        0,
        int(seconds),
    )
    return str(
        timedelta(seconds=seconds)
    )


def gpu_memory_string():
    if not torch.cuda.is_available():
        return "GPU unavailable"

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 3)
    )

    total = (
        torch.cuda.get_device_properties(0)
        .total_memory
        / (1024 ** 3)
    )

    return (
        f"VRAM {allocated:.2f}/{total:.2f} GB "
        f"(reserved {reserved:.2f} GB)"
    )


def ram_memory_string():
    mem = psutil.virtual_memory()

    used = (
        mem.used
        / (1024 ** 3)
    )

    total = (
        mem.total
        / (1024 ** 3)
    )

    return (
        f"RAM {used:.2f}/{total:.2f} GB"
    )


def count_parameters(model):
    total = sum(
        p.numel()
        for p in model.parameters()
    )
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


def flatten_conversation(example):

    messages = (
        example["input"]
        + example["output"]
    )

    parts = []

    for message in messages:
        content = message.get(
            "content",
            "",
        )

        if content is None:
            continue

        content = str(
            content
        ).strip()

        if content:
            parts.append(
                content
            )

    return {
        "text": " ".join(parts)
    }


def save_jsonl(dataset, path):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for example in dataset:
            f.write(
                json.dumps(
                    {
                        "text": example["text"]
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_or_prepare_raw_data():
    train_json = (
        PROCESSED_DIR
        / "train.jsonl"
    )

    validation_json = (
        PROCESSED_DIR
        / "validation.jsonl"
    )

    test_json = (
        PROCESSED_DIR
        / "test.jsonl"
    )

    if (
        train_json.exists()
        and validation_json.exists()
        and test_json.exists()
    ):
        log(
            "Found processed paragraph dataset on disk."
        )

        train = load_dataset(
            "json",
            data_files=str(
                train_json
            ),
            split="train",
        )

        validation = load_dataset(
            "json",
            data_files=str(
                validation_json
            ),
            split="train",
        )

        test = load_dataset(
            "json",
            data_files=str(
                test_json
            ),
            split="train",
        )

        log(
            f"Loaded local paragraphs: "
            f"train={len(train):,}, "
            f"validation={len(validation):,}, "
            f"test={len(test):,}"
        )

        return (
            train,
            validation,
            test,
        )

    log(
        f"Downloading dataset: "
        f"{DATASET_NAME} / {DATASET_CONFIG}"
    )

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
    )

    raw_train = dataset["train"]
    raw_test = dataset["test"]

    log(
        f"Original train rows: "
        f"{len(raw_train):,}"
    )

    log(
        f"Original test rows: "
        f"{len(raw_test):,}"
    )

    log(
        "Flattening each row into one continuous conversation paragraph..."
    )

    flat_train = raw_train.map(
        flatten_conversation,
        remove_columns=raw_train.column_names,
        desc="Flattening train",
    )

    flat_test = raw_test.map(
        flatten_conversation,
        remove_columns=raw_test.column_names,
        desc="Flattening test",
    )

    split = flat_train.train_test_split(
        test_size=VALIDATION_RATIO,
        seed=SEED,
    )

    train = split["train"]
    validation = split["test"]
    test = flat_test

    log(
        f"Train after split: "
        f"{len(train):,}"
    )

    log(
        f"Validation: "
        f"{len(validation):,}"
    )

    log(
        f"Test (untouched public split): "
        f"{len(test):,}"
    )

    log(
        "Saving processed paragraph datasets..."
    )

    save_jsonl(
        train,
        train_json,
    )

    save_jsonl(
        validation,
        validation_json,
    )

    save_jsonl(
        test,
        test_json,
    )

    log(
        "Processed paragraph datasets saved."
    )

    return (
        train,
        validation,
        test,
    )


def tokenize_batch(batch, tokenizer):
    texts = []

    for text in batch["text"]:
        text = text.strip()

        if tokenizer.eos_token:
            text += tokenizer.eos_token

        texts.append(text)

    return tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        add_special_tokens=True,
    )


def load_or_prepare_tokenized(
    tokenizer,
):
    train_path = (
        TOKENIZED_DIR / "train"
    )

    validation_path = (
        TOKENIZED_DIR / "validation"
    )

    test_path = (
        TOKENIZED_DIR / "test"
    )

    if (
        train_path.exists()
        and validation_path.exists()
        and test_path.exists()
    ):
        log(
            "Found tokenized datasets on disk."
        )

        train = load_from_disk(
            str(train_path)
        )

        validation = load_from_disk(
            str(validation_path)
        )

        test = load_from_disk(
            str(test_path)
        )

        return (
            train,
            validation,
            test,
        )

    train, validation, test = (
        load_or_prepare_raw_data()
    )

    log(
        "Tokenizing training set..."
    )

    tokenized_train = train.map(
        lambda batch: tokenize_batch(
            batch,
            tokenizer,
        ),
        batched=True,
        batch_size=1000,
        remove_columns=train.column_names,
        desc="Tokenizing train",
    )

    log(
        "Tokenizing validation set..."
    )

    tokenized_validation = validation.map(
        lambda batch: tokenize_batch(
            batch,
            tokenizer,
        ),
        batched=True,
        batch_size=1000,
        remove_columns=validation.column_names,
        desc="Tokenizing validation",
    )

    log(
        "Tokenizing test set..."
    )

    tokenized_test = test.map(
        lambda batch: tokenize_batch(
            batch,
            tokenizer,
        ),
        batched=True,
        batch_size=1000,
        remove_columns=test.column_names,
        desc="Tokenizing test",
    )

    log(
        "Saving tokenized datasets..."
    )

    tokenized_train.save_to_disk(
        str(train_path)
    )

    tokenized_validation.save_to_disk(
        str(validation_path)
    )

    tokenized_test.save_to_disk(
        str(test_path)
    )

    log(
        "Tokenized datasets saved."
    )

    return (
        tokenized_train,
        tokenized_validation,
        tokenized_test,
    )


def checkpoint_paths():
    if not CHECKPOINT_DIR.exists():
        return []

    paths = []

    for path in CHECKPOINT_DIR.iterdir():
        if not path.is_dir():
            continue

        if not path.name.startswith(
            "checkpoint-"
        ):
            continue

        try:
            step = int(
                path.name.split("-")[-1]
            )
        except ValueError:
            continue

        paths.append(
            (step, path)
        )

    paths.sort(
        key=lambda x: x[0]
    )

    return paths


def latest_checkpoint():
    paths = checkpoint_paths()

    if not paths:
        return None

    return paths[-1][1]


def save_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
    global_step,
    epoch,
    batch_in_epoch,
):
    checkpoint = (
        CHECKPOINT_DIR
        / f"checkpoint-{global_step}"
    )

    if checkpoint.exists():
        shutil.rmtree(
            checkpoint
        )

    checkpoint.mkdir(
        parents=True,
        exist_ok=True,
    )

    log(
        f"Saving checkpoint at step "
        f"{global_step:,}..."
    )

    model.save_pretrained(
        str(checkpoint)
    )

    tokenizer.save_pretrained(
        str(checkpoint)
    )

    torch.save(
        optimizer.state_dict(),
        checkpoint / "optimizer.pt",
    )

    torch.save(
        scheduler.state_dict(),
        checkpoint / "scheduler.pt",
    )

    if scaler.is_enabled():
        torch.save(
            scaler.state_dict(),
            checkpoint / "scaler.pt",
        )

    state = {
        "global_step": global_step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
    }

    with open(
        checkpoint / "trainer_state.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            state,
            f,
            indent=2,
        )

    paths = checkpoint_paths()

    while len(paths) > 2:
        _, old_path = paths.pop(0)

        log(
            f"Deleting old checkpoint: "
            f"{old_path}"
        )

        shutil.rmtree(
            old_path
        )

    log(
        f"Checkpoint saved: {checkpoint}"
    )

    return checkpoint


def load_checkpoint_state(
    checkpoint,
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
):
    state_file = (
        checkpoint
        / "trainer_state.json"
    )

    with open(
        state_file,
        "r",
        encoding="utf-8",
    ) as f:
        state = json.load(f)

    checkpoint_model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
    )

    model.load_state_dict(
        checkpoint_model.state_dict()
    )

    del checkpoint_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    optimizer.load_state_dict(
        torch.load(
            checkpoint / "optimizer.pt",
            map_location="cpu",
            weights_only=False,
        )
    )

    scheduler.load_state_dict(
        torch.load(
            checkpoint / "scheduler.pt",
            map_location="cpu",
            weights_only=False,
        )
    )

    scaler_file = (
        checkpoint / "scaler.pt"
    )

    if (
        scaler.is_enabled()
        and scaler_file.exists()
    ):
        scaler.load_state_dict(
            torch.load(
                scaler_file,
                map_location="cpu",
                weights_only=False,
            )
        )

    return (
        int(state["global_step"]),
        int(state["epoch"]),
        int(state["batch_in_epoch"]),
    )


@torch.no_grad()
def evaluate_model(
    model,
    dataset,
    collator,
    device,
    batch_size,
    name,
):
    log(
        f"Starting {name} evaluation..."
    )

    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    total_loss = 0.0
    total_tokens = 0

    correct_top1 = 0
    correct_top3 = 0
    measured_tokens = 0

    start = time.perf_counter()

    for batch_idx, batch in enumerate(
        loader,
        start=1,
    ):
        input_ids = (
            batch["input_ids"]
            .to(
                device,
                non_blocking=True,
            )
        )

        attention_mask = (
            batch["attention_mask"]
            .to(
                device,
                non_blocking=True,
            )
        )

        labels = (
            batch["labels"]
            .to(
                device,
                non_blocking=True,
            )
        )

        with torch.autocast(
            device_type="cuda",
            dtype=AMP_DTYPE,
            enabled=True,
        ):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        valid_tokens = (
            labels != -100
        ).sum().item()

        total_loss += (
            outputs.loss.item()
            * valid_tokens
        )

        total_tokens += (
            valid_tokens
        )

        shift_logits = (
            outputs.logits[:, :-1, :]
        )

        shift_labels = (
            labels[:, 1:]
        )

        mask = (
            shift_labels != -100
        )

        if mask.any():

            masked_logits = (
                shift_logits[mask]
            )

            masked_labels = (
                shift_labels[mask]
            )

            top1 = (
                masked_logits
                .argmax(dim=-1)
            )

            top3 = (
                masked_logits.topk(
                    k=3,
                    dim=-1,
                ).indices
            )

            correct_top1 += (
                top1
                .eq(masked_labels)
                .sum()
                .item()
            )

            correct_top3 += (
                top3.eq(
                    masked_labels.unsqueeze(-1)
                )
                .any(dim=-1)
                .sum()
                .item()
            )

            measured_tokens += (
                masked_labels.numel()
            )

        if (
            batch_idx % 100 == 0
        ):
            elapsed = (
                time.perf_counter()
                - start
            )

            log(
                f"{name} heartbeat | "
                f"batch {batch_idx:,}/{len(loader):,} | "
                f"tokens {total_tokens:,} | "
                f"{total_tokens / max(elapsed, 1e-6):,.0f} tok/s | "
                f"{gpu_memory_string()} | "
                f"{ram_memory_string()}"
            )

    avg_loss = (
        total_loss
        / max(total_tokens, 1)
    )

    perplexity = (
        math.exp(avg_loss)
        if avg_loss < 20
        else float("inf")
    )

    top1_accuracy = (
        correct_top1
        / max(measured_tokens, 1)
    )

    top3_accuracy = (
        correct_top3
        / max(measured_tokens, 1)
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    log(
        f"{name} COMPLETE | "
        f"loss={avg_loss:.6f} | "
        f"perplexity={perplexity:.4f} | "
        f"top1={top1_accuracy * 100:.2f}% | "
        f"top3={top3_accuracy * 100:.2f}% | "
        f"time={format_duration(elapsed)}"
    )

    model.train()

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "top1_accuracy": top1_accuracy,
        "top3_accuracy": top3_accuracy,
        "tokens": total_tokens,
        "time_seconds": elapsed,
    }


ensure_directories()

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision(
        "high"
    )

log(
    "============================================================"
)

log(
    "HINGLISH QWEN3-0.6B NEXT-TOKEN TRAINING"
)

log(
    "Full-parameter causal language-model fine-tuning"
)

log(
    "NO LoRA | NO n-grams"
)

log(
    "============================================================"
)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for this training job."
    )

DEVICE = torch.device("cuda")

BF16_SUPPORTED = (
    torch.cuda.is_bf16_supported()
)

AMP_DTYPE = (
    torch.bfloat16
    if BF16_SUPPORTED
    else torch.float16
)

USE_SCALER = (
    AMP_DTYPE == torch.float16
)

log(
    f"GPU: {torch.cuda.get_device_name(0)}"
)

log(
    f"BF16 supported: {BF16_SUPPORTED}"
)

log(
    f"AMP dtype: {AMP_DTYPE}"
)

log(
    gpu_memory_string()
)

log(
    ram_memory_string()
)


log(
    "Loading tokenizer..."
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = (
        tokenizer.eos_token
    )

log(
    "Tokenizer loaded."
)


train_dataset, validation_dataset, test_dataset = (
    load_or_prepare_tokenized(
        tokenizer
    )
)

log(
    f"Train examples: "
    f"{len(train_dataset):,}"
)

log(
    f"Validation examples: "
    f"{len(validation_dataset):,}"
)

log(
    f"Test examples: "
    f"{len(test_dataset):,}"
)


collator = (
    DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
)

steps_per_epoch = math.ceil(
    math.ceil(len(train_dataset) / BATCH_SIZE)
    / GRAD_ACCUM_STEPS
)

total_training_steps = (
    steps_per_epoch
    * EPOCHS
)

warmup_steps = max(
    1,
    int(
        total_training_steps
        * WARMUP_RATIO
    ),
)

log(
    f"Batch size: {BATCH_SIZE}"
)

log(
    f"Gradient accumulation: "
    f"{GRAD_ACCUM_STEPS}"
)

log(
    f"Effective batch size: "
    f"{BATCH_SIZE * GRAD_ACCUM_STEPS}"
)

batches_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)

log(
    f"Batches per epoch: "
    f"{batches_per_epoch:,}"
)

log(
    f"Optimizer steps per epoch: "
    f"{steps_per_epoch:,}"
)

log(
    f"Total optimizer steps: "
    f"{total_training_steps:,}"
)

log(
    f"Warmup steps: "
    f"{warmup_steps:,}"
)


log(
    "Loading base Qwen3-0.6B model..."
)

model = (
    AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
    )
)

model.config.use_cache = False

try:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        }
    )
except TypeError:
    model.gradient_checkpointing_enable()

model.train()

model.to(
    DEVICE
)

total_params, trainable_params = (
    count_parameters(model)
)

log(
    f"Total parameters: "
    f"{total_params:,}"
)

log(
    f"Trainable parameters: "
    f"{trainable_params:,}"
)

log(
    f"Trainable percentage: "
    f"{100 * trainable_params / total_params:.2f}%"
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    foreach=False,
)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps,
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_SCALER,
)


resume_checkpoint = (
    latest_checkpoint()
)

global_step = 0
start_epoch = 0
resume_batch = 0

if resume_checkpoint is not None:

    log(
        f"Found checkpoint: "
        f"{resume_checkpoint}"
    )

    try:
        model = (
            AutoModelForCausalLM.from_pretrained(
                str(resume_checkpoint),
            )
        )

        model.config.use_cache = False

        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False
                }
            )
        except TypeError:
            model.gradient_checkpointing_enable()

        model.train()
        model.to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            foreach=False,
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=USE_SCALER,
        )

        global_step, start_epoch, resume_batch = (
            load_checkpoint_state(
                resume_checkpoint,
                model,
                tokenizer,
                optimizer,
                scheduler,
                scaler,
            )
        )

        log(
            f"Resumed from step {global_step:,}, "
            f"epoch {start_epoch}, "
            f"batch {resume_batch}"
        )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to resume checkpoint "
            f"{resume_checkpoint}: {exc}"
        ) from exc

else:
    log(
        "No checkpoint found. "
        "Starting from the base model."
    )


train_start = time.perf_counter()

last_heartbeat = train_start

running_loss = 0.0
running_batches = 0

optimizer.zero_grad(
    set_to_none=True
)

for epoch in range(
    start_epoch,
    EPOCHS,
):

    log(
        f"========== EPOCH {epoch + 1}/{EPOCHS} =========="
    )

    epoch_start = (
        time.perf_counter()
    )

    generator = torch.Generator()
    generator.manual_seed(SEED + epoch)

    epoch_indices = torch.randperm(
        len(train_dataset),
        generator=generator,
    ).tolist()

    epoch_subset = Subset(
        train_dataset,
        epoch_indices,
    )

    train_loader = DataLoader(
        epoch_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    batches_to_skip = (
        resume_batch
        if epoch == start_epoch
        else 0
    )

    if batches_to_skip > 0:
        log(
            f"Resuming epoch {epoch + 1}: "
            f"skipping first {batches_to_skip:,} batches."
        )

    for batch_idx, batch in enumerate(train_loader):

        if batch_idx < batches_to_skip:
            continue

        input_ids = (
            batch["input_ids"]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        attention_mask = (
            batch["attention_mask"]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        labels = (
            batch["labels"]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        with torch.autocast(
            device_type="cuda",
            dtype=AMP_DTYPE,
            enabled=True,
        ):

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            raw_loss = outputs.loss

            loss = (
                raw_loss
                / GRAD_ACCUM_STEPS
            )

        if scaler.is_enabled():
            scaler.scale(
                loss
            ).backward()
        else:
            loss.backward()

        running_loss += (
            raw_loss.item()
        )

        running_batches += 1

        is_last_batch = (
            batch_idx + 1
            == len(train_loader)
        )

        accumulation_boundary = (
            (batch_idx + 1)
            % GRAD_ACCUM_STEPS
            == 0
        )

        if (
            accumulation_boundary
            or is_last_batch
        ):

            if scaler.is_enabled():
                scaler.unscale_(
                    optimizer
                )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )

            if scaler.is_enabled():
                scaler.step(
                    optimizer
                )
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            global_step += 1


            now = time.perf_counter()

            if (
                now - last_heartbeat
                >= HEARTBEAT_SECONDS
            ):

                elapsed = (
                    now - train_start
                )

                steps_per_second = (
                    global_step
                    / max(elapsed, 1e-6)
                )

                remaining_steps = max(
                    total_training_steps
                    - global_step,
                    0,
                )

                eta_seconds = (
                    remaining_steps
                    / max(
                        steps_per_second,
                        1e-9,
                    )
                )

                avg_loss = (
                    running_loss
                    / max(
                        running_batches,
                        1,
                    )
                )

                progress = (
                    global_step
                    / max(
                        total_training_steps,
                        1,
                    )
                    * 100
                )

                log(
                    f"HEARTBEAT | "
                    f"epoch {epoch + 1}/{EPOCHS} | "
                    f"step {global_step:,}/"
                    f"{total_training_steps:,} "
                    f"({progress:.2f}%) | "
                    f"loss {avg_loss:.4f} | "
                    f"lr {scheduler.get_last_lr()[0]:.3e} | "
                    f"{steps_per_second:.3f} steps/s | "
                    f"elapsed {format_duration(elapsed)} | "
                    f"ETA {format_duration(eta_seconds)} | "
                    f"{gpu_memory_string()} | "
                    f"{ram_memory_string()}"
                )

                running_loss = 0.0
                running_batches = 0
                last_heartbeat = now


            if (
                global_step % SAVE_STEPS
                == 0
            ):

                save_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    scaler,
                    global_step,
                    epoch,
                    batch_idx + 1,
                )

    resume_batch = 0

    epoch_elapsed = (
        time.perf_counter()
        - epoch_start
    )

    log(
        f"Epoch {epoch + 1} finished in "
        f"{format_duration(epoch_elapsed)}"
    )


log(
    "Training finished."
)

log(
    "Saving final model..."
)

if FINAL_MODEL_DIR.exists():
    shutil.rmtree(
        FINAL_MODEL_DIR
    )

FINAL_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

model.save_pretrained(
    str(FINAL_MODEL_DIR)
)

tokenizer.save_pretrained(
    str(FINAL_MODEL_DIR)
)

log(
    f"Final model saved to: "
    f"{FINAL_MODEL_DIR}"
)


validation_metrics = (
    evaluate_model(
        model=model,
        dataset=validation_dataset,
        collator=collator,
        device=DEVICE,
        batch_size=max(
            1,
            BATCH_SIZE,
        ),
        name="VALIDATION",
    )
)

with open(
    OUTPUT_DIR
    / "validation_metrics.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        validation_metrics,
        f,
        indent=2,
    )


test_metrics = (
    evaluate_model(
        model=model,
        dataset=test_dataset,
        collator=collator,
        device=DEVICE,
        batch_size=max(
            1,
            BATCH_SIZE,
        ),
        name="HELD-OUT TEST",
    )
)

with open(
    OUTPUT_DIR
    / "test_metrics.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        test_metrics,
        f,
        indent=2,
    )


total_elapsed = (
    time.perf_counter()
    - train_start
)

log(
    "============================================================"
)

log(
    "PIPELINE COMPLETE"
)

log(
    f"Model: {MODEL_NAME}"
)

log(
    f"Total parameters: {total_params:,}"
)

log(
    f"Trainable parameters: {trainable_params:,}"
)

log(
    f"Training time: "
    f"{format_duration(total_elapsed)}"
)

log(
    f"Validation loss: "
    f"{validation_metrics['loss']:.6f}"
)

log(
    f"Validation perplexity: "
    f"{validation_metrics['perplexity']:.4f}"
)

log(
    f"Validation top-1: "
    f"{validation_metrics['top1_accuracy'] * 100:.2f}%"
)

log(
    f"Validation top-3: "
    f"{validation_metrics['top3_accuracy'] * 100:.2f}%"
)

log(
    f"Test loss: "
    f"{test_metrics['loss']:.6f}"
)

log(
    f"Test perplexity: "
    f"{test_metrics['perplexity']:.4f}"
)

log(
    f"Test top-1: "
    f"{test_metrics['top1_accuracy'] * 100:.2f}%"
)

log(
    f"Test top-3: "
    f"{test_metrics['top3_accuracy'] * 100:.2f}%"
)

log(
    f"Final model: "
    f"{FINAL_MODEL_DIR}"
)

log(
    gpu_memory_string()
)

log(
    ram_memory_string()
)

log(
    "============================================================"
)
