import os
import math
import time
import gc
import random
import datetime
import logging
import sys
import re
import torch
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("qwen_4b_training.log")
    ]
)
logger = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen3-4B"
OUTPUT_DIR = "./qwen_4b_hinglish_lora"

TRAIN_CSV = "grammar_dataset_train.csv"
TEST_CSV = "grammar_dataset_test.csv"

MAX_LENGTH = 256
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 4
EPOCHS = 1
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else (
    torch.float16 if DEVICE == "cuda" else torch.float32
)

logger.info(f"Configured to run on: {DEVICE} ({DTYPE})")

def extract_texts(msg_list):
    if msg_list is None:
        return []
    extracted = []
    for msg in msg_list:
        try:
            if 'content' in msg:
                extracted.append(str(msg['content']).strip())
        except Exception:
            pass
    return extracted

def introduce_noise(text, error_prob=0.15):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text

    words = text.split()
    if len(words) < 3:
        return text

    new_words = []
    i = 0

    confusions = {
        "their": "there", "there": "their", "they're": "there",
        "your": "ur", "you're": "your", "ur": "your",
        "its": "it's", "it's": "its",
        "then": "than", "than": "then",
        "to": "2", "too": "to", "2": "to",
        "for": "4", "are": "r", "you": "u", "why": "y",
        "and": "nd", "because": "coz", "please": "plz",
        "affect": "effect", "effect": "affect",
        "loose": "lose", "lose": "loose",
        "accept": "except", "except": "accept",
        "ok": "k", "message": "msg", "brother": "bro",
        "in": "on", "on": "at", "at": "in", "into": "in",
        "since": "for", "from": "for",
        "is": "are", "are": "is", "was": "were", "were": "was",
        "has": "have", "have": "has", "had": "have",
        "do": "does", "does": "do", "did": "done", "done": "did",
        "hai": "h", "ha": "hai", "hain": "h", "he": "hai",
        "mein": "m", "me": "mein", "main": "mai", "mai": "m",
        "kya": "k", "ki": "k", "ke": "k", "ka": "k",
        "nahi": "ni", "nhi": "nai", "nai": "ni",
        "kuch": "kch", "kch": "kuch",
        "kaise": "kse", "kese": "kaise", "kaisa": "ksa",
        "kyun": "q", "kyu": "q", "kyunki": "cuz",
        "mujhe": "mje", "tujhe": "tje",
        "bahut": "bht", "bohot": "bht", "bht": "bahut",
        "haan": "hn", "han": "h",
        "yaar": "yr", "yar": "yaar",
        "bhi": "b", "bhee": "bhi",
        "hum": "hm", "ham": "hum",
        "tum": "tm", "tm": "tum",
        "achha": "acha", "acha": "accha", "axaa": "acha",
        "jarur": "zrur", "zaroor": "jarur",
        "theek": "tk", "thik": "tk",
        "matlab": "mtlb", "mtlb": "matlab",
        "waise": "vse", "vese": "waise",
        "jaldi": "jldi", "jldi": "jaldi",
        "koi": "koi", "koe": "koi",
        "bhai": "bhy", "bhaiya": "bhy",
        "raha": "rahi", "rahi": "raha", "rahe": "raha",
        "karta": "karti", "karti": "karte", "karte": "karta",
        "tha": "thi", "thi": "the", "the": "tha",
        "hota": "hoti", "hoti": "hote", "hote": "hota"
    }

    keyboard_typos = {'a':'s', 's':'a', 'e':'w', 'w':'e', 'i':'o', 'o':'i', 'u':'y', 'y':'u', 'n':'m', 'm':'n'}
    articles = ["a", "an", "the"]

    while i < len(words):
        word = words[i]
        action = random.random()

        if action < error_prob * 0.05:
            i += 1
            continue
        elif action < error_prob * 0.15:
            if word.lower() in articles:
                if random.random() < 0.5:
                    i += 1
                    continue
                else:
                    other_articles = [a for a in articles if a != word.lower()]
                    new_words.append(random.choice(other_articles))
                    i += 1
                    continue
        elif action < error_prob * 0.22 and i < len(words) - 1:
            new_words.append(words[i+1])
            new_words.append(word)
            i += 2
            continue
        elif action < error_prob * 0.28:
            new_words.append(word)
            new_words.append(word)
            i += 1
            continue
        elif action < error_prob * 0.45:
            clean_word = word.lower().strip(".,!?")
            if clean_word in confusions:
                new_word = confusions[clean_word]
                if word.istitle():
                    new_word = new_word.title()
                new_words.append(new_word)
            else:
                new_words.append(word)
            i += 1
            continue
        elif action < error_prob * 0.55:
            clean_word = word.strip(".,!?")
            if len(clean_word) > 4 and clean_word.isalpha():
                if clean_word.endswith('s'):
                    new_word = word.replace(clean_word, clean_word[:-1])
                else:
                    new_word = word.replace(clean_word, clean_word + 's')
                new_words.append(new_word)
            else:
                new_words.append(word)
            i += 1
            continue
        elif action < error_prob * 0.65:
            clean_word = word
            if 'sh' in clean_word.lower():
                clean_word = clean_word.replace('sh', 's').replace('Sh', 'S')
            elif 'v' in clean_word.lower():
                clean_word = clean_word.replace('v', 'w').replace('V', 'W')
            elif 'ph' in clean_word.lower():
                clean_word = clean_word.replace('ph', 'f').replace('Ph', 'F')
            elif 'z' in clean_word.lower():
                clean_word = clean_word.replace('z', 'j').replace('Z', 'J')
            new_words.append(clean_word)
            i += 1
            continue
        elif action < error_prob * 0.72:
            if len(word) > 4:
                vowels = "aeiouAEIOU"
                new_word = "".join([c for idx, c in enumerate(word) if idx == 0 or idx == len(word)-1 or c not in vowels])
                new_words.append(new_word if len(new_word) > 2 else word)
            else:
                new_words.append(word)
            i += 1
            continue
        elif action < error_prob * 0.80:
            if len(word) > 3:
                chars = list(word)
                idx = random.randint(0, len(chars)-1)
                c = chars[idx].lower()
                if c in keyboard_typos:
                    chars[idx] = keyboard_typos[c] if chars[idx].islower() else keyboard_typos[c].upper()
                    new_words.append("".join(chars))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
            i += 1
            continue
        elif action < error_prob * 0.86:
            if len(word) > 4:
                char_idx = random.randint(0, len(word)-1)
                word = word[:char_idx] + word[char_idx+1:]
            new_words.append(word)
            i += 1
            continue
        elif action < error_prob * 0.94:
            if len(word) >= 2 and word.isalpha():
                vowels = "aeiouAEIOU"
                chars = list(word)
                for v_idx, char in enumerate(chars):
                    if char in vowels:
                        chars[v_idx] = char * random.randint(3, 5)
                        break
                new_words.append("".join(chars))
            else:
                new_words.append(word)
            i += 1
            continue
        elif action < error_prob * 1.0 and i < len(words) - 1:
            new_words.append(word + words[i+1])
            i += 2
            continue
        else:
            new_words.append(word)
            i += 1

    result = " ".join(new_words)

    if random.random() < error_prob * 1.5:
        punct_action = random.random()
        if punct_action < 0.25:
            result = result.replace('.', '').replace(',', '').replace('!', '?')
        elif punct_action < 0.50:
            result = re.sub(r'([.,!?])\s+', r'\1', result)
        elif punct_action < 0.75:
            if len(result) > 0:
                result = result[0].lower() + result[1:]
        else:
            if len(result) > 5:
                idx = random.randint(1, len(result)-2)
                result = result[:idx] + result[idx].upper() + result[idx+1:]

    if result.strip() == "":
        return text

    return result

if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
    logger.info("Generating datasets from HuggingFace...")
    dataset_new = load_dataset('theguywithblacktie/hinglish-conversations', 'large', split='train')
    df_new = dataset_new.to_pandas()

    input_texts = df_new['input'].apply(extract_texts).explode().dropna()
    output_texts = df_new['output'].apply(extract_texts).explode().dropna()
    all_texts = pd.concat([input_texts, output_texts], ignore_index=True)
    all_texts = all_texts[all_texts != '']
    df_paragraphs = pd.DataFrame({'paragraph': all_texts}).drop_duplicates().reset_index(drop=True)

    df_train = pd.DataFrame()
    df_train['correct_text'] = df_paragraphs['paragraph']
    df_train['incorrect_text'] = df_train['correct_text'].apply(lambda x: introduce_noise(x, error_prob=0.15))
    df_train = df_train[df_train['correct_text'] != df_train['incorrect_text']].reset_index(drop=True)
    df_train.to_csv(TRAIN_CSV, index=False)

    dataset_test = load_dataset('theguywithblacktie/hinglish-conversations', 'large', split='test')
    df_test = dataset_test.to_pandas()
    test_inputs = df_test['input'].apply(extract_texts).explode().dropna()
    test_outputs = df_test['output'].apply(extract_texts).explode().dropna()
    all_test = pd.concat([test_inputs, test_outputs], ignore_index=True)
    df_test_para = pd.DataFrame({'paragraph': all_test[all_test != '']}).drop_duplicates().reset_index(drop=True)

    df_test_grammar = pd.DataFrame()
    df_test_grammar['correct_text'] = df_test_para['paragraph'].sample(n=10000, random_state=42)
    df_test_grammar['incorrect_text'] = df_test_grammar['correct_text'].apply(lambda x: introduce_noise(x, error_prob=0.15))
    df_test_grammar = df_test_grammar[df_test_grammar['correct_text'] != df_test_grammar['incorrect_text']].reset_index(drop=True)
    df_test_grammar.to_csv(TEST_CSV, index=False)

class GrammarCorrectionDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length):
        df = pd.read_csv(csv_file).dropna()
        self.input_ids, self.attention_masks, self.labels = [], [], []

        for _, row in df.iterrows():
            prompt = f"Fix the grammar in this Hinglish sentence.\nIncorrect: {row['incorrect_text']}\nCorrect: "
            response = f"{row['correct_text']}{tokenizer.eos_token}"

            prompt_tokens = tokenizer(prompt, add_special_tokens=False)['input_ids']
            resp_tokens = tokenizer(response, add_special_tokens=False)['input_ids']
            full_tokens = prompt_tokens + resp_tokens

            if len(full_tokens) > max_length:
                full_tokens = full_tokens[:max_length]
                if len(prompt_tokens) >= max_length:
                    prompt_tokens = prompt_tokens[:max_length]
                    resp_tokens = []
                else:
                    resp_tokens = full_tokens[len(prompt_tokens):]

            pad_len = max_length - len(full_tokens)
            padded_tokens = full_tokens + [tokenizer.pad_token_id] * pad_len
            attn_mask = [1] * len(full_tokens) + [0] * pad_len
            label = [-100] * len(prompt_tokens) + resp_tokens + [-100] * pad_len

            self.input_ids.append(padded_tokens)
            self.attention_masks.append(attn_mask)
            self.labels.append(label)

    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

dataset = GrammarCorrectionDataset(TRAIN_CSV, tokenizer, MAX_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
logger.info(f"Total training examples ready: {len(dataset):,} | Batches per epoch: {len(dataloader):,}")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
)
model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)
model.train()

no_decay = ["bias", "layer_norm", "layernorm", "rmsnorm"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay) and p.requires_grad], "weight_decay": WEIGHT_DECAY},
    {"params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

total_training_steps = (len(dataloader) // GRAD_ACCUM_STEPS) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_training_steps * WARMUP_RATIO), total_training_steps)
scaler = torch.amp.GradScaler(device='cuda', enabled=(DTYPE == torch.float16))

global_step = 0
start_time = time.perf_counter()

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with torch.amp.autocast(device_type="cuda" if DEVICE == "cuda" else "cpu", dtype=DTYPE):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        epoch_loss += loss.item() * GRAD_ACCUM_STEPS

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % 10 == 0:
                elapsed = time.perf_counter() - start_time
                eta = str(datetime.timedelta(seconds=int((total_training_steps - global_step) * (elapsed / global_step))))
                logger.info(f"Epoch {epoch+1}/{EPOCHS} | Step {global_step}/{total_training_steps} | Loss: {loss.item() * GRAD_ACCUM_STEPS:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | ETA: {eta}")

logger.info(f"Training finished in {(time.perf_counter() - start_time) / 60:.2f} minutes.")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
