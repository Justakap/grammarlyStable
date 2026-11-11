"""
Run this entire script as cells in a SINGLE Google Colab notebook (GPU runtime).

Assignment 3 models:
  - /correct     : Qwen3-4B + G2_long LoRA adapter (grammar correction, persona-aware)
  - /predict     : Qwen3-0.6B persona-aware next-word model (full fine-tune)
  - /sensitivity : Qwen3-0.6B sensitivity score, 0 (safe) to 100 (offensive) (full fine-tune, one-output head)

All routes run on ONE FastAPI server behind ONE cloudflared tunnel. The URL is
registered with the backend three times: service="correct", "predict" and "sensitivity".

GPU memory: one Qwen3-4B (~8 GB in bf16/fp16) + one Qwen3-0.6B (~1.2 GB) + the sensitivity
Qwen3-0.6B in fp32 (~2.4 GB), about 12 GB in total, so a T4 (15 GB) is enough.
For the 8B grammar model (8B_all) set BASE_MODEL_PATH = "Qwen/Qwen3-8B" (~17 GB, needs L4/A100).
"""

from google.colab import drive
drive.mount('/content/drive')
!nvidia-smi
!pip install -q fastapi "uvicorn[standard]" peft accelerate transformers
!pip uninstall -y -q torchao
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared

import os, re, subprocess, time, requests

# ---------- Settings ----------
# Grammar: G2_long adapter folder (adapter_config.json + adapter_model.safetensors),
# copied from Assignment3-D-Mix-Rank-LR/runs/G2_long/best (or runs/run_G2_long/best).
CORRECT_LORA_PATH = "/content/drive/MyDrive/QWEN4-A3-LORA"
BASE_MODEL_PATH = "Qwen/Qwen3-4B"            # "Qwen/Qwen3-8B" for the 8B_all adapter

# Next word: full model folder copied from AutoComplete/runs/run_0.6B/best (or run_1.7B/best).
PREDICT_MODEL_PATH = "/content/drive/MyDrive/QWEN3-0.6-A3"
PREDICT_BASE_MODEL = "Qwen/Qwen3-0.6B-Base"  # only used if the folder has no tokenizer; "Qwen/Qwen3-1.7B-Base" for 1.7B

# Sensitivity: full model folder copied from Assignment3_Sensitivity_Score/runs/run_qwen0.6b/best
# (config.json, model.safetensors, tokenizer.json, tokenizer_config.json, chat_template.jinja, calibration.json).
SENS_MODEL_PATH = "/content/drive/MyDrive/QWEN3-0.6-A3-Sensi"
SENS_BASE_MODEL = "Qwen/Qwen3-0.6B-Base"     # only used if the folder has no tokenizer

DEFAULT_PERSONA = "none"   # none / friends / family / colleagues / experts / student_teacher
API_KEY = ""               # optional: set e.g. "my-secret-key" to require an x-api-key header on all endpoints

BACKEND_URL = "https://grammarlystable-1.onrender.com"
BACKEND_REGISTRATION_TOKEN = "f96faec6b167e628d8ac56c6e52f6363705e82d14a853e5f9c6d213f0d81ab3e"   # must match Render's BACKEND_REGISTRATION_TOKEN

# ---------- Write the API file ----------
main_py = r'''
import os, re, json, math, threading
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel

BASE_MODEL_PATH = os.environ["BASE_MODEL_PATH"]
CORRECT_LORA_PATH = os.environ["CORRECT_LORA_PATH"]
PREDICT_MODEL_PATH = os.environ["PREDICT_MODEL_PATH"]
PREDICT_BASE_MODEL = os.environ["PREDICT_BASE_MODEL"]
SENS_MODEL_PATH = os.environ["SENS_MODEL_PATH"]
SENS_BASE_MODEL = os.environ["SENS_BASE_MODEL"]
DEFAULT_PERSONA = os.getenv("DEFAULT_PERSONA", "none")
API_KEY = os.getenv("API_KEY", "")

PERSONAS = {"none", "friends", "family", "colleagues", "experts", "student_teacher"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    DTYPE = torch.float32

state = {}
correct_lock = threading.Lock()
predict_lock = threading.Lock()
sens_lock = threading.Lock()


def _load_tokenizer(primary_path, fallback_path=None):
    try:
        tok = AutoTokenizer.from_pretrained(primary_path)
    except Exception:
        if fallback_path is None:
            raise
        print(f"No tokenizer in {primary_path}, falling back to {fallback_path}")
        tok = AutoTokenizer.from_pretrained(fallback_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _from_pretrained(path):
    kwargs = {"device_map": "auto" if DEVICE == "cuda" else None}
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=DTYPE, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=DTYPE, **kwargs)


def _load_lora(lora_path):
    # The adapter folder may not have a tokenizer; the base model's tokenizer is the same one used in training.
    tok = _load_tokenizer(lora_path, fallback_path=BASE_MODEL_PATH)
    base = _from_pretrained(BASE_MODEL_PATH)
    model = PeftModel.from_pretrained(base, lora_path).merge_and_unload()
    model.eval()
    return tok, model


def _load_full(model_path):
    tok = _load_tokenizer(model_path, fallback_path=PREDICT_BASE_MODEL)
    model = _from_pretrained(model_path)
    model.eval()
    return tok, model


def _load_sens(model_path):
    # One-output classifier, loaded in fp32 (the weights were trained and saved in fp32).
    tok = _load_tokenizer(model_path, fallback_path=SENS_BASE_MODEL)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, dtype=torch.float32)
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    model.to(DEVICE).eval()
    calib = os.path.join(model_path, "calibration.json")
    temperature = json.load(open(calib))["temperature"] if os.path.exists(calib) else 1.0
    return tok, model, float(temperature)


def _clean_persona(p):
    p = (p or DEFAULT_PERSONA).strip().lower().replace(" ", "_").replace("-", "_")
    return p if p in PERSONAS else "none"


@asynccontextmanager
async def lifespan(app):
    print("Loading correction model...")
    state["correct_tok"], state["correct_model"] = _load_lora(CORRECT_LORA_PATH)
    tok = state["correct_tok"]
    eos = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid != tok.unk_token_id:
            eos.add(tid)
    state["correct_eos"] = sorted(i for i in eos if i is not None)
    print("Loading prediction model...")
    state["predict_tok"], state["predict_model"] = _load_full(PREDICT_MODEL_PATH)
    print("Loading sensitivity model...")
    state["sens_tok"], state["sens_model"], state["sens_temperature"] = _load_sens(SENS_MODEL_PATH)
    print(f"All three models loaded. Sensitivity temperature: {state['sens_temperature']}")
    yield
    state.clear()


app = FastAPI(title="Hinglish Grammarly - Combined Model Server (Assignment 3)", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class Req(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    persona: Optional[str] = None                 # optional; default DEFAULT_PERSONA (correct and predict only)
    context: Optional[List[str]] = None           # optional: earlier messages of the chat (predict only)


def _check_api_key(x_api_key: str):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "correct_loaded": "correct_model" in state,
        "predict_loaded": "predict_model" in state,
        "sensitivity_loaded": "sens_model" in state,
    }


# ---------- Grammar correction (same prompt and decoding as evaluate_grammar_a3.py) ----------
def build_prompt(incorrect, persona):
    return ("Fix the grammar in this Hinglish sentence.\n"
            f"Persona: {persona}\n"
            f"Incorrect: {incorrect}\n"
            "Correct: ")


def first_line(text):
    line = text.strip().split("\n")[0].strip()
    return line.strip('"').strip("'").strip()


@app.post("/correct")
def correct(req: Req, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    if "correct_model" not in state:
        raise HTTPException(status_code=503, detail="Correction model not loaded yet")

    tok, model = state["correct_tok"], state["correct_model"]
    text = " ".join(req.text.split())
    prompt = build_prompt(text, _clean_persona(req.persona))
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    n = inputs["input_ids"].shape[1]
    with correct_lock, torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=n + 32, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id,
                             eos_token_id=state["correct_eos"])
    corrected = first_line(tok.decode(out[0][n:], skip_special_tokens=True))
    return {"incorrect": text, "corrected": corrected or text}


# ---------- Next-word prediction (same method as predict_words in evaluate_next_word.py) ----------
NUM_SUGGESTIONS = 3
CANDIDATE_POOL = 50
MAX_WORD_TOKENS = 6
MAX_CONTEXT_TOKENS = 900   # the model was trained on up to 1,024 tokens


@app.post("/predict")
def predict(req: Req, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    if "predict_model" not in state:
        raise HTTPException(status_code=503, detail="Prediction model not loaded yet")

    tok, model = state["predict_tok"], state["predict_model"]
    current = req.text.rstrip()                     # context ends at the last complete word, no trailing space
    messages = [m.strip() for m in (req.context or []) if m and m.strip()] + [current]
    prefix = tok(f"Persona: {_clean_persona(req.persona)}\n", add_special_tokens=False)["input_ids"]
    body = tok("\n".join(messages), add_special_tokens=False)["input_ids"]
    body = body[-(MAX_CONTEXT_TOKENS - len(prefix)):]
    ids = torch.tensor([prefix + body], device=model.device)

    predictions = []
    with predict_lock, torch.inference_mode():
        logits = model(input_ids=ids).logits[0, -1]
        firsts = []
        for t in logits.topk(CANDIDATE_POOL).indices.tolist():
            piece = tok.decode([t])
            if piece.startswith(" ") and any(c.isalnum() for c in piece):
                firsts.append(t)
            if len(firsts) == NUM_SUGGESTIONS:
                break
        if firsts:
            seqs = torch.cat([ids.repeat(len(firsts), 1),
                              torch.tensor([[t] for t in firsts], device=model.device)], dim=1)
            out = model.generate(seqs, attention_mask=torch.ones_like(seqs),
                                 max_new_tokens=MAX_WORD_TOKENS - 1, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            for t, row in zip(firsts, out[:, seqs.shape[1]:].tolist()):
                word = tok.decode([t])
                for x in row:
                    piece = tok.decode([x])
                    if x in (tok.eos_token_id, tok.pad_token_id) or piece == "" or piece[0].isspace():
                        break
                    word += piece
                word = word.strip().strip(".,!?;:\"'()[]{}-")
                if word and word.lower() not in {w.lower() for w in predictions}:
                    predictions.append(word)

    return {"input": req.text, "predictions": predictions[:NUM_SUGGESTIONS]}


# ---------- Sensitivity score (same cleaning, scoring and bands as score_text.py / sens_common.py) ----------
SENS_MAX_LENGTH = 128      # same as training and evaluation
SENS_BANDS = ((30.0, "safe"), (70.0, "sensitive"), (101.0, "offensive"))
_INVISIBLE = re.compile("[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f\\u200b-\\u200f\\xa0\\u202f\\ufeff]")
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_MENTION = re.compile(r"@\w+")


def _clean_text(text):
    text = _INVISIBLE.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _MENTION.sub("@user", text)
    return re.sub(r"\s+", " ", text).strip()


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def _band(score):
    for limit, name in SENS_BANDS:
        if score < limit:
            return name
    return SENS_BANDS[-1][1]


@app.post("/sensitivity")
def sensitivity(req: Req, x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    if "sens_model" not in state:
        raise HTTPException(status_code=503, detail="Sensitivity model not loaded yet")

    tok, model = state["sens_tok"], state["sens_model"]
    text = _clean_text(req.text)
    if not text:                                    # nothing left after removing links: nothing to score
        return {"text": req.text, "score": 0.0, "band": "safe"}
    # One message per call, so there is no padding (the classifier reads the last token).
    enc = tok(text, truncation=True, max_length=SENS_MAX_LENGTH, return_tensors="pt").to(model.device)
    with sens_lock, torch.inference_mode():
        logit = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.float().view(-1)[0].item()
    score = 100.0 * _sigmoid(logit / state["sens_temperature"])
    return {"text": req.text, "score": round(score, 1), "band": _band(score)}
'''
with open("main.py", "w") as f:
    f.write(main_py)

# ---------- Start the API in the background ----------
# Cleanup any previous instances first
subprocess.run(["pkill", "-f", "uvicorn"])
subprocess.run(["pkill", "-f", "cloudflared"])
time.sleep(3)

env = dict(os.environ,
           BASE_MODEL_PATH=BASE_MODEL_PATH,
           CORRECT_LORA_PATH=CORRECT_LORA_PATH,
           PREDICT_MODEL_PATH=PREDICT_MODEL_PATH,
           PREDICT_BASE_MODEL=PREDICT_BASE_MODEL,
           SENS_MODEL_PATH=SENS_MODEL_PATH,
           SENS_BASE_MODEL=SENS_BASE_MODEL,
           DEFAULT_PERSONA=DEFAULT_PERSONA,
           API_KEY=API_KEY)
subprocess.Popen(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
                 env=env, stdout=open("api.log", "w"), stderr=subprocess.STDOUT)

print("Loading all three models (this can take a few minutes)...")
ready = False
for _ in range(180):
    try:
        health = requests.get("http://localhost:8000/health", timeout=2).json()
        if health.get("correct_loaded") and health.get("predict_loaded") and health.get("sensitivity_loaded"):
            ready = True
            break
    except Exception:
        pass
    time.sleep(5)
if not ready:
    print(open("api.log").read()[-3000:])
    raise RuntimeError("API failed to start, see log above")

# ---------- Quick local check before going public ----------
hdr = {"x-api-key": API_KEY} if API_KEY else {}
print(requests.post("http://localhost:8000/correct", headers=hdr,
                    json={"text": "mujhe kal market jana tha lekin main nahi gaya tha"}).json())
print(requests.post("http://localhost:8000/predict", headers=hdr,
                    json={"text": "kal milte hai, movie ka plan"}).json())
# Expected from the v1 evaluation: about 1.2 (safe) and about 83.3 (offensive).
for t in ("bhai tu bahut accha kaam kar raha hai, keep it up", "chup kar bewakoof, teri aukaat kya hai"):
    print(requests.post("http://localhost:8000/sensitivity", headers=hdr, json={"text": t}).json())

# ---------- Open the public tunnel (ONE tunnel serves all routes) ----------
subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://localhost:8000"],
                 stdout=open("tunnel.log", "w"), stderr=subprocess.STDOUT)
url = None
for _ in range(30):
    time.sleep(2)
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open("tunnel.log").read())
    if m:
        url = m.group(0)
        break

print("\nAPI ready!")
print("Public URL :", url)
print("Docs       :", f"{url}/docs")

# ---------- Register this ONE url with our backend under ALL service names ----------
def register_with_backend(colab_url: str, service: str):
    if not colab_url:
        print(f"WARNING: no public URL found, skipping backend registration for '{service}'.")
        return
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/colab/register",
            json={"url": colab_url, "service": service},
            headers={"Authorization": f"Bearer {BACKEND_REGISTRATION_TOKEN}"},
            timeout=10,
        )
        if not resp.ok:
            print(f"Failed to register '{service}' with backend ({resp.status_code}): {resp.text}")
        else:
            print(f"Registered {colab_url} with backend at {BACKEND_URL} as service='{service}'")
    except Exception as e:
        print(f"Failed to reach backend to register '{service}': {e}")

register_with_backend(url, "correct")
register_with_backend(url, "predict")
register_with_backend(url, "sensitivity")
