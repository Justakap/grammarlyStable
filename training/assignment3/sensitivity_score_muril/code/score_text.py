"""Score Hinglish messages: python3 score_text.py "message one" "message two"
Uses runs/run_muril/best by default (MODEL_PATH to change)."""
import os
import sys
import json

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sens_common import sigmoid, band, clean_text

MODEL_PATH = os.getenv("MODEL_PATH", "runs/run_muril/best")
texts = sys.argv[1:] or [line.strip() for line in sys.stdin if line.strip()]
if not texts:
    sys.exit('Usage: python3 score_text.py "message" ["message" ...]')

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).eval()
model.config.pad_token_id = tokenizer.pad_token_id
calib = os.path.join(MODEL_PATH, "calibration.json")
temperature = json.load(open(calib))["temperature"] if os.path.exists(calib) else 1.0

with torch.no_grad():
    enc = tokenizer([clean_text(t) for t in texts], padding=True, truncation=True, max_length=128, return_tensors="pt")
    logits = model(**enc).logits.view(-1).tolist()
for t, z in zip(texts, logits):
    s = 100 * sigmoid(z / temperature)
    print(f"{s:5.1f}%  {band(s):9}  {t}")
