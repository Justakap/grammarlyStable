"""
FastAPI server for the Hinglish correction model, meant to run inside Google Colab.

This module only defines the app + model loading. The actual Colab notebook
(`hinglish_colab.ipynb`) installs dependencies, imports this module, starts the
server, opens a tunnel, and registers the tunnel's public URL with our backend.

The model is loaded exactly once at import time (module load), not per-request.
"""

import os
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-3-4b-it")

SYSTEM_PROMPT = """You are a Hinglish grammar correction assistant.

Correct the user's Hinglish sentence while preserving the original meaning and natural Hinglish style.

Fix:
- spelling
- grammar
- word forms
- sentence structure
- obvious transliteration mistakes
- punctuation when appropriate

Do not translate Hinglish into English.
Do not translate Hinglish into pure Hindi.
Do not add explanations.
Do not add quotation marks.
Return ONLY the corrected sentence."""

app = FastAPI(title="Hinglish Correction Model Server")

# --- Model loading (happens once, when this module is imported) ---

_model_lock = threading.Lock()
_tokenizer = None
_model = None


def load_model():
    """Loads the tokenizer and model once. Safe to call multiple times."""
    global _tokenizer, _model

    with _model_lock:
        if _model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {MODEL_NAME} ...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("Model loaded.")


def generate_correction(text: str) -> str:
    """Runs the model on a single Hinglish sentence and returns the corrected text."""
    import torch

    if _model is None or _tokenizer is None:
        raise RuntimeError("Model is not loaded yet.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    inputs = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated = output_ids[0][inputs.shape[-1]:]
    corrected = _tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Defensive cleanup in case the model wraps the answer in quotes anyway.
    corrected = corrected.strip('"').strip("'").strip()

    return corrected


# --- API schema ---

class CorrectRequest(BaseModel):
    text: str


class CorrectResponse(BaseModel):
    corrected: str


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": _model is not None}


@app.post("/correct", response_model=CorrectResponse)
def correct(payload: CorrectRequest):
    text = (payload.text or "").strip()

    if not text:
        raise HTTPException(status_code=400, detail='"text" must not be empty.')

    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="Text is too long.")

    try:
        corrected = generate_correction(text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    return CorrectResponse(corrected=corrected)


# Load the model as soon as this module is imported (once per Colab runtime).
load_model()
