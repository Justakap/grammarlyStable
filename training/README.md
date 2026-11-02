# Hinglish Writing Assistant: Grammar Correction and Autocomplete

Two models trained on the Hugging Face dataset `theguywithblacktie/hinglish-conversations` (`large` config).

```
training/
├── README.md
├── requirements.txt
├── grammar_correction_qwen3_4b_lora/
│   └── code/      train_4b.py, train_4b.sh
└── autocomplete_qwen3_0.6b_full_finetune/
    ├── code/      train_next_token.py, train_next_token.sh
    └── results/   validation_metrics.json, test_metrics.json
```

## Models and datasets (Google Drive)

The trained models and datasets are in one zip on Google Drive (1.58 GB):

**[Hinglish_Models_and_Datasets.zip](https://drive.google.com/file/d/1ULqkyu2Zgv-6X46nm1BqIo0RXQL5jrhl/view?usp=sharing)**

```
Hinglish_Models_and_Datasets/
├── grammar_model_qwen3_4b_lora/      LoRA adapter + tokenizer (load on top of Qwen/Qwen3-4B)
├── grammar_dataset/                  grammar_dataset_train.csv (642,001 pairs), grammar_dataset_test.csv
├── autocomplete_model_qwen3_0.6b/    full fine-tuned model + tokenizer (standalone)
└── autocomplete_dataset/             processed_dataset/ train.jsonl (357,544), validation.jsonl (7,297), test.jsonl (40,538)
```

## Models

| | Grammar correction | Autocomplete |
| --- | --- | --- |
| Base model | Qwen/Qwen3-4B | Qwen/Qwen3-0.6B-Base |
| Method | LoRA r=32, alpha=64, dropout 0.05, all 7 projection layers | Full fine-tuning |
| Trainable parameters | 66,060,288 of 4,088,528,384 (1.62%) | 596,049,920 (100%) |
| Max length | 256 tokens | 1,024 tokens |
| Effective batch | 64 (16 x 4) | 32 (8 x 4) |
| Learning rate / warm-up | 2e-4 / 5% | 1e-5 / 3% |
| Epochs | 1 | 1 |
| Training time (1x NVIDIA L40S 48 GB) | 1,073 min | 142 min |

## Results

Grammar correction, samples from the synthetic test set:

| Model | Exact match | BLEU |
| --- | --- | --- |
| Qwen3-4B, no fine-tuning | 0.00% | 15.69 |
| Qwen3-4B + LoRA | 50.00% | 90.14 |

Autocomplete, token-level:

| Split | Loss | Perplexity | Top-1 | Top-3 |
| --- | --- | --- | --- | --- |
| Validation | 2.659 | 14.28 | 46.41% | 63.26% |
| Test | 2.663 | 14.33 | 46.35% | 63.20% |

## Running

```bash
pip install -r requirements.txt
```

Grammar correction: copy the two CSVs from `grammar_dataset/` into `code/` and run `sbatch train_4b.sh` (if they are missing, the script rebuilds them from Hugging Face). Inference: load `Qwen/Qwen3-4B`, attach `grammar_model_qwen3_4b_lora/` with `PeftModel.from_pretrained`, and send:

```
Fix the grammar in this Hinglish sentence.
Incorrect: <noisy text>
Correct: 
```

Autocomplete: copy `autocomplete_dataset/processed_dataset/` into `code/` and run `sbatch train_next_token.sh` (without it, the script downloads and rebuilds the data). Inference: load `autocomplete_model_qwen3_0.6b/` with `AutoModelForCausalLM.from_pretrained` and give it plain Hinglish text with no prompt template.
