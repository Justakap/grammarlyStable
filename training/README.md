# Training code

Each assignment has its own folder with its own code, requirements, README and results. Nothing is shared
between them in code; Assignment 3 only starts from the trained Assignment 2 adapter (from the Google Drive zip).

| Folder | Assignment | Models |
| --- | --- | --- |
| [`assignment2/`](assignment2/README.md) | Assignment 2 | Grammar correction (Qwen3-4B + LoRA) and autocomplete (Qwen3-0.6B-Base, full fine-tuning) |
| [`assignment3/`](assignment3/grammar_correction_qwen3_4b_lora/README.md) | Assignment 3 | Grammar correction (Qwen3-4B + LoRA) with IndicTalk persona data, improved Hinglish errors and new metrics |

```
training/
├── README.md
├── assignment2/
│   ├── README.md
│   ├── requirements.txt
│   ├── grammar_correction_qwen3_4b_lora/
│   │   ├── code/      train_4b.py, train_4b.sh
│   │   └── logs/      qwen4b_out_6113.log, qwen4b_err_6113.log
│   └── autocomplete_qwen3_0.6b_full_finetune/
│       ├── code/      train_next_token.py, train_next_token.sh
│       ├── results/   validation_metrics.json, test_metrics.json
│       └── logs/      hinglish_next_6148.out, hinglish_next_6148.err, training.log
└── assignment3/
    └── grammar_correction_qwen3_4b_lora/
        ├── README.md
        ├── code/      data preparation, training, evaluation and SLURM scripts, requirements.txt
        └── results/   results_table.md
```
