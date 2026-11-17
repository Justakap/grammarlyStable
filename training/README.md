# Training code

Each assignment has its own folder with its own code, requirements, README and results. Nothing is shared
between them in code; Assignment 3 only starts from the trained Assignment 2 adapter (from the Google Drive zip).

| Folder | Assignment | Models |
| --- | --- | --- |
| [`assignment2/`](assignment2/README.md) | Assignment 2 | Grammar correction (Qwen3-4B + LoRA) and autocomplete (Qwen3-0.6B-Base, full fine-tuning) |
| [`assignment3/`](assignment3/grammar_correction_qwen3_4b_lora/README.md) | Assignment 3 | Grammar correction (Qwen3-4B + LoRA, and a Qwen3-8B run) with IndicTalk persona data, improved Hinglish errors and new metrics; [persona-aware next-word prediction](assignment3/next_word_prediction_qwen3_full_finetune/README.md) (Qwen3-0.6B-Base and 1.7B-Base, full fine-tuning); [Hinglish sensitivity score](assignment3/sensitivity_score_muril/README.md) (MuRIL, compared with Qwen3-0.6B) |

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
    ├── grammar_correction_qwen3_4b_lora/
    │   ├── README.md
    │   ├── code/      data preparation, training, evaluation and SLURM scripts, requirements.txt
    │   │   └── sweep_lr_rank/   learning rate and LoRA rank sweep
    │   └── results/   results_table.md, results_long_table.md, sweep_results.md
    ├── next_word_prediction_qwen3_full_finetune/
    │   ├── README.md
    │   ├── code/      data preparation, training, evaluation and SLURM scripts
    │   └── results/   results_table.md, all_results.csv
    └── sensitivity_score_muril/
        ├── README.md
        ├── code/      data preparation, training, evaluation, scoring and SLURM scripts
        └── results/   results_table.md
```
