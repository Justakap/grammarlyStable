# Learning rate and LoRA rank sweep on Data Mix D

Finds the best learning rate and LoRA rank for run D: the Assignment 2 adapter trained further on Data Mix D
(IndicTalk + 25% Assignment 2 sentences, new errors). Everything this folder produces stays inside it, so the
main Assignment 3 runs and results are not touched.

## Folder layout

Put this folder **inside** the Assignment 3 folder. It reuses the parent's data, training and evaluation code:

```
Assignment3_Grammar_Correction/
├── data_a3/                    used as is (Data Mix D, val.csv, test sets)
├── qwen_4b_hinglish_lora/      Assignment 2 adapter (starting point)
├── train_4b_a3.py, evaluate_grammar_a3.py, summarize_results.py
└── sweep_lr_rank/              this folder
    ├── resize_adapter.py       changes the Assignment 2 adapter's rank (16, 64, ...)
    ├── split_val.py            splits val.csv into val_indictalk.csv and val_a2.csv
    ├── select_best.py          ranks the sweep runs and picks the winner
    ├── sweep_train.sh          trains one configuration (SLURM)
    ├── sweep_eval.sh           evaluates one configuration on validation or test (SLURM)
    ├── submit_sweep.sh         submits the whole sweep
    ├── submit_final.sh         retrains the winner on all of Data Mix D and scores the test sets
    ├── adapters/               resized Assignment 2 adapters (created)
    ├── data/                   validation split by source (created)
    ├── runs/<name>/            trained adapters and logs (created)
    ├── results/                validation results, results_table.md, selection.md (created)
    └── results_test/           test results of the final run (created)
```

## Changing the rank of a continued adapter

Run D continues the Assignment 2 adapter, which has rank 32, so other ranks need a resized copy of it first.
`resize_adapter.py` keeps alpha at twice the rank, so the update is scaled the same way at every rank:

- **Higher rank (64):** the 32 trained directions are kept and 32 new ones are added that start at zero. The
  resized adapter gives exactly the same output as the original before training continues.
- **Lower rank (16):** the 16 most important directions of each layer's update are kept, which is the best
  possible rank-16 approximation. The log reports how much of each update is kept.

This happens automatically in `sweep_train.sh` the first time a rank is used.

## Running it

From inside `sweep_lr_rank/` on the cluster (with `bash`, not `sbatch`):

```bash
dos2unix *.sh
bash submit_sweep.sh
```

This submits 5 training runs, each followed by a validation evaluation:

| Run | Learning rate | Rank | Tests |
| --- | --- | --- | --- |
| `lr5e-5_r32` | 5e-5 | 32 | Learning rate |
| `lr1e-4_r32` | 1e-4 | 32 | Learning rate (run D's setting) |
| `lr2e-4_r32` | 2e-4 | 32 | Learning rate |
| `lr1e-4_r16` | 1e-4 | 16 | Rank |
| `lr1e-4_r64` | 1e-4 | 64 | Rank |

Each run trains on a random 50,000 pairs of Data Mix D (same 75/25 mix) so the sweep is quick. Scores are
therefore a little below the full run D; what matters is the comparison between configurations.

Options:

```bash
MAX_TRAIN_EXAMPLES=100000 bash submit_sweep.sh
LRS="1e-4 3e-4" RANKS="16 64 128" bash submit_sweep.sh
A2_ADAPTER=/path/to/qwen_4b_hinglish_lora bash submit_sweep.sh
```

By default the rank runs use run D's learning rate (1e-4) so all five run at once. If the learning-rate runs
show a clearly better rate, rerun only the rank runs at that rate:

```bash
LRS="" RANKS="16 64" RANK_LR=2e-4 bash submit_sweep.sh
```

## Choosing the winner

After each evaluation, `results/selection.md` is updated:

- Score: mean edit F0.5 over `val_indictalk` and `val_a2`.
- Runs within 1 point of the best are treated as tied; among those, the lowest over-correction wins, then the
  smaller rank.
- Selection uses the validation set only. The test sets are used once, for the final run.

## Final run

`selection.md` ends with the command for the winner, for example:

```bash
bash submit_final.sh 1e-4 64
```

This trains on all of Data Mix D (about 200,000 pairs) and evaluates on the test sets. Results go to
`results_test/results_table.md`. Compare them with run D in `../results/results_table.md` to see whether the
new learning rate or rank beats the original D.
