# Assignment 3: Hinglish Grammar Correction (Qwen3-4B + LoRA)

Continues the Assignment 2 grammar model (`training/assignment2/grammar_correction_qwen3_4b_lora`) with the
IndicTalk persona dataset, richer Hinglish errors and better evaluation. The scripts are in `code/`;
results are in `results/results_table.md`.

## Results

Run D (the Assignment 2 adapter, trained further on IndicTalk plus 25% Assignment 2 sentences) is the best run.
Compared with the Assignment 2 model (run A):

| Test set | Model | Exact match | Over-correction | Edit F0.5 | GLEU | chrF |
| --- | --- | --- | --- | --- | --- | --- |
| `test_indictalk` | A (Assignment 2) | 38.85 | 42.40 | 81.72 | 92.63 | 98.03 |
| `test_indictalk` | **D** | **70.75** | **1.94** | **91.85** | **97.26** | **99.10** |
| `test_a2` | A (Assignment 2) | 63.40 | 38.44 | 85.29 | 93.42 | 98.18 |
| `test_a2` | **D** | **76.75** | **6.31** | **89.90** | **96.09** | **98.78** |

- The untuned Qwen3-4B (BASE) scores below simply copying the input, so fine-tuning is essential.
- Keeping 15% of training sentences unchanged cut over-correction from about 40% to 2-6%.
- The new errors alone help: run E beats A on `test_a2` with less than a quarter of the data.
- Training on IndicTalk alone (run C) loses accuracy on Assignment 2 sentences; mixing in 25% old data (run D) prevents this.
- The persona line has no measurable effect (D vs D_persona0).

All runs and metrics: [`results/results_table.md`](results/results_table.md).

## Files

| File | What it does |
| --- | --- |
| `hinglish_noise.py` | Error generator: the Assignment 2 noise (fixed) plus new context-aware Hinglish grammar errors, each tagged |
| `prompt_a3.py` | The training and inference prompt (with a `Persona:` line) |
| `prepare_data_a3.py` / `prepare_data_a3.sh` | Builds `data_a3/` from IndicTalk (Hindi, Romanized only) and Assignment 2 data |
| `prepare_a2_full_a3.py` / `prepare_a2_full_a3.sh` | Builds `data_a3/train_a2_full.csv` (Assignment 2 sentences, new errors) for runs E and F; run after `prepare_data_a3.sh` |
| `train_4b_a3.py` / `train_4b_a3.sh` | LoRA training; runs B, C, D, E or F |
| `evaluate_grammar_a3.py` / `evaluate_a3.sh` | Generates corrections and computes all metrics; runs BASE, A, B, C, D, E or F |
| `summarize_results.py` | Builds `results/results_table.md` from every evaluation so far |
| `submit_all_a3.sh` | Submits all jobs in the right order using SLURM dependencies |

## Before you start

1. Copy `code/` to the cluster and run everything from inside it.
2. Put the Assignment 2 adapter folder `qwen_4b_hinglish_lora/` (from the Google Drive zip) inside it, or pass its path with
   `A2_ADAPTER=/path/to/qwen_4b_hinglish_lora`.
3. Optional: copy `grammar_dataset_train.csv` and `grammar_dataset_test.csv` from Assignment 2 (also in the Drive zip) into it.
   Without them the script re-downloads the Assignment 2 dataset, and `test_a2_original.csv` (the exact
   Assignment 2 test pairs, for comparison with last time) is not created.
4. The `qwen_env` environment from Assignment 2 is used. `evaluate_a3.sh` installs `sacrebleu` if it is missing.

## Run order

Jobs depend on each other:

| Step | Job | Needs | Can run alongside |
| --- | --- | --- | --- |
| 1 | `prepare_data_a3.sh` | nothing | – |
| 2 | Train D, C, B | step 1 | each other, if you get more than one GPU |
| 2 | Evaluate BASE and A | step 1 | training (they don't use a new model) |
| 3 | Evaluate D, C, B | that run's training finished | each other |

The easiest way is to submit everything at once with SLURM dependencies, from the login node (with `bash`, not `sbatch`):

```bash
bash submit_all_a3.sh
```

It submits all jobs in one go and exits in a few seconds (no waiting loop). SLURM holds each job as
`PENDING (Dependency)` until what it needs has finished, and a pending job uses no GPU. Jobs that are ready run
in parallel up to the cluster's per-user limit (2 at a time on our cluster); the rest wait their turn.
An evaluation only runs if its training succeeded; if a training job fails, SLURM cancels its evaluation.

```bash
A2_ADAPTER=/path/to/qwen_4b_hinglish_lora bash submit_all_a3.sh
RUNS="D" bash submit_all_a3.sh
SKIP_PREPARE=1 bash submit_all_a3.sh
SEQUENTIAL=1 bash submit_all_a3.sh
```

`RUNS="D"` trains only the final model; `RUNS="E F"` adds the from-scratch runs (data step included); `BASELINES=0` skips the A and BASE evaluations if you have already run them; `SKIP_PREPARE=1` reuses an existing `data_a3/`; `SEQUENTIAL=1` runs the jobs
strictly one after another on a single GPU.

Or submit by hand, waiting for each step to finish (`squeue -u $USER`):

```bash
sbatch prepare_data_a3.sh

sbatch --export=ALL,RUN=BASE evaluate_a3.sh
sbatch --export=ALL,RUN=A evaluate_a3.sh
sbatch --export=ALL,RUN=D train_4b_a3.sh
sbatch --export=ALL,RUN=C train_4b_a3.sh
sbatch --export=ALL,RUN=B train_4b_a3.sh

sbatch --export=ALL,RUN=D evaluate_a3.sh
sbatch --export=ALL,RUN=D,USE_PERSONA=0 evaluate_a3.sh
sbatch --export=ALL,RUN=C evaluate_a3.sh
sbatch --export=ALL,RUN=B evaluate_a3.sh
```

`evaluate_a3.sh` refuses to run B, C or D until that run's training has finished, because `runs/run_<RUN>/best/`
exists from the first step of training and would otherwise give scores for an unfinished model.

If `A2_ADAPTER` is somewhere else, add it to each `--export`, e.g. `--export=ALL,RUN=D,A2_ADAPTER=/path/qwen_4b_hinglish_lora`.

| Run | Starts from | Trains on | Learning rate | Shows |
| --- | --- | --- | --- | --- |
| BASE | Qwen3-4B, no adapter | nothing | – | Untuned model |
| A | Assignment 2 adapter | nothing (already trained) | – | Baseline from last assignment |
| B | Qwen3-4B, new adapter | IndicTalk only | 2e-4 | What the new data does alone |
| C | Assignment 2 adapter | IndicTalk only | 1e-4 | Sequential training; check for forgetting on `test_a2` |
| D | Assignment 2 adapter | IndicTalk + 25% Assignment 2 data | 1e-4 | Recommended setup |
| D_persona0 | Run D | – | – | Same model without the persona line, to measure persona's effect |
| E | Qwen3-4B, new adapter | Assignment 2 sentences with the new errors (up to 150,000) | 2e-4 | Whether the better errors alone improve on A (same data, no Assignment 2 habits) |
| F | Qwen3-4B, new adapter | IndicTalk + Assignment 2 sentences, all with the new errors | 2e-4 | Everything from scratch; a cleaner final model than D, which inherits Assignment 2's mistakes |

Every setting in `train_4b_a3.py` can be changed through an environment variable of the same name,
e.g. `--export=ALL,RUN=D,MAX_TRAIN_EXAMPLES=50000` for a quick trial.

## What the data step produces (`data_a3/`)

| File | Contents |
| --- | --- |
| `train_indictalk.csv` | Up to 150,000 noisy/clean pairs from IndicTalk Romanized Hindi, with persona |
| `train_a2_replay.csv` | Assignment 2 messages re-noised with the new generator (25% of the training mix) |
| `val.csv` | 1,500 IndicTalk + 500 Assignment 2 pairs for validation loss |
| `test_indictalk.csv`, `test_a2.csv` | 2,000 pairs each, new noise |
| `test_a2_original.csv` | 2,000 of the exact Assignment 2 test pairs |
| `train_a2_full.csv` | Up to 150,000 Assignment 2 sentences with the new errors (made by `prepare_a2_full_a3.sh`, for runs E and F) |
| `coverage_report.txt`, `data_stats.json` | Count of each error type and other statistics per file |

IndicTalk is split by conversation, and no sentence in validation or test appears in training. About 15% of the
pairs are left uncorrupted, so the model learns to leave correct text alone.

## Errors added

The Assignment 2 noise in the repo is mostly English, SMS slang and typing errors. Some of its Hinglish swaps
went in the wrong direction (for example clean `bht` became noisy `bahut`, teaching the model to turn
`bahut` into `bht`). Others swapped two valid words (`main`/`mai`, `tha`/`thi`, `in`/`on`).
Those are removed or fixed. New errors are injected only where the context makes the result wrong:

| Tag | Example (correct → noisy) |
| --- | --- |
| `gender` | mera ghar → meri ghar, nayi kitab → naya kitab |
| `person` | main thak gaya hoon → main thak gaya ho |
| `number` | hum log ja rahe hain → hum log ja rahe hai |
| `honorific` | aap kya kar rahe hain → aap kya kar raha hai |
| `ne_drop` | maine khana khaya → main khana khaya |
| `ne_misuse` | main school gaya → maine school gaya |
| `compound_postposition` | dost ke liye → dost ki liye, barish ki wajah se → barish ke wajah se |
| `tense` | pichle hafte main bimaar tha → ... bimaar hai |
| `eng_agreement` | she has → she have, they are → they is |
| `eng_tense` | yesterday I went → yesterday I go |
| `aspirate_drop` | bhai → bai, kharab → karab |

## Metrics

| Metric | Meaning | Role |
| --- | --- | --- |
| chrF / chrF++ | Character n-gram overlap with the reference | Main; gives partial credit for near-miss Romanized spellings |
| GLEU | BLEU for grammar correction: rewards needed changes, penalises errors left in | Main; standard for this task |
| Edit precision / recall / F0.5 | Whether the words the model changed are the changes that were needed | Main |
| Over-correction rate | Share of already-correct inputs the model changed | Main |
| Exact match | Output identical to the reference | Kept for comparison with Assignment 2 |
| BLEU | Kept for comparison with Assignment 2 | Secondary |
| Semantic similarity (LaBSE) | Whether meaning is kept; off by default (`USE_SIMILARITY=1`) | Meaning check only |
| Copy-input baseline | Every metric for the unchanged noisy input | Reported beside every score |

Similarity is not used as a main metric: the noisy input already means the same as the reference, so copying it
unchanged scores very high. `results/<run>/metrics_<test>.json` also breaks results down by persona and by error type.

## Outputs

- `runs/run_<RUN>/best/`: adapter with the lowest validation loss (use this one); `final/` is the last step.
- `runs/run_<RUN>/train_summary.json`: trainable parameters, time, peak GPU memory, validation history.
- `results/<RUN>/predictions_<test>.csv`: every input, reference and prediction.
- `results/results_table.md`: all runs side by side.

## Using the trained adapter

Load `Qwen/Qwen3-4B`, attach `runs/run_D/best` with `PeftModel.from_pretrained`, and send the prompt from
`prompt_a3.py` (with `Persona: <friends|family|colleagues|experts|student_teacher|none>`), reading the
output up to the end-of-text token. The app's `colab/colab_server.py` currently loads `google/gemma-3-4b-it` with
a system prompt, so it needs this change to use the fine-tuned model.
