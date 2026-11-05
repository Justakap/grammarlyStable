# Assignment 3: Hinglish Grammar Correction (Qwen3-4B + LoRA)

Continues the Assignment 2 grammar model (`training/assignment2/grammar_correction_qwen3_4b_lora`) with the
IndicTalk persona dataset, richer Hinglish errors and better evaluation. The scripts are in `code/` (the learning
rate and rank sweep in `code/sweep_lr_rank/`); results are in `results/`.

## Results

Edit F0.5 and over-correction on the two test sets (2,000 pairs each). All runs: [`results/results_table.md`](results/results_table.md).

| Run | F0.5 IndicTalk | F0.5 Assignment 2 | Over-correction IndicTalk | Over-correction Assignment 2 | Exact match IndicTalk / Assignment 2 |
| --- | --- | --- | --- | --- | --- |
| A (Assignment 2 model) | 81.72 | 85.29 | 42.40 | 38.44 | 38.85 / 63.40 |
| D | 91.85 | 89.90 | 1.94 | 6.31 | 70.75 / 76.75 |
| G1 | 91.43 | **90.10** | 1.94 | 6.31 | 70.35 / 77.50 |
| G2 | **92.08** | 89.88 | **1.29** | 6.31 | 71.75 / 76.90 |
| F | 90.27 | 87.61 | 1.94 | 6.61 | 68.35 / 73.85 |
| Copy input unchanged | 0 | 0 | 0 | 0 | 15.45 / 16.65 |

- The untuned Qwen3-4B (BASE) scores below simply copying the input, so fine-tuning is essential.
- Keeping 15% of training sentences unchanged cut over-correction from about 40% (run A) to 1-6%.
- The new errors alone help: run E beats A on `test_a2` with less than a quarter of the data.
- Training on one source does not transfer. B (IndicTalk only) changes 80% of already-correct Assignment 2
  messages; C (IndicTalk only, from the Assignment 2 model) forgets Assignment 2. Mixing in 25% Assignment 2
  sentences (D, F, G) avoids both, at no cost on IndicTalk.
- D, G1 and G2 are within 0.2 points of each other on mean F0.5. G1 (fresh model, about as much data as D in one
  stage) matches D, so D's lead over F came from the amount of data, not from starting at the Assignment 2 model.
  G2 (D's two-stage recipe with the new errors throughout) also matches D.
- The persona line has no measurable effect (D vs D_persona0).
- Learning rate (5e-5 to 2e-4) and LoRA rank (16 to 64) do not matter in this range: all five sweep settings are
  within 0.17 points ([`results/sweep_results.md`](results/sweep_results.md)).
- The best runs sit at about 92 (IndicTalk) and 90 (Assignment 2); recipe and hyperparameters are no longer the
  bottleneck. The final runs (G2_long, 8B_long) therefore add data: whole IndicTalk messages up to 120 words.

## Files (`code/`)

| File | What it does |
| --- | --- |
| `hinglish_noise.py` | Error generator: the Assignment 2 noise (fixed) plus new context-aware Hinglish grammar errors, each tagged |
| `prompt_a3.py` | The training and inference prompt (with a `Persona:` line) |
| `prepare_data_a3.py` / `prepare_data_a3.sh` | Builds `data_a3/` from IndicTalk (Hindi, Romanized only) and Assignment 2 data |
| `prepare_a2_full_a3.py` / `prepare_a2_full_a3.sh` | Builds Assignment 2 sentences with the new errors: `train_a2_full.csv` (runs E, F) or `train_a2_all.csv` (run G1) |
| `train_4b_a3.py` / `train_4b_a3.sh` | LoRA training; runs B, C, D, E, F, G1, G2, G2_long or 8B_long |
| `evaluate_grammar_a3.py` / `evaluate_a3.sh` | Generates corrections and computes all metrics; any run, on the original or the long-message test sets |
| `summarize_results.py` | Builds `results/results_table.md` from every evaluation so far |
| `submit_all_a3.sh` | Submits runs B-G and their evaluations in the right order using SLURM dependencies |
| `submit_final_long.sh` | Final runs G2_long (4B) and 8B_long on whole messages up to 120 words, with their data step and evaluations |
| `sweep_lr_rank/` | Learning rate and LoRA rank sweep on Data Mix D (see its README) |

## Before you start

1. Copy `code/` to the cluster and run everything from inside it.
2. Put the Assignment 2 adapter folder `qwen_4b_hinglish_lora/` (from the Google Drive zip) inside it, or pass its path with
   `A2_ADAPTER=/path/to/qwen_4b_hinglish_lora`.
3. Optional: copy `grammar_dataset_train.csv` and `grammar_dataset_test.csv` from Assignment 2 (also in the Drive zip) into it.
   Without them the script re-downloads the Assignment 2 dataset, and `test_a2_original.csv` (the exact
   Assignment 2 test pairs, for comparison with last time) is not created.
4. The `qwen_env` environment from Assignment 2 is used. `evaluate_a3.sh` installs `sacrebleu` if it is missing.

## Run order

Submit everything at once from the login node (with `bash`, not `sbatch`):

```bash
bash submit_all_a3.sh                                   # data step, runs D C B, evaluations, baselines A and BASE
RUNS="E F" SKIP_PREPARE=1 BASELINES=0 bash submit_all_a3.sh
RUNS="G" SKIP_PREPARE=1 BASELINES=0 bash submit_all_a3.sh
bash submit_final_long.sh                               # final runs G2_long and 8B_long
```

Each command exits in a few seconds. SLURM holds each job as `PENDING (Dependency)` until what it needs has
finished, and a pending job uses no GPU. Ready jobs run in parallel up to the per-user limit (2 at a time on our
cluster). An evaluation only runs if its training succeeded; if a training job fails, SLURM cancels its evaluation.
`SEQUENTIAL=1` runs `submit_all_a3.sh` strictly one job after another.

Single jobs can also be submitted by hand, e.g. `sbatch prepare_data_a3.sh`, `sbatch --export=ALL,RUN=D train_4b_a3.sh`,
`sbatch --export=ALL,RUN=D evaluate_a3.sh`, `sbatch --export=ALL,RUN=D,USE_PERSONA=0 evaluate_a3.sh`.
`evaluate_a3.sh` refuses to evaluate a run until its training has finished, because `runs/run_<RUN>/best/` exists
from the first step of training.

| Run | Starts from | Trains on | Learning rate | Shows |
| --- | --- | --- | --- | --- |
| BASE | Qwen3-4B, no adapter | nothing | – | Untuned model |
| A | Assignment 2 adapter | nothing (already trained) | – | Baseline from last assignment |
| B | Qwen3-4B, new adapter | IndicTalk only | 2e-4 | What the new data does alone |
| C | Assignment 2 adapter | IndicTalk only | 1e-4 | Sequential training; check for forgetting on `test_a2` |
| D | Assignment 2 adapter | Data Mix D: IndicTalk + 25% Assignment 2 sentences | 1e-4 | Two-stage recipe (about 760,000 pairs in total with Assignment 2) |
| D_persona0 | Run D | – | – | Same model without the persona line, to measure persona's effect |
| E | Qwen3-4B, new adapter | 150,000 Assignment 2 sentences with the new errors | 2e-4 | Whether the better errors alone improve on A |
| F | Qwen3-4B, new adapter | IndicTalk + 150,000 Assignment 2 sentences, new errors | 2e-4 | Everything from scratch, about 240,000 pairs |
| G1 | Qwen3-4B, new adapter | IndicTalk + all Assignment 2 sentences, new errors (about 700,000 pairs) | 2e-4 | Same amount of data as D, one stage from scratch |
| G2 | G1's adapter | Data Mix D | 1e-4 | D's two-stage recipe, with the new errors in every stage |
| G2_long | G1's adapter | Data Mix D rebuilt with whole messages up to 120 words (about 430,000 pairs) | 2e-4 | Final 4B model |
| 8B_long | Qwen3-8B, new adapter | Same as G2_long | 2e-4 | Larger model, one stage from scratch |

Every setting in `train_4b_a3.py` can be changed through an environment variable of the same name,
e.g. `--export=ALL,RUN=D,MAX_TRAIN_EXAMPLES=50000` for a quick trial.

## Run G: a fair comparison with D

D started from the Assignment 2 model, so in total it saw about 760,000 training pairs (about 640,000 in
Assignment 2 plus about 121,000 here), while F saw about 240,000. Run G removes that difference while starting
from a fresh Qwen3-4B: G1 trains in one stage on every Assignment 2 sentence plus IndicTalk, all with the new errors,
and G2 continues G1 on Data Mix D at learning rate 1e-4, exactly like D's second stage.

## Learning rate and rank sweep (`code/sweep_lr_rank/`)

Trains run D's setup with learning rates 5e-5, 1e-4 and 2e-4 (rank 32) and ranks 16 and 64 (learning rate 1e-4),
each on 50,000 pairs of Data Mix D, and picks a winner on the validation sets only. For other ranks the rank-32
Assignment 2 adapter is resized first: rank 64 adds directions that start at zero (same output before training),
rank 16 keeps the 16 most important directions of each layer's update. Results: [`results/sweep_results.md`](results/sweep_results.md).

## Final runs on long messages (G2_long and 8B_long)

```bash
bash submit_final_long.sh
```

Trains two models in parallel, each on its own 48 GB GPU:

| Run | Model | Starts from | Learning rate | Batch x accumulation |
| --- | --- | --- | --- | --- |
| G2_long | Qwen3-4B | `runs/run_G1/best` (G2's recipe) | 2e-4 (the sweep's winner) | 16 x 4 (about 28 GB) |
| 8B_long | Qwen3-8B | a new rank-32 adapter, one stage | 2e-4 | 8 x 8 (same effective batch of 64) |

It first builds `data_a3_full/` (below), then submits both trainings and, for each model, an evaluation on the
original test sets (added to `results/results_table.md`) and on the long-message IndicTalk test set
(`results_long/results_table.md`). Run G2 is also scored on the long-message test set for comparison.
Training uses `MAX_LENGTH=768` (the longest training example is about 400 tokens at the 99th percentile);
evaluation uses `MAX_NEW_TOKENS=512`.

The script reads G1's `training_minutes` and estimates G2_long at about 1.2 times G1 and 8B_long at about 2.6
times G1. If either would come close to the 48-hour job limit it stops and says what to change, since a job's
time limit cannot be raised after submission:

```bash
TRAIN_TIME_8B=3-00:00:00 bash submit_final_long.sh       # if the partition allows it: sinfo -p gpu -o '%P %l'
MAX_TRAIN_EXAMPLES_8B=280000 bash submit_final_long.sh   # or train the 8B on a random subset (same 75/25 mix)
RUNS=G2_long bash submit_final_long.sh                   # one model only
BATCH_SIZE_4B=8 GRAD_ACCUM_STEPS_4B=8 bash submit_final_long.sh   # if the 4B run runs out of memory
```

The first 8B job downloads Qwen3-8B (about 16 GB) into the Hugging Face cache. 8B_long is one stage from scratch
while G2_long has G1's stage behind it, so their difference mixes model size and recipe.

## What the data step produces (`data_a3/`)

| File | Contents |
| --- | --- |
| `train_indictalk.csv` | IndicTalk Romanized messages of 3-60 words, with errors and persona (roughly 90,000; the 150,000 cap is not reached) |
| `train_a2_replay.csv` | Assignment 2 messages with the new errors, 25% of the training mix (about 30,000) |
| `val.csv` | 1,500 IndicTalk + 500 Assignment 2 pairs for validation loss |
| `test_indictalk.csv`, `test_a2.csv` | 2,000 pairs each, new errors |
| `test_a2_original.csv` | 2,000 of the exact Assignment 2 test pairs (only if `grammar_dataset_test.csv` is present) |
| `train_a2_full.csv` | 150,000 Assignment 2 sentences with the new errors (`prepare_a2_full_a3.sh`, runs E and F) |
| `train_a2_all.csv` | All Assignment 2 sentences (about 600,000) with the new errors (`prepare_a2_full_a3.sh` with `OUT_FILE=train_a2_all.csv MAX_A2_TRAIN=0`, run G1) |
| `coverage_report.txt`, `data_stats.json` | Count of each error type, word edits per 100 words and other statistics per file |

IndicTalk is split by conversation, and no sentence in validation or test appears in training. About 15% of the
pairs are left uncorrupted, so the model learns to leave correct text alone. Each IndicTalk message (one speaker's
turn) is one example. IndicTalk messages are long (median 71 words), so the 60-word limit keeps only about 28% of
them; the log line `IndicTalk train messages available` gives the exact count.

### Long-message data (`data_a3_full/`)

Built by `submit_final_long.sh` for G2_long and 8B_long: 322,574 IndicTalk and 107,525 Assignment 2 pairs.

| Setting | Value | Effect |
| --- | --- | --- |
| `MAX_WORDS` | 120 | Keeps 99.5% of IndicTalk messages (`0` means no limit) |
| `MAX_INDICTALK_TRAIN` | 0 | No cap on IndicTalk messages |
| `GRAMMAR_SCALE_WORDS` | 40 | Raises the cap of 1-3 grammar errors per message for long messages (x2 from 60 words, x3 from 100) |
| `KEEP_SPLIT_FROM` | `data_a3` | Keeps every conversation in the same split as `data_a3/`, copies `test_a2.csv` and the Assignment 2 validation rows, and keeps every sentence any earlier run trained on out of validation and test, so both test sets can be used for any run without leakage |

## Errors added

The Assignment 2 noise is mostly English, SMS slang and typing errors. Its main problems were that it removed
every pair it left unchanged (so the model never learned to leave correct text alone) and that about 3% of its
corrupted messages contain a swap between two valid words without checking context (`raha`/`rahi`, `tha`/`thi`,
`is`/`are`, `in`/`on`), which can still be a correct sentence. Those swaps are removed (common English confusions such as
`their`/`there` and `then`/`than` are kept). New errors are injected
only where the context makes the result wrong:

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

Both generators measured on the same 5,000 sentences:

| | Assignment 2 noise | Assignment 3 noise |
| --- | --- | --- |
| Word edits per 100 words (Assignment 2 messages / IndicTalk) | 11.1 / 10.5 | 11.9 / 9.3 |
| Untouched pairs kept for training | 0% | about 15% |
| Corrupted messages with a context-checked grammar error | none | 22% / 40% |
| Hinglish verb-form, English agreement and preposition swaps without checking context | 3% | none |
| Error types | about 16, not tagged | 28, tagged |

The amount of corruption is about the same; the Assignment 3 errors are harder (they need the sentence's grammar
to fix) and cover more types. Coverage is uneven: `compound_postposition` appears in 25-34% of corrupted IndicTalk
messages, gender in 3-4% and tense in under 2%.

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
- `results_long/results_table.md`: G2_long, 8B_long and G2 on the long-message test set.

## Using the trained adapter

Load `Qwen/Qwen3-4B`, attach `runs/run_G2/best` (or `runs/run_D/best`; they score the same) with
`PeftModel.from_pretrained`, and send the prompt from `prompt_a3.py` (with
`Persona: <friends|family|colleagues|experts|student_teacher|none>`), reading the output up to the end-of-text
token. For the 8B model, load `Qwen/Qwen3-8B` and attach `runs/run_8B_long/best`. The app's
`colab/colab_server.py` currently loads `google/gemma-3-4b-it` with a system prompt, so it needs this change to use
the fine-tuned model.
