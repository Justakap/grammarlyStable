# Assignment 3: Hinglish Sensitivity Score (0–100%)

A classifier that gives every Romanized Hinglish message a **sensitivity score from 0% (safe) to 100% (offensive or hateful)**, so the app can warn before a message is sent. It sits next to the grammar corrector and the autocomplete as a separate model. The scripts are in `code/`; results are in `results/`.

## Results

TRAC-1 Hindi test set (2,160 Romanized Facebook comments, the official dev split, never trained on). All numbers and sample messages: [`results/results_table.md`](results/results_table.md).

| Model | AUROC NAG vs OAG | AUROC NAG vs aggressive | Band accuracy | Band macro-F1 | OAG F1 at 50% | ECE before / after |
| --- | --- | --- | --- | --- | --- | --- |
| MuRIL | **87.73** | **82.00** | **59.68** | **57.55** | **66.82** | 13.80 / 13.80 |
| Qwen3-0.6B | 87.18 | 81.88 | 58.47 | 54.07 | 65.66 | 13.72 / 15.11 |

Everyday chat (1,000 messages per set; share scored 30% or more, lower is better):

| Model | IndicTalk | Assignment 2 |
| --- | --- | --- |
| MuRIL | 0.0% | 0.3% |
| Qwen3-0.6B | 0.0% | 0.4% |

- MuRIL is slightly better on every ranking and band metric with under half the parameters (237M vs 596M), and trains in
  1.5 minutes against 4.6.
- Almost no false alarms on everyday chat.
- The score partly follows writing style: safe chat averages about 2, while non-aggressive TRAC-1 comments
  average 34 (MuRIL) and 39 (Qwen3-0.6B). Every aggressive training example is a Facebook comment and almost every
  safe one is chat.
- The profanity check set (81 messages) averages 2.3 (MuRIL), but it is not a valid check: the word list matches
  everyday words such as `banda`, `chod` (to leave) and `meetha`, so most of those messages are harmless.
- Temperature scaling did not improve calibration: MuRIL's fitted temperature is 1.0 (ECE stays 13.80) and
  Qwen3-0.6B's ECE rises from 13.72 to 15.11.

| Score | Band | Meaning |
| --- | --- | --- |
| under 30% | safe | Normal message |
| 30–70% | sensitive | Rude, aggressive or borderline: check the tone |
| 70% and over | offensive | Abusive, aggressive or hateful |

## How the score works

- The model has one output (a logit). Training uses binary cross-entropy against a target between 0 and 1:

  | Data | Label | Target |
  | --- | --- | --- |
  | TRAC-1 | NAG (not aggressive) | 0.0 |
  | TRAC-1 | CAG (covertly aggressive: sarcasm, veiled attacks) | 0.5 (`CAG_TARGET`) |
  | TRAC-1 | OAG (overtly aggressive: abuse, threats, slurs) | 1.0 |
  | Kaggle (optional) | non-hate / hate | 0.0 / 1.0 |
  | IndicTalk and Assignment 2 messages without profanity | safe chat | 0.0 |

- Score = 100 × sigmoid(logit / T). T is fitted on the validation set after training (temperature scaling), so the percentages are calibrated: messages scored around 80% should be offensive about 80% of the time.

## Data (all Romanized Hinglish)

| Source | What it is | Used for |
| --- | --- | --- |
| [TRAC-1 Hindi](https://github.com/kmi-linguistics/trac-1) (Kumar et al., 2018) | Facebook comments, human-labelled NAG / CAG / OAG. 15,001 comments, of which about 75% are Romanized; Devanagari ones are dropped | Train/val (official train split, 90/10) and test (official dev split, never trained on) |
| [HOT Hinglish profanity list](https://github.com/pmathur5k10/Hinglish-Offensive-Text-Classification) (Mathur et al., 2018) | 209 Hinglish swear words with severity 1–10 | Removes profane Assignment 2 messages from the "safe" set; builds a profanity check set |
| [IndicTalk](https://huggingface.co/datasets/LingoIITGN/IndicTalk) (Romanized) | Persona conversations | 2,500 / 300 / 1,000 safe messages (train / val / test), split by conversation |
| [Assignment 2 data](https://huggingface.co/datasets/theguywithblacktie/hinglish-conversations) | Casual chat messages | 2,500 / 300 safe messages from train; 1,000 safe + up to 1,000 profanity messages from test |
| Kaggle Hinglish hate-speech CSV (optional, `KAGGLE_CSV`) | Binary hate labels | Only rows marked Hinglish/code-mixed and in Roman script; 80/10/10 split |

Not usable: the Bohra et al. (2018) dataset is published as tweet IDs only, and the HOT tweets are only shared on request.

### Data files (`data_sens/`)

| File | Contents |
| --- | --- |
| `train.jsonl`, `val.jsonl` | TRAC-1 train split + Kaggle (if given) + safe chat |
| `test_trac.jsonl` | TRAC-1 official dev split (Romanized) |
| `test_kaggle.jsonl` | Kaggle test split (only with `KAGGLE_CSV`) |
| `test_safe_indictalk.jsonl`, `test_safe_a2.jsonl` | Everyday messages: false-alarm check |
| `test_a2_lexicon.jsonl` | A2 test messages with a strong swear word (severity ≥ 5): should score high |
| `data_stats.json`, `prepare_data_sens.log` | Counts per source and label |

Each line: `{"text", "target", "label", "source"}`.

## Models

| | MuRIL (default) | Qwen3-0.6B (comparison) |
| --- | --- | --- |
| Base | google/muril-base-cased (~237M), pretrained on Indian languages incl. transliterated text | Qwen/Qwen3-0.6B-Base, same family as the autocomplete |
| Head | One output (sequence classification) | Same |
| Training | Full fine-tuning, fp32 weights, bf16 autocast, LR 2e-5, batch 32, 4 epochs | Full fine-tuning, LR 1e-5, batch 16 × 2, 3 epochs |
| Max length | 128 tokens | Same |
| Selection | Lowest validation loss, checked every epoch (`best/`) | Same |

## Evaluation

| Test | Metrics |
| --- | --- |
| TRAC-1 test | AUROC for NAG vs OAG and NAG vs aggressive (CAG+OAG); mean score per label; band accuracy and macro-F1 (NAG→safe, CAG→sensitive, OAG→offensive); OAG precision/recall/F1 at 50%; calibration error (ECE) before and after temperature scaling |
| Kaggle test | AUROC; precision/recall/F1 at 50%; ECE before and after |
| Safe chat (IndicTalk, A2) | Mean score; share of messages at or above 30% and 70% (false alarms; lower is better) |
| A2 profanity set | Same; higher is better |
| Six sample messages | Scores printed in the log and table |

## Running

Copy `code/` to the cluster and, from inside it on the login node:

```bash
bash submit_all_sens.sh                       # downloads TRAC-1 and the profanity list, builds data, trains and evaluates MuRIL and Qwen3-0.6B
MODELS=muril bash submit_all_sens.sh          # MuRIL only
KAGGLE_CSV=raw/hinglish_hate.csv bash submit_all_sens.sh   # also use a Kaggle CSV uploaded to raw/
```

Single steps:

```bash
sbatch prepare_data_sens.sh
sbatch --export=ALL,MODEL=muril train_sens.sh
sbatch --export=ALL,MODEL=muril evaluate_sens.sh
sbatch --export=ALL,MODEL=muril,MAX_TRAIN_EXAMPLES=2000,EPOCHS=1,RUN_NAME=trial train_sens.sh   # quick trial
```

Every setting in the Python scripts can be changed with an environment variable of the same name (e.g. `CAG_TARGET=0.3`, `N_SAFE_A2=5000,300,1000`). For a Kaggle CSV with unusual column names, set `KAGGLE_TEXT_COL`, `KAGGLE_LABEL_COL` and `KAGGLE_LANG_COL`.

## Outputs

- `runs/run_<model>/best/`: model with the lowest validation loss, plus `calibration.json` (temperature). `final/` is the last epoch.
- `runs/run_<model>/train_summary.json`: settings, time, peak GPU memory, validation history, calibration.
- `results/<model>/metrics.json`, `predictions_<test>.csv`, `examples_<test>.md` (lowest and highest scores).
- `results/results_table.md`: all models side by side.

## Using the model

```bash
python3 score_text.py "tu pagal hai kya" "kal milte hai yaar"
MODEL_PATH=runs/run_qwen0.6b/best python3 score_text.py "..."
```

## Limitations

- TRAC-1 labels aggression, not hate speech specifically. CAG is political sarcasm or veiled attacks, so it gets a middle target (0.5) rather than "offensive".
- TRAC-1 comes from 2018 Facebook comments about news and politics. Everyday chat is covered by the IndicTalk and Assignment 2 safe messages, but those are only "safe" by assumption (Assignment 2 messages with a known swear word are removed).
- The profanity check set uses a word list, so it measures profanity, not every kind of offensive message.
- Scores reflect the annotators' view; a high score is a prompt to re-read, not a verdict.
