# Assignment 3: Persona-Aware Next-Word Prediction (Qwen3-0.6B and 1.7B, Full Fine-Tuning)

Continues the Assignment 2 autocomplete model (`training/assignment2/autocomplete_qwen3_0.6b_full_finetune`):
same base model and full fine-tuning, but now conditioned on the conversation's persona and trained on both
datasets. The scripts are in `code/`; results are in `results/`.

## Results

Word-level scores on the same 3,000 words per test set, with the true persona for the Assignment 3 models.
All runs and persona modes: [`results/results_table.md`](results/results_table.md).

| Model | Perplexity IndicTalk / Assignment 2 | Word top-1 IndicTalk / Assignment 2 | Word top-3 IndicTalk / Assignment 2 | Keystroke savings IndicTalk / Assignment 2 |
| --- | --- | --- | --- | --- |
| BASE (untuned Qwen3-0.6B-Base) | 54.76 / 46.72 | 16.17 / 21.10 | 26.87 / 32.80 | 21.52 / 29.34 |
| A2 (Assignment 2 model, 0.6B) | 31.70 / 15.03 | 21.10 / 30.83 | 32.63 / 46.30 | 26.74 / 42.40 |
| 0.6B | 5.92 / 5.20 | 41.53 / 46.87 | 61.17 / 65.17 | 54.35 / 61.03 |
| 1.7B | **4.76** / **4.18** | **46.30** / **50.47** | **65.67** / **68.70** | **59.60** / **65.10** |

Persona ablation on the IndicTalk test set (same model, different persona line):

| Model | Persona | Perplexity | Word top-3 | Keystroke savings |
| --- | --- | --- | --- | --- |
| 0.6B | true | **5.92** | **61.17** | **54.35** |
| 0.6B | none | 6.16 | 60.40 | 53.58 |
| 0.6B | wrong | 6.19 | 59.67 | 52.84 |
| 1.7B | true | **4.76** | **65.67** | **59.60** |
| 1.7B | none | 4.94 | 64.50 | 58.56 |
| 1.7B | wrong | 5.01 | 64.63 | 58.59 |

- At the same size as the Assignment 2 model, the 0.6B model almost doubles word top-3 on IndicTalk (32.63 to
  61.17) and also improves on Assignment 2's own test set (46.30 to 65.17).
- The 1.7B model adds 3.5 to 5.2 points of word top-1, top-3 and keystroke savings over the 0.6B model on both
  test sets.
- The persona has a small, consistent effect: the true persona gives the lowest perplexity for both sizes, `none`
  is next and a wrong persona is the highest. The word-level differences are about 1 point or less.

| | Assignment 2 | Assignment 3 |
| --- | --- | --- |
| Data | `theguywithblacktie/hinglish-conversations` | The same + IndicTalk (Hindi, Romanized only), with persona |
| Input format | All messages of a conversation joined with spaces | A persona line, then one message per line |
| Long conversations | Truncated at 1,024 tokens | Split at message boundaries (at most 450 words per example) |
| Persona | – | `Persona: friends / family / colleagues / experts / student_teacher / none` |
| Metrics | Token-level loss, perplexity, top-1/top-3 | The same + word-level suggestions (top-1, top-3, MRR, keystroke savings) and a persona ablation |
| Baselines | – | Untuned Qwen3-0.6B-Base and the Assignment 2 model on the same test sets |

## Model size

Qwen3-0.6B-Base is the default. Autocomplete runs on every keystroke, so a small model keeps suggestions fast,
and using the same model as Assignment 2 means any gain comes from the persona and the data. Qwen3-1.7B-Base can
be trained with the same scripts (`SIZES="0.6B 1.7B"`) to see whether a larger model is worth about three times
the cost; it fits on one 48 GB L40S with full fine-tuning at batch 4 x 8.

## Files (`code/`)

| File | What it does |
| --- | --- |
| `nwp_common.py` | Persona line, text format, splitting long conversations, word positions for evaluation |
| `prepare_data_nwp.py` / `prepare_data_nwp.sh` | Builds `data_nwp/` from both datasets |
| `train_next_word.py` / `train_next_word.sh` | Full fine-tuning (`SIZE=0.6B` or `SIZE=1.7B`) |
| `evaluate_next_word.py` / `evaluate_next_word.sh` | Token-level and word-level metrics; trained runs, `BASE` or `A2` |
| `summarize_nwp.py` | Builds `results/results_table.md` |
| `submit_all_nwp.sh` | Submits the data step, training and all evaluations with SLURM dependencies |

## Before you start

1. Copy `code/` to the cluster and run everything from inside it. It does not use any other folder.
2. The `qwen_env` environment from Assignment 2 is used; both datasets are downloaded from Hugging Face.
3. For the Assignment 2 baseline, find the Assignment 2 autocomplete model folder (`autocomplete_model_qwen3_0.6b/`
   from the Google Drive zip, or `outputs/qwen3-0.6b-hinglish-next-token/final_model` where Assignment 2 was trained).

## Running

From the login node (with `bash`, not `sbatch`):

```bash
A2_MODEL=/path/to/autocomplete_model_qwen3_0.6b bash submit_all_nwp.sh
```

This submits, with dependencies: the data step, training of the 0.6B model, its evaluations (true persona on both
test sets, persona `none` and a wrong persona on the IndicTalk test set), and the baselines (untuned Qwen3-0.6B-Base
and the Assignment 2 model). Options:

```bash
SIZES="0.6B 1.7B" A2_MODEL=... bash submit_all_nwp.sh   # also train and evaluate the 1.7B model
BASELINES=0 bash submit_all_nwp.sh                      # skip the BASE and A2 evaluations
sbatch --export=ALL,SIZE=0.6B,MAX_TRAIN_EXAMPLES=20000 train_next_word.sh   # quick trial
sbatch --export=ALL,RUN=0.6B,PERSONA_MODE=none evaluate_next_word.sh         # a single evaluation
```

Every setting in `train_next_word.py` and `evaluate_next_word.py` can be changed through an environment variable of
the same name.

## Data (`data_nwp/`)

| File | Contents |
| --- | --- |
| `train.jsonl` | IndicTalk training conversations (94%) and Assignment 2 training conversations, shuffled |
| `val.jsonl` | 2% of IndicTalk conversations and 2,000 Assignment 2 conversations, for validation loss |
| `test_indictalk.jsonl` | 4% of IndicTalk conversations |
| `test_a2.jsonl` | The Assignment 2 test split (the same conversations as Assignment 2's test set) |
| `data_stats.json`, `prepare_data_nwp.log` | Counts per source and persona, words per example |

Each line is `{"id", "source", "persona", "turns"}`. IndicTalk is split by conversation, so no conversation
appears in two files. Assignment 2 conversations have persona `none`. A conversation longer than 450 words is split
into several examples at message boundaries, so nothing is cut off at the 1,024-token limit.

## Training

The model sees:

```
Persona: friends
<message 1>
<message 2>
...<|endoftext|>
```

The persona line is not part of the loss, so the scores measure only the conversation text. For 15% of IndicTalk
training examples the persona is replaced by `none` (`PERSONA_DROPOUT`), so the model also works when the persona is
unknown and the persona's effect can be measured within one model.

| Setting | 0.6B | 1.7B |
| --- | --- | --- |
| Base model | Qwen/Qwen3-0.6B-Base | Qwen/Qwen3-1.7B-Base |
| Method | Full fine-tuning, fp32 weights, bf16 autocast, gradient checkpointing | Same |
| Batch x accumulation | 8 x 4 (effective 32, as in Assignment 2) | 4 x 8 (effective 32) |
| Learning rate / warm-up / epochs | 1e-5 / 3% / 1 (as in Assignment 2) | Same |
| Max length | 1,024 tokens | Same |

Batches group examples of similar length, so short Assignment 2 conversations are not padded to the length of long
IndicTalk ones. Validation runs every 1,000 steps on each source; `best/` is the model with the lowest validation
loss.

## Evaluation

Token-level, on the conversation text: loss, perplexity, and next-token top-1/top-3 accuracy (the Assignment 2
metrics).

Word-level, on 3,000 randomly chosen words per test set (not the first word of a message): the model sees the
conversation up to the end of the previous word and proposes the 3 most likely next words, the way a suggestion bar
would. Reported:

| Metric | Meaning |
| --- | --- |
| `word_top1` | The first suggestion is the next word |
| `word_top3` | The next word is among the 3 suggestions |
| `word_mrr` | Mean reciprocal rank of the next word among the suggestions (0 if absent) |
| `keystroke_savings` | Share of characters (word + space) the user would not type by tapping a correct suggestion at the start of each word |

Persona ablation on the IndicTalk test set, same model: `true` (the conversation's persona), `none`, and `wrong`
(a different persona). True better than none shows the model uses the persona; wrong worse than none shows it relies
on it. `results/<run>__persona_<mode>/metrics_<test>.json` also breaks every metric down by persona, and
`word_predictions_<test>.csv` lists each scored word with its suggestions.

The Assignment 2 model is evaluated without a persona line and with messages joined by spaces, as it was trained.
Word-level metrics use exactly the same words for every model, so they compare directly; token counts differ
slightly between the two formats.

## Outputs

- `runs/run_<size>/best/`: fine-tuned model with the lowest validation loss (use this one); `final/` is the last step.
- `runs/run_<size>/train_summary.json`: parameters, time, peak GPU memory, validation history.
- `results/results_table.md`: all runs and persona modes side by side.

## Using the model

Load `runs/run_0.6B/best` with `AutoModelForCausalLM.from_pretrained`. Give it `Persona: <persona>\n`, the previous
messages one per line, and the current message up to the last complete word (without a trailing space). For three
suggestions, take the 3 most likely next tokens that start with a space and complete each one greedily until the
next space, as `predict_words` in `evaluate_next_word.py` does.
