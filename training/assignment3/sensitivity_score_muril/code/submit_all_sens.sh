#!/bin/bash
# Run from the login node with bash (not sbatch):  bash submit_all_sens.sh
# Options: MODELS="muril qwen0.6b" (default) or MODELS=muril; KAGGLE_CSV=raw/<file>.csv to add the Kaggle set.
set -e
cd "$(dirname "$0")"

MODELS=${MODELS:-"muril qwen0.6b"}
DATA_DIR=${DATA_DIR:-data_sens}
RAW_DIR=${RAW_DIR:-raw}

for M in $MODELS; do
  case "$M" in
    muril|qwen0.6b) ;;
    *) echo "Unknown model $M in MODELS. Use muril and/or qwen0.6b."; exit 1 ;;
  esac
  if [ -d "runs/run_$M" ]; then
    echo "runs/run_$M already exists. Remove or rename it before training $M again."
    exit 1
  fi
done
if [ -n "$KAGGLE_CSV" ] && [ ! -f "$KAGGLE_CSV" ]; then
  echo "KAGGLE_CSV=$KAGGLE_CSV not found."
  exit 1
fi

# Public data from GitHub, downloaded on the login node.
mkdir -p "$RAW_DIR"
if [ ! -f "$RAW_DIR/trac-1/hindi/agr_hi_train.csv" ]; then
  git clone --depth 1 https://github.com/kmi-linguistics/trac-1 "$RAW_DIR/trac-1"
fi
if [ ! -f "$RAW_DIR/Hinglish-Offensive-Text-Classification/Hinglish_Profanity_List.csv" ]; then
  git clone --depth 1 https://github.com/pmathur5k10/Hinglish-Offensive-Text-Classification \
    "$RAW_DIR/Hinglish-Offensive-Text-Classification" || echo "Profanity list not downloaded; continuing without it."
fi

jobid() { echo "${1%%;*}"; }
submit() {
  local deps="$1"; shift
  if [ -n "$deps" ]; then
    JOB=$(jobid "$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=$deps "$@")")
  else
    JOB=$(jobid "$(sbatch --parsable "$@")")
  fi
  if ! [[ "$JOB" =~ ^[0-9]+$ ]]; then
    echo "Submission failed for: $*"
    echo "Nothing after this was submitted. Check squeue -u \$USER, cancel any jobs listed, and run again."
    exit 1
  fi
}

if [ -f "$DATA_DIR/train.jsonl" ] && [ -f "$DATA_DIR/test_trac.jsonl" ]; then
  echo "Using existing $DATA_DIR/"
  DATA_DEP=""
else
  submit "" --export=ALL,DATA_DIR=$DATA_DIR,RAW_DIR=$RAW_DIR,KAGGLE_CSV=$KAGGLE_CSV prepare_data_sens.sh
  DATA_DEP="afterok:$JOB"
  echo "Data step: job $JOB builds $DATA_DIR/"
fi

for M in $MODELS; do
  submit "$DATA_DEP" --export=ALL,MODEL=$M,DATA_DIR=$DATA_DIR train_sens.sh
  T=$JOB
  submit "afterok:$T" --export=ALL,MODEL=$M,RUN_NAME=$M,DATA_DIR=$DATA_DIR evaluate_sens.sh
  echo "$M: train job $T, evaluation job $JOB"
done

echo
echo "Check progress with: squeue -u \$USER"
echo "Results: results/results_table.md"
