#!/bin/bash
set -e
cd "$(dirname "$0")"

LRS=${LRS:-"5e-5 1e-4 2e-4"}
RANKS=${RANKS:-"16 64"}
RANK_LR=${RANK_LR:-1e-4}
export A2_ADAPTER=${A2_ADAPTER:-../qwen_4b_hinglish_lora}
export MAX_TRAIN_EXAMPLES=${MAX_TRAIN_EXAMPLES:-50000}

if [ ! -f "$A2_ADAPTER/adapter_config.json" ]; then
  echo "Assignment 2 adapter not found at $A2_ADAPTER. Run: A2_ADAPTER=/path/to/qwen_4b_hinglish_lora bash submit_sweep.sh"
  exit 1
fi
if [ ! -f ../data_a3/val.csv ] || [ ! -f ../train_4b_a3.py ]; then
  echo "../data_a3/ or ../train_4b_a3.py not found. Put this folder inside the Assignment 3 folder after data preparation."
  exit 1
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
    echo "Nothing after this was submitted. Fix the error above, check squeue -u \$USER, cancel any jobs listed, and run again."
    exit 1
  fi
}

CONFIGS=""
for LR in $LRS; do
  CONFIGS="$CONFIGS ${LR}:32"
done
for RANK in $RANKS; do
  if [ "$RANK" = "32" ] && [[ " $LRS " == *" $RANK_LR "* ]]; then
    continue
  fi
  CONFIGS="$CONFIGS ${RANK_LR}:${RANK}"
done

echo "Training on $MAX_TRAIN_EXAMPLES pairs of Data Mix D per run, continuing from $A2_ADAPTER"
for C in $CONFIGS; do
  LR=${C%%:*}
  RANK=${C##*:}
  NAME="lr${LR}_r${RANK}"
  submit "" --export=ALL,LR=$LR,RANK=$RANK,SWEEP_NAME=$NAME sweep_train.sh
  T=$JOB
  submit "afterok:$T" --export=ALL,SWEEP_NAME=$NAME,EVAL_SET=val sweep_eval.sh
  printf "%-16s train job %s | validation eval job %s\n" "$NAME" "$T" "$JOB"
done

echo
echo "Check progress with: squeue -u \$USER"
echo "Ranking after each evaluation: results/selection.md"
