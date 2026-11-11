#!/bin/bash
set -e
cd "$(dirname "$0")"

LR=${1:?Usage: bash submit_final.sh <learning rate> <rank>, e.g. bash submit_final.sh 1e-4 32}
RANK=${2:?Usage: bash submit_final.sh <learning rate> <rank>}
export A2_ADAPTER=${A2_ADAPTER:-../qwen_4b_hinglish_lora}
NAME="final_lr${LR}_r${RANK}"

jobid() { echo "${1%%;*}"; }
T=$(jobid "$(sbatch --parsable --export=ALL,LR=$LR,RANK=$RANK,SWEEP_NAME=$NAME,MAX_TRAIN_EXAMPLES=0,EVAL_EVERY=500 sweep_train.sh)")
if ! [[ "$T" =~ ^[0-9]+$ ]]; then
  echo "Submission failed"
  exit 1
fi
E=$(jobid "$(sbatch --parsable --kill-on-invalid-dep=yes --dependency=afterok:$T --export=ALL,SWEEP_NAME=$NAME,EVAL_SET=test sweep_eval.sh)")
echo "$NAME: train job $T (full Data Mix D) | test evaluation job $E"
echo "Test results: results_test/results_table.md (compare with run D in ../results/results_table.md)"
