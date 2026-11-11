#!/bin/bash
#SBATCH --job-name=a3-sweep-train
#SBATCH --output=sweep_train_out_%j.log
#SBATCH --error=sweep_train_err_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=2-00:00:00

cd $SLURM_SUBMIT_DIR

module load anaconda3-2024
source /apps/compilers/anaconda3-2024/etc/profile.d/conda.sh
conda activate qwen_env

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

LR=${LR:?Set LR, e.g. LR=1e-4}
RANK=${RANK:-32}
A2_ADAPTER=${A2_ADAPTER:-../qwen_4b_hinglish_lora}
NAME=${SWEEP_NAME:-lr${LR}_r${RANK}}

if [ ! -f "$A2_ADAPTER/adapter_config.json" ]; then
  echo "Assignment 2 adapter not found at $A2_ADAPTER. Set A2_ADAPTER=/path/to/qwen_4b_hinglish_lora"
  exit 1
fi
for f in ../data_a3/train_indictalk.csv ../data_a3/train_a2_replay.csv ../data_a3/val.csv ../train_4b_a3.py; do
  if [ ! -f "$f" ]; then
    echo "$f not found. This folder must sit inside the Assignment 3 folder, after prepare_data_a3.sh has run."
    exit 1
  fi
done

A2_RANK=$(python3 -c "import json;print(json.load(open('$A2_ADAPTER/adapter_config.json'))['r'])")
if [ "$RANK" = "$A2_RANK" ]; then
  INIT="$A2_ADAPTER"
else
  INIT="adapters/a2_r${RANK}"
  if [ ! -f "$INIT/adapter_config.json" ]; then
    SRC_ADAPTER="$A2_ADAPTER" DST_ADAPTER="$INIT" RANK="$RANK" python3 -u resize_adapter.py || exit 1
  fi
fi

export RUN_NAME="$NAME"
export INIT_ADAPTER="$INIT"
export OUTPUT_DIR="runs/$NAME"
export LEARNING_RATE="$LR"
export TRAIN_FILES="../data_a3/train_indictalk.csv,../data_a3/train_a2_replay.csv"
export VAL_FILE="../data_a3/val.csv"
export MAX_TRAIN_EXAMPLES=${MAX_TRAIN_EXAMPLES:-50000}
export EVAL_EVERY=${EVAL_EVERY:-200}
export USE_PERSONA=1

python3 -u ../train_4b_a3.py
