#!/bin/bash
#SBATCH --job-name=a3-grammar-train
#SBATCH --output=a3_train_out_%j.log
#SBATCH --error=a3_train_err_%j.log
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

RUN=${RUN:-D}
A2_ADAPTER=${A2_ADAPTER:-qwen_4b_hinglish_lora}

case "$RUN" in
  B)
    export INIT_ADAPTER=""
    export TRAIN_FILES="data_a3/train_indictalk.csv"
    export LEARNING_RATE=${LEARNING_RATE:-2e-4}
    ;;
  C)
    export INIT_ADAPTER="$A2_ADAPTER"
    export TRAIN_FILES="data_a3/train_indictalk.csv"
    export LEARNING_RATE=${LEARNING_RATE:-1e-4}
    ;;
  D)
    export INIT_ADAPTER="$A2_ADAPTER"
    export TRAIN_FILES="data_a3/train_indictalk.csv,data_a3/train_a2_replay.csv"
    export LEARNING_RATE=${LEARNING_RATE:-1e-4}
    ;;
  E)
    export INIT_ADAPTER=""
    export TRAIN_FILES="data_a3/train_a2_full.csv"
    export LEARNING_RATE=${LEARNING_RATE:-2e-4}
    ;;
  F)
    export INIT_ADAPTER=""
    export TRAIN_FILES="data_a3/train_indictalk.csv,data_a3/train_a2_full.csv"
    export LEARNING_RATE=${LEARNING_RATE:-2e-4}
    ;;
  *)
    echo "Unknown RUN=$RUN. Use B, C, D, E or F."
    exit 1
    ;;
esac

if [ -n "$INIT_ADAPTER" ] && [ ! -f "$INIT_ADAPTER/adapter_config.json" ]; then
  echo "Assignment 2 adapter not found at $INIT_ADAPTER. Set A2_ADAPTER=/path/to/qwen_4b_hinglish_lora"
  exit 1
fi
for f in ${TRAIN_FILES//,/ }; do
  if [ ! -f "$f" ]; then
    echo "$f not found. Run prepare_data_a3.sh first (and prepare_a2_full_a3.sh for runs E and F)."
    exit 1
  fi
done

export RUN_NAME="$RUN"
export OUTPUT_DIR="runs/run_$RUN"

python3 -u train_4b_a3.py
