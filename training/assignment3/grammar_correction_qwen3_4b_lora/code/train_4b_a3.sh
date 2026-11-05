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
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RUN=${RUN:-D}
A2_ADAPTER=${A2_ADAPTER:-qwen_4b_hinglish_lora}
LONG_DATA=${LONG_DATA:-data_a3_full}

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
  G1)
    export INIT_ADAPTER=""
    export TRAIN_FILES="data_a3/train_indictalk.csv,data_a3/train_a2_all.csv"
    export LEARNING_RATE=${LEARNING_RATE:-2e-4}
    export EVAL_EVERY=${EVAL_EVERY:-1000}
    ;;
  G2)
    if [ ! -f runs/run_G1/train_summary.json ]; then
      echo "Run G1 has not finished (runs/run_G1/train_summary.json missing). G2 continues from G1."
      exit 1
    fi
    export INIT_ADAPTER="runs/run_G1/best"
    export TRAIN_FILES="data_a3/train_indictalk.csv,data_a3/train_a2_replay.csv"
    export LEARNING_RATE=${LEARNING_RATE:-1e-4}
    ;;
  G2_long)
    if [ ! -f runs/run_G1/train_summary.json ]; then
      echo "Run G1 has not finished (runs/run_G1/train_summary.json missing). G2_long continues from G1."
      exit 1
    fi
    export INIT_ADAPTER="runs/run_G1/best"
    export TRAIN_FILES="$LONG_DATA/train_indictalk.csv,$LONG_DATA/train_a2_replay.csv"
    export VAL_FILE="$LONG_DATA/val.csv"
    export LEARNING_RATE=${LEARNING_RATE:-2e-4}
    export MAX_LENGTH=${MAX_LENGTH:-768}
    export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
    ;;
  8B_long)
    export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-8B}
    export INIT_ADAPTER=""
    export TRAIN_FILES="$LONG_DATA/train_indictalk.csv,$LONG_DATA/train_a2_replay.csv"
    export VAL_FILE="$LONG_DATA/val.csv"
    export LEARNING_RATE=${LEARNING_RATE:-2e-4}
    export MAX_LENGTH=${MAX_LENGTH:-768}
    export BATCH_SIZE=${BATCH_SIZE:-8}
    export GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
    export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
    ;;
  *)
    echo "Unknown RUN=$RUN. Use B, C, D, E, F, G1, G2, G2_long or 8B_long."
    exit 1
    ;;
esac

if [ -n "$INIT_ADAPTER" ] && [ ! -f "$INIT_ADAPTER/adapter_config.json" ]; then
  echo "Assignment 2 adapter not found at $INIT_ADAPTER. Set A2_ADAPTER=/path/to/qwen_4b_hinglish_lora"
  exit 1
fi
for f in ${TRAIN_FILES//,/ } ${VAL_FILE:-data_a3/val.csv}; do
  if [ ! -f "$f" ]; then
    echo "$f not found. Run prepare_data_a3.sh first (submit_final_long.sh builds $LONG_DATA/ for G2_long and 8B_long) (and prepare_a2_full_a3.sh for runs E and F, or with OUT_FILE=train_a2_all.csv MAX_A2_TRAIN=0 for run G1)."
    exit 1
  fi
done

export RUN_NAME="$RUN"
export OUTPUT_DIR="runs/run_$RUN"

python3 -u train_4b_a3.py
