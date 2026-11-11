#!/bin/bash
#SBATCH --job-name=sens-train
#SBATCH --output=sens_train_out_%j.log
#SBATCH --error=sens_train_err_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=12:00:00

cd $SLURM_SUBMIT_DIR

module load anaconda3-2024
source /apps/compilers/anaconda3-2024/etc/profile.d/conda.sh
conda activate qwen_env

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

MODEL=${MODEL:-muril}
case "$MODEL" in
  muril)
    export MODEL_NAME=${MODEL_NAME:-google/muril-base-cased}
    export LEARNING_RATE=${LEARNING_RATE:-2e-5}
    export BATCH_SIZE=${BATCH_SIZE:-32}
    export EPOCHS=${EPOCHS:-4}
    ;;
  qwen0.6b)
    export MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-0.6B-Base}
    export LEARNING_RATE=${LEARNING_RATE:-1e-5}
    export BATCH_SIZE=${BATCH_SIZE:-16}
    export GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-2}
    export EPOCHS=${EPOCHS:-3}
    ;;
  *)
    echo "Unknown MODEL=$MODEL. Use muril or qwen0.6b."
    exit 1
    ;;
esac
export RUN_NAME=${RUN_NAME:-$MODEL}
DATA_DIR=${DATA_DIR:-data_sens}

for f in "$DATA_DIR/train.jsonl" "$DATA_DIR/val.jsonl"; do
  if [ ! -f "$f" ]; then
    echo "$f not found. Run prepare_data_sens.sh first."
    exit 1
  fi
done

python3 -u train_sens.py
