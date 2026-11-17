#!/bin/bash
#SBATCH --job-name=nwp-train
#SBATCH --output=nwp_train_out_%j.log
#SBATCH --error=nwp_train_err_%j.log
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

SIZE=${SIZE:-0.6B}
case "$SIZE" in
  0.6B)
    export MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-0.6B-Base}
    export BATCH_SIZE=${BATCH_SIZE:-8}
    export GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
    ;;
  1.7B)
    export MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-1.7B-Base}
    export BATCH_SIZE=${BATCH_SIZE:-4}
    export GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
    export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
    ;;
  *)
    echo "Unknown SIZE=$SIZE. Use 0.6B or 1.7B."
    exit 1
    ;;
esac
export RUN_NAME=${RUN_NAME:-$SIZE}
export OUTPUT_DIR="runs/run_$RUN_NAME"
DATA_DIR=${DATA_DIR:-data_nwp}

for f in "$DATA_DIR/train.jsonl" "$DATA_DIR/val.jsonl"; do
  if [ ! -f "$f" ]; then
    echo "$f not found. Run prepare_data_nwp.sh first."
    exit 1
  fi
done

python3 -u train_next_word.py
