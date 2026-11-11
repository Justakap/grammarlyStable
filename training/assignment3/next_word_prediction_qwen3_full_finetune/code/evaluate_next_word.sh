#!/bin/bash
#SBATCH --job-name=nwp-eval
#SBATCH --output=nwp_eval_out_%j.log
#SBATCH --error=nwp_eval_err_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

cd $SLURM_SUBMIT_DIR

module load anaconda3-2024
source /apps/compilers/anaconda3-2024/etc/profile.d/conda.sh
conda activate qwen_env

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

RUN=${RUN:-0.6B}
DATA_DIR=${DATA_DIR:-data_nwp}

case "$RUN" in
  BASE)
    export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B-Base}
    export PERSONA_MODE=off
    ;;
  A2)
    if [ -z "$A2_MODEL" ] || [ ! -f "$A2_MODEL/config.json" ]; then
      echo "Set A2_MODEL=/path/to/the/Assignment 2 autocomplete model folder (final_model)"
      exit 1
    fi
    export MODEL_PATH="$A2_MODEL"
    export PERSONA_MODE=off
    export TURN_SEPARATOR=space
    ;;
  *)
    if [ ! -f "runs/run_$RUN/train_summary.json" ]; then
      echo "Training for run $RUN has not finished (runs/run_$RUN/train_summary.json missing)."
      exit 1
    fi
    export MODEL_PATH="runs/run_$RUN/best"
    export PERSONA_MODE=${PERSONA_MODE:-true}
    ;;
esac
export RUN_NAME="$RUN"

if [ ! -f "$DATA_DIR/test_indictalk.jsonl" ]; then
  echo "$DATA_DIR/ not found. Run prepare_data_nwp.sh first."
  exit 1
fi

python3 -u evaluate_next_word.py
python3 -u summarize_nwp.py
