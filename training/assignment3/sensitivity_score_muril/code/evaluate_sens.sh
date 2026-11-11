#!/bin/bash
#SBATCH --job-name=sens-eval
#SBATCH --output=sens_eval_out_%j.log
#SBATCH --error=sens_eval_err_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

cd $SLURM_SUBMIT_DIR

module load anaconda3-2024
source /apps/compilers/anaconda3-2024/etc/profile.d/conda.sh
conda activate qwen_env

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export RUN_NAME=${RUN_NAME:-${MODEL:-muril}}
export MODEL_PATH=${MODEL_PATH:-runs/run_$RUN_NAME/best}
if [ ! -f "runs/run_$RUN_NAME/train_summary.json" ] && [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "Training for run $RUN_NAME has not finished (runs/run_$RUN_NAME/train_summary.json missing)."
  exit 1
fi

python3 -u evaluate_sens.py
python3 -u summarize_sens.py
