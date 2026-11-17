#!/bin/bash
#SBATCH --job-name=a3-sweep-eval
#SBATCH --output=sweep_eval_out_%j.log
#SBATCH --error=sweep_eval_err_%j.log
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

python3 -c "import sacrebleu" 2>/dev/null || python3 -m pip install -q sacrebleu

NAME=${SWEEP_NAME:?Set SWEEP_NAME}
EVAL_SET=${EVAL_SET:-val}

if [ ! -f "runs/$NAME/train_summary.json" ]; then
  echo "Training for $NAME has not finished (runs/$NAME/train_summary.json missing)."
  exit 1
fi

export RUN_NAME="$NAME"
export ADAPTER="runs/$NAME/best"
export USE_PERSONA=1

if [ "$EVAL_SET" = "test" ]; then
  export RESULTS_DIR="results_test"
  export TEST_FILES="../data_a3/test_indictalk.csv,../data_a3/test_a2.csv,../data_a3/test_a2_original.csv"
  python3 -u ../evaluate_grammar_a3.py
  python3 -u ../summarize_results.py
else
  python3 -u split_val.py || exit 1
  export RESULTS_DIR="results"
  export TEST_FILES="data/val_indictalk.csv,data/val_a2.csv"
  python3 -u ../evaluate_grammar_a3.py
  python3 -u ../summarize_results.py
  python3 -u select_best.py
fi
