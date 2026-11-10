#!/bin/bash
#SBATCH --job-name=a3-grammar-eval
#SBATCH --output=a3_eval_out_%j.log
#SBATCH --error=a3_eval_err_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

cd $SLURM_SUBMIT_DIR

module load anaconda3-2024
source /apps/compilers/anaconda3-2024/etc/profile.d/conda.sh
conda activate qwen_env

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python3 -c "import sacrebleu" 2>/dev/null || python3 -m pip install -q sacrebleu
if [ "${USE_SIMILARITY:-0}" = "1" ]; then
  python3 -c "import sentence_transformers" 2>/dev/null || python3 -m pip install -q sentence-transformers
fi

RUN=${RUN:-D}
A2_ADAPTER=${A2_ADAPTER:-qwen_4b_hinglish_lora}
LONG_DATA=${LONG_DATA:-data_a3_full}
TEST_SET=${TEST_SET:-original}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}

case "$RUN" in
  BASE)
    export ADAPTER=""
    DEFAULT_PERSONA=0
    ;;
  A)
    export ADAPTER="$A2_ADAPTER"
    DEFAULT_PERSONA=0
    ;;
  B|C|D|E|F|G1|G2|G2_long)
    export ADAPTER="runs/run_$RUN/best"
    DEFAULT_PERSONA=1
    ;;
  8B_long|8B_all)
    export ADAPTER="runs/run_$RUN/best"
    export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-8B}
    DEFAULT_PERSONA=1
    ;;
  *)
    echo "Unknown RUN=$RUN. Use BASE, A, B, C, D, E, F, G1, G2, G2_long, 8B_long or 8B_all."
    exit 1
    ;;
esac

if [ -n "$ADAPTER" ] && [ ! -f "$ADAPTER/adapter_config.json" ]; then
  echo "Adapter not found at $ADAPTER"
  exit 1
fi
case "$RUN" in
  B|C|D|E|F|G1|G2|G2_long|8B_long|8B_all)
    if [ ! -f "runs/run_$RUN/train_summary.json" ]; then
      echo "Training for run $RUN has not finished (runs/run_$RUN/train_summary.json missing). Evaluate after training ends."
      exit 1
    fi
    ;;
esac
if [ ! -f data_a3/test_indictalk.csv ]; then
  echo "data_a3/ not found. Run prepare_data_a3.sh first."
  exit 1
fi
if [ "$TEST_SET" = "long" ]; then
  if [ ! -f "$LONG_DATA/test_indictalk.csv" ]; then
    echo "$LONG_DATA/test_indictalk.csv not found. submit_final_long.sh builds it."
    exit 1
  fi
  export TEST_FILES="$LONG_DATA/test_indictalk.csv"
  export RESULTS_DIR="results_long"
fi

export USE_PERSONA=${USE_PERSONA:-$DEFAULT_PERSONA}
if [ "$USE_PERSONA" != "$DEFAULT_PERSONA" ]; then
  export RUN_NAME="${RUN}_persona${USE_PERSONA}"
else
  export RUN_NAME="$RUN"
fi

python3 -u evaluate_grammar_a3.py
python3 -u summarize_results.py
