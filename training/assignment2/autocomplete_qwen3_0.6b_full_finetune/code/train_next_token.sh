#!/bin/bash
#SBATCH --job-name=hinglish-next
#SBATCH --output=hinglish_next_%j.out
#SBATCH --error=hinglish_next_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=2-00:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module load anaconda3-2024
source /apps/compilers/anaconda3-2024/etc/profile.d/conda.sh
conda activate qwen_env

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

export MODEL_NAME="Qwen/Qwen3-0.6B-Base"
export DATASET_NAME="theguywithblacktie/hinglish-conversations"
export DATASET_CONFIG="large"

export MAX_LENGTH=1024

export BATCH_SIZE=8
export GRAD_ACCUM_STEPS=4

export EPOCHS=1
export LEARNING_RATE=1e-5
export WEIGHT_DECAY=0.01
export WARMUP_RATIO=0.03

export SAVE_STEPS=2000
export HEARTBEAT_SECONDS=60

export NUM_WORKERS=4

echo "============================================================"
echo "HINGLISH QWEN3-0.6B NEXT-TOKEN TRAINING"
echo "============================================================"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"

echo ""
echo "GPU:"
nvidia-smi

echo ""
echo "Python:"
python3 --version

echo ""
echo "PyTorch:"
python3 -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('BF16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"

echo ""
echo "Transformers:"
python3 -c "import transformers; print(transformers.__version__)"

echo ""
echo "Datasets:"
python3 -c "import datasets; print(datasets.__version__)"

echo ""
echo "============================================================"

python3 -u train_next_token.py

echo ""
echo "Job finished: $(date)"
