#!/bin/bash
#SBATCH --job-name=a3-prepare-a2full
#SBATCH --output=a3_prepare_a2full_out_%j.log
#SBATCH --error=a3_prepare_a2full_err_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

cd $SLURM_SUBMIT_DIR

module load anaconda3-2024
source /apps/compilers/anaconda3-2024/etc/profile.d/conda.sh
conda activate qwen_env

export PYTHONUNBUFFERED=1

python3 -u prepare_a2_full_a3.py
