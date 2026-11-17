#!/bin/bash
#SBATCH --job-name=qwen4b-grammar
#SBATCH --output=qwen4b_out_%j.log
#SBATCH --error=qwen4b_err_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=2-00:00:00

cd $SLURM_SUBMIT_DIR

module load anaconda3-2024
source /apps/compilers/anaconda3-2024/etc/profile.d/conda.sh
conda activate qwen_env

python3 -u train_4b.py
