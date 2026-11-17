#!/bin/bash
#SBATCH --job-name=sens-prepare
#SBATCH --output=sens_prepare_out_%j.log
#SBATCH --error=sens_prepare_err_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

cd $SLURM_SUBMIT_DIR

module load anaconda3-2024
source /apps/compilers/anaconda3-2024/etc/profile.d/conda.sh
conda activate qwen_env

export PYTHONUNBUFFERED=1

python3 -u prepare_data_sens.py
