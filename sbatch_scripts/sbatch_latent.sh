#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=data-prep
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=80GB

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/.conda/envs/diffusion
python3 /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/Diffusion_as_Memory/testing_refactored.py
