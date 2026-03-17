#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=p0-train
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=80GB

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/.conda/envs/diffusion
python3 /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/Diffusion_as_Memory/scripts/training/training_dl_augmented.py \
 --latents-dir ./data/latents/mod_g_psi \
 --checkpoint-dir ./checkpoints/p0/mod_g_psi \
 --output-dir ./output/p0/mod_g_psi \
 --wandb-project diffusion-as-memory \
 --wandb-run-name p0-training-run-mod_g_psi_$(date +%Y%m%d_%H%M%S)
