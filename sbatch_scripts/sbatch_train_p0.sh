#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a16|gh200|h100|a100|l40s|a40|rtx8000"
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --job-name=p0-train
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=80GB
#SBATCH --output=sbatch_scripts/slurm_output/p0/p0-train-%j.out
#SBATCH --error=sbatch_scripts/slurm_output/p0/p0-train-%j.err

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/.conda/envs/diffusion
python3 /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/Diffusion_as_Memory/scripts/training/training_dl_augmented.py \
 --latents-dir ./data/latents/rev-diff \
 --checkpoint-dir ./checkpoints/p0/rev-diff \
 --output-dir ./output/p0/rev-diff \
 --wandb-project diffusion-as-memory \
 --wandb-run-name p0-training-run_$(date +%Y%m%d_%H%M%S)
