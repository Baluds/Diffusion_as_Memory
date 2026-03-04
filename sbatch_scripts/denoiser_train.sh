#!/bin/bash
#SBATCH --partition=superpod-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=denoiser-train
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=80GB

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/.conda/envs/diffusion

cd /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/Diffusion_as_Memory/denoiser

echo "DENOISER TRAINING"

# Create checkpoint directory
mkdir -p checkpoints

# Run training with default config
echo ""
echo "Starting training..."
echo "Train latents: ../data/latents/train_latents.pt (2198 samples)"
echo "Val latents: ../data/latents/val_latents.pt"
echo ""

python train_on_latents.py \
    --train-latents ../data/latents/train_latents.pt \
    --val-latents ../data/latents/val_latents.pt \
    --wandb-project diffusion-as-memory \
    --wandb-run-name p1-training-run_$(date +%Y%m%d_%H%M%S)

echo "Training complete!"
