#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pi_dagarwal_umass_edu
#SBATCH --gpus=l4:1
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=p2-train
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=80GB

module load conda/latest
conda activate /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/.conda/envs/diffusion

echo "PHASE 3 (P2): G_psi Semantic Projection + Decoder Fine-tuning"
echo ""

python /work/pi_dagarwal_umass_edu/project_3/bdevarangadi/Diffusion_as_Memory/scripts/training/train_phase3.py \
    --p0-checkpoint ./checkpoints/p0/mod_g_psi/best_model.pt \
    --denoiser-checkpoint ./checkpoints/p1/mod_g_psi/best_model.pt \
    --checkpoint-dir ./checkpoints/p2/mod_g_psi_no_cln \
    --output-dir ./output/p2/mod_g_psi_no_cln \
    --data-dir ./data/final \
    --wandb-project diffusion-as-memory \
    --wandb-run-name p2-gpsi-run_mod_g_psi_no_cln_$(date +%Y%m%d_%H%M%S)

echo "Phase 3 training complete!"
