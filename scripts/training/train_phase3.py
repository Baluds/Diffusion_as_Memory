"""
Phase 3 (P2) training: Train G_psi (Semantic Projection Module) + fine-tune Decoder.

Freezes: E (encoder), P (slot pooling), U (u-head), V (v-head), N (denoiser)
Trains:  G_psi (new semantic projection module), D (decoder_x, fine-tuned)
Label selection from xt (progressive degradations):
  index = min(t // bucket_size, len(xt) - 1)
  e.g. T=1000, bucket=100: t=50→xt[0], t=150→xt[1], t=950→xt[9] or last

Usage:
    python train_phase3.py \\
        --p0-checkpoint ./checkpoints/p0/2loss/best_model.pt \\
        --wandb-run-name p2-gpsi-run
"""

import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
import json
import os
import argparse
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.training_utils import ETATracker
from tqdm import tqdm
from dataloader.dataloader_augmentated import MSRAugmentedDataset
from models.encoder_prep.encoder import TextEncoder
from models.slot_pooling_prep.slot_pooling import SlotPooling
from models.uv_heads_prep.u_head import UHead
from models.uv_heads_prep.v_head import VHead
from models.decoder_prep.decoder_x import DecoderX
from models.forgetting_model import ForgettingModel
from denoiser_module.semantic_projection import SemanticProjectionModule
from denoiser_module.config import DenoiserConfig
from denoiser_module.denoiser import Denoiser, NoiseSchedule, forward_diffusion, one_step_estimate
from denoiser_module.g_psi_config import G_psi_config
from train_phase3_config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    LAMBDA_CLEAN,
    VAL_INTERVAL,
    GPSI_N_BLOCKS,
    GPSI_N_HEADS,
    GPSI_D_FF,
    GPSI_DROPOUT,
    T_DIFFUSION,
    NOISE_SCHEDULE,
    XT_BUCKET_SIZE,
    L_SLOTS,
    D_MODEL,
    U_DIM,
)


# Helpers

def build_p0_model(device):
    """Reconstruct the P0 ForgettingModel architecture (needed to load state dict)."""
    encoder = TextEncoder()
    slot_pool = SlotPooling(hidden_dim=encoder.hidden_dim_size, num_slots=L_SLOTS)
    u_head = UHead(hidden_dim=encoder.hidden_dim_size, output_dim=U_DIM)
    v_head = VHead(hidden_dim=encoder.hidden_dim_size)
    decoder_x = DecoderX()

    model = ForgettingModel(
        encoder=encoder,
        slot_pooling=slot_pool,
        u_head=u_head,
        v_head=v_head,
        decoder_x=decoder_x,
    )
    model.to(device)
    return model


def load_denoiser(denoiser_checkpoint, device):
    """Load a trained denoiser checkpoint and freeze it for Phase 3."""
    checkpoint = torch.load(denoiser_checkpoint, map_location=device)

    config = DenoiserConfig()
    if "config" in checkpoint:
        saved_cfg = checkpoint["config"]
        config.L = saved_cfg.get("L", config.L)
        config.d = saved_cfg.get("d", config.d)
        config.u_dim = saved_cfg.get("u_dim", config.u_dim)
        config.T = saved_cfg.get("T", config.T)
        config.N_blocks = saved_cfg.get("N_blocks", config.N_blocks)
        config.n_heads = saved_cfg.get("n_heads", config.n_heads)
        config.d_ff = saved_cfg.get("d_ff", config.d_ff)
        config.schedule = saved_cfg.get("schedule", config.schedule)

    denoiser = Denoiser(config).to(device)
    denoiser.load_state_dict(checkpoint["model_state_dict"])

    denoiser.eval()
    for param in denoiser.parameters():
        param.requires_grad = False

    print(f"  Loaded denoiser from {denoiser_checkpoint} (epoch {checkpoint.get('epoch', '?')})")
    return denoiser


def select_xt_labels(batch, t, device):
    """Pick the right degraded xt label for each sample based on timestep.

    Bucket mapping: index = min(t // XT_BUCKET_SIZE, xt_count - 1)
    Variable-length handling: xt lists can be 1–10 items long.
    If a sample has only 3 items and t maps to index 5, it clamps
    to the last available item (index 2).
    """
    xt_input_ids = batch["xt_input_ids"].to(device)   # [B, max_xt_items, seq_len]
    xt_count = batch["xt_count"].to(device)            # [B]
    B = t.shape[0]

    raw_index = t // XT_BUCKET_SIZE                           # [B]
    xt_index = torch.min(raw_index, xt_count - 1)             # clamp per sample
    labels = xt_input_ids[torch.arange(B, device=device), xt_index]  # [B, seq_len]
    return labels, xt_index


def train_epoch(p0_model, denoiser, g_psi, noise_schedule, dataloader, optimizer, device):
    """Run one training epoch. Returns (total_loss, loss_recon, loss_clean) averages."""
    g_psi.train()
    p0_model.decoder_x.train()

    total_loss = 0
    total_loss_recon = 0
    total_loss_clean = 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        batch_size = batch["x_input_ids"].shape[0]

        # Frozen encode — encode_latents handles .to(device) internally
        with torch.no_grad():
            u, v0 = p0_model.encode_latents(batch)
        u = u.detach()    # [B, 128]
        v0 = v0.detach()  # [B, 8, 512]


        # Noisy reconstruction loss (labels = degraded xt)
        t = torch.randint(1, T_DIFFUSION + 1, (batch_size,), device=device)
        vt, _ = forward_diffusion(v0, t, noise_schedule)
        with torch.no_grad():
            eps_hat = denoiser(vt, t, u)
            v_hat_0 = one_step_estimate(vt, eps_hat, t, noise_schedule)

        labels_noisy, _ = select_xt_labels(batch, t, device)   # [B, seq_len]
        vt_tilde = g_psi(v_hat_0=v_hat_0, t=t, v_t=vt, u=u)    # [B, L, d]
        slot_mask = torch.ones(batch_size, L_SLOTS, device=device)
        loss_recon, logits = p0_model.decoder_x(vt_tilde, slot_mask, labels_noisy)

        # Clean reconstruction loss (t=0, labels = original x)
        labels_clean = batch["x_input_ids"].to(device)          # [B, seq_len]
        t_zero = torch.zeros(batch_size, dtype=torch.long, device=device)
        v0_tilde = g_psi(v_hat_0=v0, t=t_zero, v_t=v0, u=u)
        loss_clean, _ = p0_model.decoder_x(v0_tilde, slot_mask, labels_clean)

        # Total loss
        loss = loss_recon + LAMBDA_CLEAN * loss_clean
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(g_psi.parameters()) + list(p0_model.decoder_x.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        total_loss += loss.item()
        total_loss_recon += loss_recon.item()
        total_loss_clean += loss_clean.item()

    n = len(dataloader)
    return total_loss / n, total_loss_recon / n, total_loss_clean / n


@torch.no_grad()
def validate_epoch(p0_model, denoiser, g_psi, noise_schedule, dataloader, device):
    """Run one validation epoch. Returns (total_loss, loss_recon, loss_clean, sample_outputs)."""
    g_psi.eval()
    p0_model.decoder_x.eval()

    total_loss = 0
    total_loss_recon = 0
    total_loss_clean = 0
    sample_outputs = []

    for batch in tqdm(dataloader, desc="Validating"):
        batch_size = batch["x_input_ids"].shape[0]

        u, v0 = p0_model.encode_latents(batch)


        # Noisy (labels = degraded xt)
        t = torch.randint(1, T_DIFFUSION + 1, (batch_size,), device=device)
        vt, _ = forward_diffusion(v0, t, noise_schedule)
        eps_hat = denoiser(vt, t, u)
        v_hat_0 = one_step_estimate(vt, eps_hat, t, noise_schedule)
        labels_noisy, xt_index = select_xt_labels(batch, t, device)
        # Noisy reconstruction
        vt_tilde = g_psi(v_hat_0=v_hat_0, t=t, v_t=vt, u=u)
        slot_mask = torch.ones(batch_size, L_SLOTS, device=device)
        loss_recon, logits_noisy = p0_model.decoder_x(vt_tilde, slot_mask, labels_noisy)

        # Clean (labels = original x)
        labels_clean = batch["x_input_ids"].to(device)
        t_zero = torch.zeros(batch_size, dtype=torch.long, device=device)
        v0_tilde = g_psi(v_hat_0=v0, t=t_zero, v_t=v0, u=u)
        loss_clean, logits_clean = p0_model.decoder_x(v0_tilde, slot_mask, labels_clean)

        loss = loss_recon + LAMBDA_CLEAN * loss_clean
        total_loss += loss.item()
        total_loss_recon += loss_recon.item()
        total_loss_clean += loss_clean.item()

        sample_outputs.append((batch, logits_noisy, logits_clean, t, xt_index))

    n = len(dataloader)
    return total_loss / n, total_loss_recon / n, total_loss_clean / n, sample_outputs


def log_sample_outputs(sample_outputs, tokenizer, epoch, output_dir):
    """Decode and save predictions for all validation batches."""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for batch, logits_noisy, logits_clean, t_vals, xt_idx in sample_outputs:
        pred_noisy = tokenizer.batch_decode(
            torch.argmax(logits_noisy, dim=-1), skip_special_tokens=True
        )
        pred_clean = tokenizer.batch_decode(
            torch.argmax(logits_clean, dim=-1), skip_special_tokens=True
        )
        original = tokenizer.batch_decode(
            batch["x_input_ids"], skip_special_tokens=True
        )
        # Decode the xt target that was used as the noisy label
        xt_all = batch["xt_input_ids"]  # [batch_size, max_xt_items, seq_len]
        batch_size = xt_all.shape[0]
        xt_target_ids = xt_all[torch.arange(batch_size), xt_idx.cpu()]
        xt_target = tokenizer.batch_decode(xt_target_ids, skip_special_tokens=True)

        for i in range(len(original)):
            results.append(
                {
                    "original": original[i],
                    "xt_target": xt_target[i],
                    "xt_index": xt_idx[i].item(),
                    "recon_noisy": pred_noisy[i],
                    "recon_clean": pred_clean[i],
                    "t": t_vals[i].item(),
                }
            )

    out_path = os.path.join(output_dir, f"epoch_{epoch + 1}_samples.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)


def save_checkpoint(g_psi, decoder, optimizer, epoch, train_loss, val_loss, path):
    """Save Phase 3 checkpoint (G_psi + fine-tuned decoder)."""
    torch.save(
        {
            "epoch": epoch,
            "g_psi_state_dict": g_psi.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )



def main():
    print(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Phase 3 (P2): Train G_psi + fine-tune Decoder"
    )
    parser.add_argument(
        "--p0-checkpoint", type=str, required=True,
        help="Path to Phase 0 best_model.pt checkpoint",
    )
    parser.add_argument(
        "--denoiser-checkpoint", type=str, required=True,
        help="Path to Phase 1 denoiser best_model.pt checkpoint",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output/p2/temp",
        help="Directory to write sample output JSONs",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints/p2/temp",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(ROOT, "data", "final"),
        help="Directory containing train.json and validate.json",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="diffusion-as-memory",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-run-name", type=str, required=True,
        help="W&B run name",
    )
    parser.add_argument(
        "--wandb-off", action="store_true",
        help="Disable W&B logging entirely",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    # Load P0 model
    print("\nLoading P0 checkpoint...")
    p0_model = build_p0_model(device)
    checkpoint = torch.load(args.p0_checkpoint, map_location=device)
    p0_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded from {args.p0_checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    print("\nLoading P1 denoiser checkpoint...")
    denoiser = load_denoiser(args.denoiser_checkpoint, device)

    # Freeze everything in P0 model
    for param in p0_model.parameters():
        param.requires_grad = False

    # Unfreeze decoder for fine-tuning
    for param in p0_model.decoder_x.parameters():
        param.requires_grad = True

    # Create G_psi
    g_psi = SemanticProjectionModule(
        config=G_psi_config
    ).to(device)

    # can be deleted after confirming correct loading and training
    trainable_params = sum(p.numel() for p in g_psi.parameters()) + sum(
        p.numel() for p in p0_model.decoder_x.parameters()
    )
    frozen_params = sum(
        p.numel() for p in p0_model.parameters() if not p.requires_grad
    )
    print(f"Trainable params: {trainable_params:,}  Frozen params: {frozen_params:,}")

    # Noise schedule (alphas)
    noise_schedule = NoiseSchedule(T=T_DIFFUSION, schedule_type=NOISE_SCHEDULE)

    # Data 
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    train_dataset = MSRAugmentedDataset(
        os.path.join(args.data_dir, "train.json"), tokenizer
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = MSRAugmentedDataset(
        os.path.join(args.data_dir, "validate.json"), tokenizer
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Optimizer (only G_psi + decoder)
    optimizer = torch.optim.Adam(
        list(g_psi.parameters()) + list(p0_model.decoder_x.parameters()),
        lr=LEARNING_RATE,
    )

    output_dir = args.output_dir
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    use_wandb = not args.wandb_off
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "phase": "P2",
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "lambda_clean": LAMBDA_CLEAN,
                "gpsi_n_blocks": GPSI_N_BLOCKS,
                "gpsi_n_heads": GPSI_N_HEADS,
                "gpsi_d_ff": GPSI_D_FF,
                "T_diffusion": T_DIFFUSION,
                "noise_schedule": NOISE_SCHEDULE,
                "trainable_params": trainable_params,
            },
        )

    print(f"\n{'-'*60}")
    print("STARTING PHASE 3 (P2) TRAINING")
    print(f"  Epochs={EPOCHS}  Batch={BATCH_SIZE}  LR={LEARNING_RATE}")
    print(f"  Lambda_clean={LAMBDA_CLEAN}  G_psi blocks={GPSI_N_BLOCKS}")
    print(f"  T={T_DIFFUSION}  Schedule={NOISE_SCHEDULE}")
    print(f"{'-'*60}\n")

    best_val_loss = float("inf")
    eta_tracker = ETATracker(total_epochs=EPOCHS)

    for epoch in range(EPOCHS):
        eta_tracker.start_epoch()

        train_loss, train_recon, train_clean = train_epoch(
            p0_model, denoiser, g_psi, noise_schedule, train_loader, optimizer, device,
        )

        epoch_elapsed, eta_seconds, eta_str = eta_tracker.end_epoch()

        # Log train metrics
        if use_wandb:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/loss_recon": train_recon,
                    "train/loss_clean": train_clean,
                    **eta_tracker.wandb_metrics(epoch_elapsed, eta_seconds),
                },
                step=epoch + 1,
            )

        # Validate every VAL_INTERVAL epochs and on the last epoch
        if (epoch + 1) % VAL_INTERVAL == 0 or (epoch + 1) == EPOCHS:
            val_loss, val_recon, val_clean, sample_outputs = validate_epoch(
                p0_model, denoiser, g_psi, noise_schedule, val_loader, device,
            )

            print(
                f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} "
                f"(recon={val_recon:.4f} clean={val_clean:.4f}) | ETA: {eta_str}",
                flush=True,
            )
            print("-" * 30, flush=True)

            if use_wandb:
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "val/loss_recon": val_recon,
                        "val/loss_clean": val_clean,
                    },
                    step=epoch + 1,
                )

            if sample_outputs:
                log_sample_outputs(sample_outputs, tokenizer, epoch, output_dir)

                # Log a sample table to W&B (first batch only)
                if use_wandb:
                    batch0, logits_n, logits_c, t0, xi0 = sample_outputs[0]
                    pred_n = tokenizer.batch_decode(
                        torch.argmax(logits_n, dim=-1), skip_special_tokens=True
                    )
                    pred_c = tokenizer.batch_decode(
                        torch.argmax(logits_c, dim=-1), skip_special_tokens=True
                    )
                    orig = tokenizer.batch_decode(
                        batch0["x_input_ids"], skip_special_tokens=True
                    )
                    B0 = batch0["xt_input_ids"].shape[0]
                    xt_tgt = tokenizer.batch_decode(
                        batch0["xt_input_ids"][torch.arange(B0), xi0.cpu()],
                        skip_special_tokens=True,
                    )
                    table = wandb.Table(
                        columns=["original", "xt_target", "recon_noisy", "recon_clean", "t", "xt_idx"]
                    )
                    for o, xt, pn, pc, ti, xi in zip(orig, xt_tgt, pred_n, pred_c, t0, xi0):
                        table.add_data(o, xt, pn, pc, ti.item(), xi.item())
                    wandb.log({"val/samples": table}, step=epoch + 1)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    g_psi,
                    p0_model.decoder_x,
                    optimizer,
                    epoch + 1,
                    train_loss,
                    val_loss,
                    os.path.join(checkpoint_dir, "best_model.pt"),
                )
                print(
                    f"  New best model saved (val_loss={val_loss:.4f})", flush=True
                )
                if use_wandb:
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    wandb.run.summary["best_epoch"] = epoch + 1
        else:
            print(
                f"Epoch {epoch+1} | Train: {train_loss:.4f} | ETA: {eta_str}",
                flush=True,
            )

    # Save final checkpoint
    save_checkpoint(
        g_psi,
        p0_model.decoder_x,
        optimizer,
        EPOCHS,
        train_loss,
        best_val_loss,
        os.path.join(checkpoint_dir, "final_model.pt"),
    )
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
