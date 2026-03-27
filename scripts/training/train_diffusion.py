import argparse
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader

from dataloader.dataloader_diffusion import MSRDiffusionDataset
from utils.training_utils import build_p0_model, set_trainable_params, convert_tokens_to_text_and_log, save_decoder_gpsi_checkpoint, save_model_checkpoint
from models.reverse_diffusion.diffusion_model import DiffusionModel

# CONSTANTS
BATCH_SIZE = 10
NUM_EPOCHS = 500
LEARNING_RATE = 5e-5
L_SLOTS = 8
D_MODEL = 512
U_DIM = 128
T_DIFFUSION = 1000
XT_BUCKET_SIZE = T_DIFFUSION // 10
VAL_INTERVAL = 10


def train_epoch(p0_model, diffusion_model, train_loader, optimizer, device):
    """
    1. Get encoded u and v0, vt, using the encoder
    2. 
    
    :param p0_model: Description
    :param diffusion_model: Description
    :param train_loader: Description
    :param optimizer: Description
    :param device: Description
    """
    p0_model.g_psi.train()
    p0_model.decoder_x.train()
    diffusion_model.train()

    mse_criterion = nn.MSELoss()

    total_loss = 0.0
    for batch_idx, batch in enumerate(train_loader, description="Training", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        batch_size = batch["x0_input_ids"].shape[0]

        # encoder latents
        with torch.no_grad():
            u, v0 = p0_model.encode_xt_latents(batch["x0_input_ids"], batch["x0_attention"])
            _, vt = p0_model.encode_xt_latents(batch["xt_input_ids"], batch["xt_attention"])
            _, vprev = p0_model.encode_xt_latents(batch["xprev_input_ids"], batch["xprev_attention"])

        # Forward pass through diffusion model
        vprev_hat = diffusion_model(vt, batch["t"], u)
         
        latent_loss = mse_criterion(vprev_hat, vprev)

        # run decoder and gpsi, compute their loss
        vprev_hat = p0_model.g_psi(vprev_hat, batch["t"], vt, u)
        slot_mask = torch.ones(batch_size, L_SLOTS, device=device)
        decoder_loss, _ = p0_model.decoder_x(vprev_hat, slot_mask, batch["xprev_text"])

        loss = 0.5 * latent_loss + 0.5 * decoder_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate_epoch(p0_model, diffusion_model, val_loader, device):
    p0_model.g_psi.eval()
    p0_model.decoder_x.eval()
    diffusion_model.eval()

    mse_criterion = nn.MSELoss()
    total_loss = 0.0
    sample_outputs = []

    for batch_idx, batch in enumerate(val_loader, description="Validation", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch_size = batch["x0_input_ids"].shape[0]

        with torch.no_grad():
            u, v0 = p0_model.encode_xt_latents(batch["x0_input_ids"], batch["x0_attention"])
            _, vt = p0_model.encode_xt_latents(batch["xt_input_ids"], batch["xt_attention"])
            _, vprev = p0_model.encode_xt_latents(batch["xprev_input_ids"], batch["xprev_attention"])

            vprev_hat = diffusion_model(vt, batch["t"], u)
            latent_loss = mse_criterion(vprev_hat, vprev)

            vprev_hat = p0_model.g_psi(vprev_hat, batch["t"], vt, u)
            slot_mask = torch.ones(batch_size, L_SLOTS, device=device)
            decoder_loss, xprev_hat_logits = p0_model.decoder_x(vprev_hat, slot_mask, batch["xprev_text"])

            loss = 0.5 * latent_loss + 0.5 * decoder_loss
            total_loss += loss.item()

            sample_outputs.append(batch, xprev_hat_logits, batch["t"], batch["xprev_text"])

    avg_loss = total_loss / len(val_loader)
    return avg_loss, sample_outputs


def train(p0_model, diffusion_model, train_loader, val_loader, optimizer, device, output_dir, checkpoint_dir):
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(p0_model, diffusion_model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}")

        if (epoch + 1) % VAL_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
            val_loss, sample_outputs = validate_epoch(p0_model, diffusion_model, val_loader, device)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation Loss: {val_loss:.4f}")

            if sample_outputs:
                convert_tokens_to_text_and_log(sample_outputs, p0_model.tokenizer, epoch, output_dir)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_decoder_gpsi_checkpoint(
                    p0_model.g_psi,
                    p0_model.decoder_x,
                    optimizer,
                    epoch + 1,
                    train_loss,
                    val_loss,
                    os.path.join(checkpoint_dir, "best_decoder_gpsi_model.pt"),
                )
                save_model_checkpoint(
                    diffusion_model,
                    optimizer,
                    epoch + 1,
                    train_loss,
                    val_loss,
                    os.path.join(checkpoint_dir, "best_diffusion_model.pt"),
                )
                print(
                    f"  New best model saved (val_loss={val_loss:.4f})", flush=True
                )


def main():

    print("Starting training script...")
    parser = argparse.ArgumentParser(
        description="Train denoiser + G_psi + fine-tune Decoder"
    )
    parser.add_argument(
        "--p0-checkpoint", type=str, required=True,
        help="Path to Phase 0 best_model.pt checkpoint",
    )
    parser.add_argument(
        "--train-dataset", type=str, required=True,
        help="Path to the training dataset"
    )
    parser.add_argument(
        "--val-dataset", type=str, required=True,
        help="Path to the validation dataset"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="diffusion-as-memory",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-run-name", type=str, required=False,
        help="W&B run name",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = "/project/pi_dagarwal_umass_edu/project_3/issinha/checkpoints/reverse_diffusion"
    output_dir = "/project/pi_dagarwal_umass_edu/project_3/issinha/output/reverse_diffusion"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_dataset = MSRDiffusionDataset(args.train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle = True)
    val_dataset = MSRDiffusionDataset(args.val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle = True)

    p0_model = build_p0_model(device, L_SLOTS, U_DIM)
    checkpoint = torch.load(args.p0_checkpoint, map_location=device)
    p0_model.load_state_dict(checkpoint["model_state_dict"])
    p0_trainable_params = set_trainable_params(p0_model)
    print(f"  Loaded from {args.p0_checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    diffusion_model = DiffusionModel(D_MODEL, L_SLOTS, U_DIM, D_MODEL).to(device)

    train(
        p0_model,
        diffusion_model,
        train_loader,
        val_loader,
        torch.optim.Adam(list(p0_trainable_params) + list(diffusion_model.parameters()), lr=LEARNING_RATE),
        device,
        output_dir,
        checkpoint_dir,
    )
    



if __name__ == "__main__":
    main()
