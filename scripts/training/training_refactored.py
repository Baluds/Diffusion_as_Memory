import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import T5Tokenizer
import json
import os
import numpy as np

from tqdm import tqdm
from dataloader.dataloader_static import MSRDataset
from models.encoder_prep.encoder import TextEncoder
from models.slot_pooling_prep.slot_pooling import SlotPooling
from models.uv_heads_prep.u_head import UHead
from models.uv_heads_prep.v_head import VHead
from models.decoder_prep.decoder_x import DecoderX
from models.decoder_prep.decoder_y import DecoderY
from models.forgetting_model import ForgettingModel


def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        loss, _, _ = model(batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(dataloader)


@torch.no_grad()
def validate_epoch(model, dataloader):
    model.eval()
    total_loss = 0
    sample_outputs = None
    
    #Added tqdm progress bar for validation loop
    pbar = tqdm(dataloader, desc="Validating", leave=False)
    for i, batch in enumerate(pbar):
        loss, logits_x, logits_y = model(batch)
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        if i == 0:
            sample_outputs = (batch, logits_x, logits_y)
            
    return total_loss / len(dataloader), sample_outputs


def log_sample_outputs(sample_outputs, tokenizer, epoch):
    batch, logits_x, logits_y = sample_outputs

    pred_ids_x = torch.argmax(logits_x, dim=-1)
    pred_ids_y = torch.argmax(logits_y, dim=-1)
    decoded_x = tokenizer.batch_decode(pred_ids_x, skip_special_tokens=True)
    decoded_y = tokenizer.batch_decode(pred_ids_y, skip_special_tokens=True)

    original_x = tokenizer.batch_decode(batch["x_input_ids"], skip_special_tokens=True)
    original_y = tokenizer.batch_decode(batch["y_input_ids"], skip_special_tokens=True)

    results = []
    for i in range(len(decoded_x)):
        results.append({
            "original_x": original_x[i],
            "v0": decoded_x[i],
            "original_y": original_y[i],
            "u": decoded_y[i]
        })

    with open(f"./output_big/epoch_{epoch+1}_samples.json", "w") as f:
        json.dump(results, f, indent=4)

# Utility function to save model checkpoints
def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path):
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, path)


@torch.no_grad()
def extract_and_save_latents(model, dataloader, tokenizer, save_path):
    """
    Run frozen model on data to extract u and v0 latents by running data through trained model.
    Saves a .pt file with all latents for use in denoiser.
    """
    model.eval()

    all_u = []
    all_v0 = []
    all_texts = []

    pbar = tqdm(dataloader, desc="Extracting latents", leave=False)
    for batch in pbar:
        u, v0 = model.encode_latents(batch)
        all_u.append(u.cpu())
        all_v0.append(v0.cpu())

        texts = tokenizer.batch_decode(batch["x_input_ids"], skip_special_tokens=True)
        all_texts.extend(texts)

    # Concatenate all batches
    all_u = torch.cat(all_u, dim=0)    # [N, 128]
    all_v0 = torch.cat(all_v0, dim=0)  # [N, 8, 512]

    torch.save({
        "u": all_u,
        "v0": all_v0,
        "texts": all_texts,
    }, save_path)

    print(f"Saved latents to {save_path}")
    print(f"  u shape:  {all_u.shape}")
    print(f"  v0 shape: {all_v0.shape}")
    print(f"  samples:  {len(all_texts)}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    train_dataset = MSRDataset("./data/final/train.json")
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle = True)
    val_dataset = MSRDataset("./data/final/validate.json")
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle = True)
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    encoder = TextEncoder()
    slot_pool = SlotPooling(hidden_dim = encoder.hidden_dim_size, num_slots = 8)
    u_head = UHead(hidden_dim = encoder.hidden_dim_size, output_dim = 128)
    v_head = VHead(hidden_dim = encoder.hidden_dim_size)
    decoder_x = DecoderX()
    decoder_y = DecoderY(hidden_dim = encoder.hidden_dim_size, u_dim = 128, num_slots = 8)

    model = ForgettingModel(
        encoder = encoder,
        slot_pooling = slot_pool,
        u_head = u_head,
        v_head = v_head,
        decoder_x = decoder_x,
        decoder_y = decoder_y,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    epochs = 500

    # Validate every n epochs
    val_interval = 10

    os.makedirs("./output_big", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    best_val_loss = float("inf")

    for epoch in tqdm(range(epochs), desc="Epochs"):
        train_loss = train_epoch(model, train_loader, optimizer)

        # Validate every val_interval epochs and on the last epoch
        if (epoch + 1) % val_interval == 0 or (epoch + 1) == epochs:
            val_loss, sample_outputs = validate_epoch(model, val_loader)

            tqdm.write(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            tqdm.write("-" * 30)

            if sample_outputs:
                log_sample_outputs(sample_outputs, tokenizer, epoch)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch+1, train_loss, val_loss,
                                "./checkpoints/best_model.pt")
                tqdm.write(f"  -> New best model saved (val_loss={val_loss:.4f})")
        else:
            tqdm.write(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}")

    # Save final checkpoint
    save_checkpoint(model, optimizer, epochs, train_loss, best_val_loss,
                    "./checkpoints/final_model.pt")
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    print("\nExtracting latents from frozen model")

    # Load best checkpoint
    checkpoint = torch.load("./checkpoints/best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    os.makedirs("./data/latents", exist_ok=True)

    # Extract latents from training data
    train_loader_noshuffle = DataLoader(train_dataset, batch_size=10, shuffle=False)
    extract_and_save_latents(model, train_loader_noshuffle, tokenizer,
                             "./data/latents/train_latents.pt")

    # Extract latents from validation data
    val_loader_noshuffle = DataLoader(val_dataset, batch_size=10, shuffle=False)
    extract_and_save_latents(model, val_loader_noshuffle, tokenizer,
                             "./data/latents/val_latents.pt")

if __name__ == "__main__":
    main()
