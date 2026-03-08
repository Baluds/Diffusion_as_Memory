import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer

# Ensure project root is on sys.path when running this script from a subdirectory.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataloader.dataloader_augmentated import MSRAugmentedDataset
from models.decoder_prep.decoder_x import DecoderX
from models.decoder_prep.decoder_y import DecoderY
from models.encoder_prep.encoder import TextEncoder
from models.forgetting_model import ForgettingModel
from models.slot_pooling_prep.slot_pooling import SlotPooling
from models.uv_heads_prep.u_head import UHead
from models.uv_heads_prep.v_head import VHead


def build_model(device: torch.device) -> ForgettingModel:
    encoder = TextEncoder()
    slot_pool = SlotPooling(hidden_dim=encoder.hidden_dim_size, num_slots=8)
    u_head = UHead(hidden_dim=encoder.hidden_dim_size, output_dim=128)
    v_head = VHead(hidden_dim=encoder.hidden_dim_size)
    decoder_x = DecoderX()
    decoder_y = DecoderY(hidden_dim=encoder.hidden_dim_size, u_dim=128, num_slots=8)

    model = ForgettingModel(
        encoder=encoder,
        slot_pooling=slot_pool,
        u_head=u_head,
        v_head=v_head,
        decoder_x=decoder_x,
        decoder_y=decoder_y,
    )
    return model.to(device)


def load_checkpoint(model: ForgettingModel, checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {
            "epoch": checkpoint.get("epoch"),
            "train_loss": checkpoint.get("train_loss"),
            "val_loss": checkpoint.get("val_loss"),
        }
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        metadata = {"epoch": None, "train_loss": None, "val_loss": None}
    else:
        raise ValueError("Unsupported checkpoint format. Expected dict with model weights.")

    model.load_state_dict(state_dict)
    model.eval()
    return metadata


@torch.no_grad()
def run_inference(
    model: ForgettingModel,
    dataloader: DataLoader,
    tokenizer: T5Tokenizer,
    max_samples: int,
    log_every: int,
    use_wandb: bool,
) -> Tuple[float, List[Dict[str, str]], int]:
    total_loss = 0.0
    num_batches = 0
    total_examples = 0
    collected_samples: List[Dict[str, str]] = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference"), start=1):
        loss, logits_x, logits_y = model(batch)

        loss_value = loss.item()
        total_loss += loss_value
        num_batches += 1

        batch_size = batch["x_input_ids"].size(0)
        total_examples += batch_size

        if use_wandb and (batch_idx % log_every == 0):
            wandb.log({"inference/batch_loss": loss_value, "inference/batch_idx": batch_idx})

        if len(collected_samples) >= max_samples:
            continue

        pred_ids_x = torch.argmax(logits_x, dim=-1).detach().cpu()
        pred_ids_y = torch.argmax(logits_y, dim=-1).detach().cpu()

        decoded_x_pred = tokenizer.batch_decode(pred_ids_x, skip_special_tokens=True)
        decoded_y_pred = tokenizer.batch_decode(pred_ids_y, skip_special_tokens=True)

        decoded_x_true = tokenizer.batch_decode(batch["x_input_ids"], skip_special_tokens=True)
        decoded_y_true = tokenizer.batch_decode(batch["y_input_ids"], skip_special_tokens=True)

        remaining = max_samples - len(collected_samples)
        for x_true, x_pred, y_true, y_pred in zip(
            decoded_x_true[:remaining],
            decoded_x_pred[:remaining],
            decoded_y_true[:remaining],
            decoded_y_pred[:remaining],
        ):
            collected_samples.append(
                {
                    "x_true": x_true,
                    "x_pred": x_pred,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )

    if num_batches == 0:
        raise RuntimeError("Inference dataloader is empty.")

    avg_loss = total_loss / num_batches
    return avg_loss, collected_samples, total_examples


def log_samples_to_wandb(samples: List[Dict[str, str]], use_wandb: bool) -> None:
    if not use_wandb or not samples:
        return

    table = wandb.Table(columns=["x_true", "x_pred", "y_true", "y_pred"])
    for row in samples:
        table.add_data(row["x_true"], row["x_pred"], row["y_true"], row["y_pred"])

    wandb.log({"inference/samples": table})


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference script for ForgettingModel with W&B logging")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data-path", type=str, required=True, help="Path to inference dataset JSON")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max-samples", type=int, default=50, help="Max decoded samples to save/log")
    parser.add_argument("--log-every", type=int, default=10, help="Log batch loss every N batches")
    parser.add_argument(
        "--output-json",
        type=str,
        default="./output/p0/inference/forgetting_model_predictions.json",
        help="Where to save decoded predictions",
    )
    parser.add_argument("--wandb-project", type=str, default="diffusion-as-memory", help="W&B project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-off", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    if args.log_every <= 0:
        raise ValueError("--log-every must be a positive integer")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    dataset = MSRAugmentedDataset(args.data_path, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(device)
    checkpoint_meta = load_checkpoint(model, args.model_path, device)

    use_wandb = not args.wandb_off
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model_path": args.model_path,
                "data_path": args.data_path,
                "batch_size": args.batch_size,
                "max_samples": args.max_samples,
                "device": str(device),
                "checkpoint_epoch": checkpoint_meta.get("epoch"),
                "checkpoint_train_loss": checkpoint_meta.get("train_loss"),
                "checkpoint_val_loss": checkpoint_meta.get("val_loss"),
            },
        )

    avg_loss, samples, total_examples = run_inference(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        log_every=args.log_every,
        use_wandb=use_wandb,
    )

    output_parent = os.path.dirname(args.output_json)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Inference complete. Avg loss: {avg_loss:.4f}")
    print(f"Examples processed: {total_examples}")
    print(f"Saved decoded predictions to: {args.output_json}")

    if use_wandb:
        wandb.log(
            {
                "inference/avg_loss": avg_loss,
                "inference/num_examples": total_examples,
                "inference/num_samples_logged": len(samples),
            }
        )
        log_samples_to_wandb(samples, use_wandb=True)
        wandb.run.summary["avg_loss"] = avg_loss
        wandb.run.summary["num_examples"] = total_examples
        wandb.finish()


if __name__ == "__main__":
    main()
