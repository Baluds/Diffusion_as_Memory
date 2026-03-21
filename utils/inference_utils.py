"""
Helper functions for inference of trained models
"""

import torch
from utils.training_utils import build_p0_model
from models.denoiser_module.denoiser import Denoiser, DenoiserConfig


def load_p0_model_from_checkpoint(path, device, l_slots, u_dim):
    """
    Reconstruct the P0 ForgettingModel architecture (needed to load state dict).
    
    :param: device: "cuda" or "cpu"
    :param: l_slots: number of slots
    :param: u_dim: dimension of u vector
    """
    model = build_p0_model(device, l_slots, u_dim)
    checkpoint = torch.load(path, map_location=device)
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
    
    return model, metadata
    

def load_denoiser_from_checkpoint(checkpoint_path, device):
    """Load a trained denoiser from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct config from saved values if available
    config = DenoiserConfig()
    if "config" in checkpoint:
        saved_cfg = checkpoint["config"]
        config.L = saved_cfg.get("L", config.L)
        config.d = saved_cfg.get("d", config.d)
        config.T = saved_cfg.get("T", config.T)
        config.N_blocks = saved_cfg.get("N_blocks", config.N_blocks)
        config.n_heads = saved_cfg.get("n_heads", config.n_heads)
        config.d_ff = saved_cfg.get("d_ff", config.d_ff)
        config.schedule = saved_cfg.get("schedule", config.schedule)

    denoiser = Denoiser(config).to(device)
    denoiser.load_state_dict(checkpoint["model_state_dict"])
    denoiser.eval()

    metadata = {
        "epoch": checkpoint.get("epoch"),
    }

    print(f"Loaded denoiser from {checkpoint_path} (epoch {metadata['epoch']})")
    print(f"  Config: L={config.L}, d={config.d}, T={config.T}, blocks={config.N_blocks}")
    return denoiser, config, metadata

