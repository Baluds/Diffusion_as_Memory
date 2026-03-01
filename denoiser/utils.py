"""
Utility functions for the Denoiser module.
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import json


def create_batch_tensors(
    v0_list: List[np.ndarray],
    u_list: List[np.ndarray]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create batched tensors from lists of numpy arrays.
    
    Args:
        v0_list: list of clean latents
        u_list: list of semantic anchors
    
    Returns:
        (v0_batch, u_batch) as torch tensors
    """
    v0_batch = torch.stack([torch.from_numpy(v).float() for v in v0_list])
    u_batch = torch.stack([torch.from_numpy(u).float() for u in u_list])
    return v0_batch, u_batch


def sample_timesteps(
    batch_size: int,
    T: int,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Sample random timesteps for a batch.
    
    Args:
        batch_size: size of batch
        T: total number of timesteps
        device: device to create tensor on
    
    Returns:
        timestep tensor of shape [batch_size]
    """
    return torch.randint(1, T + 1, (batch_size,), device=device)


def compute_snr(
    t: torch.Tensor,
    alpha_bar: torch.Tensor
) -> torch.Tensor:
    """
    Compute Signal-to-Noise Ratio at timestep t.
    
    SNR(t) = alpha_bar(t) / (1 - alpha_bar(t))
    
    Args:
        t: timestep indices [batch_size]
        alpha_bar: noise schedule values
    
    Returns:
        snr: [batch_size]
    """
    ab = alpha_bar[t - 1]  # [batch_size]
    snr = ab / (1 - ab)
    return snr


def exponential_moving_average(
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    decay: float = 0.99
):
    """
    Update EMA model with decay.
    
    Args:
        model: current model
        ema_model: EMA model to update
        decay: decay rate (typically 0.99 or 0.999)
    """
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data


def save_config(config_obj, save_path: str):
    """Save configuration to JSON file."""
    config_dict = vars(config_obj)
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config to {save_path}")


def load_config_from_json(json_path: str):
    """Load configuration from JSON file."""
    with open(json_path, 'r') as f:
        config_dict = json.load(f)
    return config_dict


def get_model_size(model: torch.nn.Module) -> int:
    """
    Calculate total number of parameters in model.
    
    Args:
        model: PyTorch model
    
    Returns:
        total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def get_trainable_params(model: torch.nn.Module) -> int:
    """
    Calculate number of trainable parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module, config=None):
    """Print model architecture summary."""
    total_params = get_model_size(model)
    trainable_params = get_trainable_params(model)
    
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    if config:
        print(f"Config: L={config.L}, d={config.d}, T={config.T}")
        print(f"        N_blocks={config.N_blocks}, n_heads={config.n_heads}, d_ff={config.d_ff}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*60 + "\n")
    
    return total_params, trainable_params


def get_device() -> torch.device:
    """Get appropriate device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device
