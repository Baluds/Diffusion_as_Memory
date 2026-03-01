"""
Denoiser module: Diffusion-based denoising network for latent memory.

This module implements:
- Forward diffusion: corrupts clean latents with noise
- Denoiser Nθ: Transformer-based noise prediction
- One-step estimation: recovers clean latents
- Training and inference pipelines
"""

from .config import DenoiserConfig
from .denoiser import (
    NoiseSchedule,
    TimestepEmbedding,
    AdaLN,
    MultiHeadAttention,
    TransformerBlock,
    Denoiser,
    forward_diffusion,
    one_step_estimate
)
from .trainer import DenoiserTrainer, DummyLatentDataset
from .inference import DenoiserInference
from .utils import (
    create_batch_tensors,
    sample_timesteps,
    compute_snr,
    exponential_moving_average,
    save_config,
    load_config_from_json,
    get_model_size,
    get_trainable_params,
    print_model_summary,
    get_device
)

__all__ = [
    # Config
    "DenoiserConfig",
    
    # Core components
    "NoiseSchedule",
    "TimestepEmbedding",
    "AdaLN",
    "MultiHeadAttention",
    "TransformerBlock",
    "Denoiser",
    "forward_diffusion",
    "one_step_estimate",
    
    # Training and inference
    "DenoiserTrainer",
    "DummyLatentDataset",
    "DenoiserInference",
    
    # Utilities
    "create_batch_tensors",
    "sample_timesteps",
    "compute_snr",
    "exponential_moving_average",
    "save_config",
    "load_config_from_json",
    "get_model_size",
    "get_trainable_params",
    "print_model_summary",
    "get_device",
]
