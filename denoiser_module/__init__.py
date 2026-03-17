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
from .trainer import DenoiserTrainer
from .inference import DenoiserInference


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
    "DenoiserInference",

]
