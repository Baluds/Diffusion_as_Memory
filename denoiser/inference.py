"""
Inference script for the Denoiser model.
At inference time, given a stored (vt, u, t), recover v0_hat.
"""

import torch
from pathlib import Path
from typing import Tuple

from config import DenoiserConfig
from denoiser import Denoiser, NoiseSchedule, one_step_estimate


class DenoiserInference:
    """Inference wrapper for the Denoiser model."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config: DenoiserConfig = None
    ):
        """
        Args:
            checkpoint_path: path to saved checkpoint
            config: DenoiserConfig (if None, loaded from checkpoint)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Use config from checkpoint if not provided
        if config is None:
            config = DenoiserConfig()
            # Update with checkpoint config
            checkpoint_config = checkpoint.get('config', {})
            for key, value in checkpoint_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        self.config = config
        
        # Initialize model
        self.denoiser = Denoiser(config).to(self.device)
        self.denoiser.load_state_dict(checkpoint['model_state_dict'])
        self.denoiser.eval()
        
        # Initialize noise schedule
        self.noise_schedule = NoiseSchedule(config.T, config.schedule)
        
        print(f"Loaded denoiser from {checkpoint_path}")
        print(f"Model on device: {self.device}")
    
    def recover(
        self,
        vt: torch.Tensor,
        u: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Single-step recovery of clean latent.
        
        Args:
            vt: noisy latent [batch_size, L, d] or [L, d]
            u: semantic anchor [batch_size, L, d] or [L, d]
            t: timestep indices [batch_size] or scalar
        
        Returns:
            v0_hat: estimated clean latent [batch_size, L, d] or [L, d]
        """
        # Add batch dimension if needed
        vt_input = vt if vt.dim() == 3 else vt.unsqueeze(0)
        u_input = u if u.dim() == 3 else u.unsqueeze(0)
        t_input = t if t.dim() == 1 else torch.tensor([t], dtype=torch.long)
        
        # Move to device
        vt_input = vt_input.to(self.device)
        u_input = u_input.to(self.device)
        t_input = t_input.to(self.device)
        
        # Predict noise
        with torch.no_grad():
            eps_hat = self.denoiser(vt_input, t_input, u_input)
        
        # Recover clean latent
        v0_hat = one_step_estimate(vt_input, eps_hat, t_input, self.noise_schedule)
        
        # Remove batch dimension if input didn't have it
        if vt.dim() == 2:
            v0_hat = v0_hat.squeeze(0)
        
        return v0_hat
    
    def batch_recover(
        self,
        vt_list: list,
        u_list: list,
        t_list: list
    ) -> list:
        """
        Recover multiple samples.
        
        Args:
            vt_list: list of noisy latents
            u_list: list of semantic anchors
            t_list: list of timestep indices
        
        Returns:
            list of estimated clean latents
        """
        v0_hat_list = []
        for vt, u, t in zip(vt_list, u_list, t_list):
            v0_hat = self.recover(vt, u, t)
            v0_hat_list.append(v0_hat.cpu())
        return v0_hat_list


def demo_inference():
    """
    Demo: load a checkpoint and perform inference.
    """
    config = DenoiserConfig()
    
    # Try to load best model, otherwise use epoch 1
    checkpoint_dir = Path("./checkpoints")
    best_checkpoint = checkpoint_dir / "best_model.pt"
    
    if not best_checkpoint.exists():
        print("No checkpoint found. Please train the model first.")
        return
    
    # Create inference wrapper
    inference = DenoiserInference(str(best_checkpoint), config)
    
    # Create dummy test data
    batch_size = 2
    L = config.L
    d = config.d
    T = config.T
    
    # Dummy noisy latent and semantic anchor
    vt = torch.randn(batch_size, L, d)
    u = torch.randn(batch_size, L, d)
    t = torch.tensor([100, 500], dtype=torch.long)  # timesteps
    
    print(f"\nInput shapes:")
    print(f"  vt: {vt.shape}")
    print(f"  u: {u.shape}")
    print(f"  t: {t.shape}")
    
    # Recover
    v0_hat = inference.recover(vt, u, t)
    
    print(f"\nOutput shape:")
    print(f"  v0_hat: {v0_hat.shape}")
    print(f"\nRecovery complete!")
    
    return v0_hat


if __name__ == "__main__":
    demo_inference()
