"""
Training script for the Denoiser model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Dict, Optional
import json
from pathlib import Path
from tqdm import tqdm

from config import DenoiserConfig
from denoiser import Denoiser, NoiseSchedule, forward_diffusion


class LatentDataset(Dataset):
    """
    Load pre-computed latents (v0, u) from saved PyTorch tensors.
    """
    
    def __init__(self, latent_path: str, L: int, d: int):
        """
        Args:
            latent_path: path to latent .pt file
            L: expected number of slots
            d: expected embedding dimension
        """
        self.latent_path = latent_path
        self.L = L
        self.d = d
        
        # Load latents
        latents_dict = torch.load(latent_path, map_location='cpu')
        v0_loaded = latents_dict['v0']  # [num_samples, L_saved, d]
        self.u_raw = latents_dict['u']  # [num_samples, d_u]
        
        num_samples = v0_loaded.shape[0]
        L_saved = v0_loaded.shape[1]
        d_saved = v0_loaded.shape[2]
        d_u = self.u_raw.shape[1]
        
        # Validate dimensions
        assert d_saved == d, \
            f"Embedding dimension mismatch: config d={d}, data d={d_saved}"
        assert L_saved == L, \
            f"Slot mismatch: config L={L}, data L={L_saved}. Update L in config.py to match your data."
        
        self.v0 = v0_loaded
        
        # Project u to [num_samples, L, d] using linear projection
        # u_raw is [num_samples, d_u], we want [num_samples, L, d]
        self.proj = nn.Linear(d_u, L * d)
        
        # Project and reshape: [num_samples, L*d] -> [num_samples, L, d]
        with torch.no_grad():
            u_projected = self.proj(self.u_raw)  # [num_samples, L*d]
            self.u = u_projected.view(num_samples, L, d)
        
        # Validate shapes
        assert self.v0.shape == (num_samples, L, d), \
            f"v0 shape mismatch: expected [{num_samples}, {L}, {d}], got {self.v0.shape}"
        assert self.u.shape == (num_samples, L, d), \
            f"u shape mismatch: expected [{num_samples}, {L}, {d}], got {self.u.shape}"
        
        print(f"Loaded {num_samples} samples from {latent_path}")
        print(f"  v0: {self.v0.shape}, u (projected): {self.u.shape}")
    
    def __len__(self) -> int:
        return self.v0.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (v0, u) pair for a single sample."""
        return self.v0[idx], self.u[idx]


class DummyLatentDataset(Dataset):
    """
    Dummy dataset for testing. In practice, use LatentDataset instead.
    """
    
    def __init__(self, num_samples: int, L: int, d: int):
        """
        Args:
            num_samples: number of sample pairs
            L: number of slots
            d: embedding dimension
        """
        self.num_samples = num_samples
        self.L = L
        self.d = d
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (v0, u) pair."""
        # Random clean latent and semantic anchor
        v0 = torch.randn(self.L, self.d)
        u = torch.randn(self.L, self.d)
        return v0, u


class DenoiserTrainer:
    """Trainer for the Denoiser model."""
    
    def __init__(
        self,
        config: DenoiserConfig,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Args:
            config: DenoiserConfig object
            checkpoint_dir: directory to save checkpoints
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models
        self.denoiser = Denoiser(config).to(self.device)
        self.noise_schedule = NoiseSchedule(config.T, config.schedule)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.denoiser.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # adjust based on num_epochs
            eta_min=1e-6
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Checkpoint tracking
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.best_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: training DataLoader
        
        Returns:
            average loss over the epoch
        """
        self.denoiser.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for v0_batch, u_batch in pbar:
            # Move to device
            v0_batch = v0_batch.to(self.device)  # [batch_size, L, d]
            u_batch = u_batch.to(self.device)    # [batch_size, L, d]
            batch_size = v0_batch.shape[0]
            
            # Sample random timesteps
            t = torch.randint(1, self.config.T + 1, (batch_size,), device=self.device)
            
            # Forward diffusion
            vt, eps = forward_diffusion(v0_batch, t, self.noise_schedule)
            
            # Predict noise with Denoiser
            eps_hat = self.denoiser(vt, t, u_batch)
            
            # Compute loss
            loss = self.criterion(eps_hat, eps)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: validation DataLoader
        
        Returns:
            average validation loss
        """
        self.denoiser.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            for v0_batch, u_batch in pbar:
                # Move to device
                v0_batch = v0_batch.to(self.device)
                u_batch = u_batch.to(self.device)
                batch_size = v0_batch.shape[0]
                
                # Sample random timesteps
                t = torch.randint(1, self.config.T + 1, (batch_size,), device=self.device)
                
                # Forward diffusion
                vt, eps = forward_diffusion(v0_batch, t, self.noise_schedule)
                
                # Predict noise
                eps_hat = self.denoiser(vt, t, u_batch)
                
                # Compute loss
                loss = self.criterion(eps_hat, eps)
                
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'val_loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.denoiser.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': {
                'L': self.config.L,
                'd': self.config.d,
                'T': self.config.T,
                'N_blocks': self.config.N_blocks,
                'n_heads': self.config.n_heads,
                'd_ff': self.config.d_ff,
                'schedule': self.config.schedule,
            }
        }
        
        save_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint: {save_path}")
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.denoiser.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int
    ):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: training DataLoader
            val_loader: validation DataLoader
            num_epochs: number of training epochs
        """
        for epoch in range(1, num_epochs + 1):
            print(f"\n--- Epoch {epoch}/{num_epochs} ---")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)
            print(f"Training Loss: {train_loss:.6f}")
            
            # Validate
            val_loss = self.validate(val_loader)
            self.training_history['val_loss'].append(val_loss)
            print(f"Validation Loss: {val_loss:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Update learning rate
            self.scheduler.step()
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        # Save training history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"\nSaved training history to {history_path}")


def main():
    """Example training script."""
    config = DenoiserConfig()
    
    # Create dummy datasets
    train_dataset = DummyLatentDataset(num_samples=100, L=config.L, d=config.d)
    val_dataset = DummyLatentDataset(num_samples=20, L=config.L, d=config.d)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create trainer
    trainer = DenoiserTrainer(
        config,
        checkpoint_dir="./checkpoints"
    )
    
    # Train
    trainer.train(train_loader, val_loader, num_epochs=config.num_epochs)


if __name__ == "__main__":
    main()
