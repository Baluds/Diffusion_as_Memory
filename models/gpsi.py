import torch
import torch.nn as nn

class GPsi(nn.Module):
    """
    Gψ: semantic projection module.
    Takes u (global semantic code) and conditions v (slot latent: v0 or vt)
    using AdaLayerNorm style conditioning.

    u: [B, 128]
    v: [B, L, 512]   (each slot is 512-dim)
    out: [B, L, 512]
    """
    def __init__(self, u_dim: int, d_model: int, hidden_mult: int = 4, dropout: float = 0.0, eps: float = 1e-5):
        super().__init__()
        hidden = hidden_mult * d_model

        # LayerNorm is used to normalize each slot vector over its last dimension, which is 512
        # elementwise_affine is set to false because we provide own affine params (gamma and beta) from u
        self.ln = nn.LayerNorm(d_model, eps=eps, elementwise_affine=False)

        # turn u into [gamma, beta] params
        self.mlp = nn.Sequential(
            nn.Linear(u_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2 * d_model)   # outputs gamma and beta
        )

        # settings gamma and beta close to 0 makes condiitoning close to identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.d_model = d_model

    def forward(self, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        v: [B, L, 512]  (slot latents, e.g. v0 or vt)
        u: [B, 128]       (semantic code)
        """
        B, L, D = v.shape
        assert D == self.d_model, f"Expected last dim {self.d_model}, got {D}"

        # Normalise each slot vector
        v_norm = self.ln(v)              # [B, L, 512]

        # Compute conditioning parameters from u
        gamma_beta = self.mlp(u)         # [B, 2*512]

        # Split into gamma and beta
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B,512], [B,512]

        # Broadcast gamma and beta across the slot dimension L
        gamma = gamma.view(B, 1, D)      # where D is 512
        beta  = beta.view(B, 1, D)

        # AdaLN-style conditioning. (1+gamma) trick keeps identity-like scaling at init
        return v_norm * (1.0 + gamma) + beta
