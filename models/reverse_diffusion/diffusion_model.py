import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding + MLP projection."""
    
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        
        # MLP to project sinusoidal embedding
        self.mlp = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.SiLU(),
            nn.Linear(d * 4, d)
        ) # linear layer to project from d to 4d, then activation, then linear layer back to d, this expand and compress makes it learn richer transformations
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: tensor of shape [batch_size] containing timestep indices (1 to T)
        
        Returns:
            t_emb: tensor of shape [batch_size, d]
        """
        # Sinusoidal encoding
        device = t.device
        batch_size = t.shape[0]
        
        # Create sinusoidal positional encoding
        t_float = t.float()
        half_d = self.d // 2
        emb = math.log(10000) / (half_d - 1)
        emb = torch.exp(torch.arange(half_d, device=device) * -emb)
        emb = t_float[:, None] * emb[None, :]
        
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.d % 2 == 1:
            emb = torch.cat([emb, torch.zeros(batch_size, 1, device=device)], dim=-1)
        
        # Project through MLP
        t_emb = self.mlp(emb)  # [batch_size, d]
        return t_emb


class AdaLN(nn.Module):
    """Adaptive Layer Normalization modulated by timestep embedding."""
    
    def __init__(self, d: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.affine = nn.Linear(d_cond, 2 * d)  # outputs gamma and beta
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape [batch_size, L, d]
            t_emb: tensor of shape [batch_size, d_cond]
        
        Returns:
            output: tensor of shape [batch_size, L, d]
        """
        normalized = self.norm(x)
        affine_params = self.affine(t_emb)  # [batch_size, 2*d]
        
        # Split into gamma and beta
        gamma, beta = affine_params.chunk(2, dim=-1)  # each [batch_size, d] so d is split into two parts
        
        # Reshape for broadcasting: [batch_size, 1, d]
        if gamma.dim() == 2:
            gamma = gamma.unsqueeze(1)
        if beta.dim() == 2:
            beta = beta.unsqueeze(1)# adds 1 at position (index) of value specifed here it was 1, if 2. was given then it would add 1 at index 2 so the shape would be [batch_size, d, 1]
        # print("normalized", normalized.shape, "gamma", gamma.shape, "beta", beta.shape)
        return gamma * normalized + beta


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        assert d % n_heads == 0, "d must be divisible by n_heads"
        
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads
        
        self.Q = nn.Linear(d, d)
        self.K = nn.Linear(d, d)
        self.V = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [batch_size, seq_q, d]
            key: [batch_size, seq_k, d]
            value: [batch_size, seq_k, d]
            mask: optional [batch_size, seq_q, seq_k]
        
        Returns:
            output: [batch_size, seq_q, d]
        """
        batch_size = query.shape[0]
        
        # Project to multiple heads
        Q = self.Q(query)  # [batch_size, seq_q, d]
        K = self.K(key)    # [batch_size, seq_k, d]
        V = self.V(value)  # [batch_size, seq_k, d]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        # [batch_size, n_heads, seq_q, d_head]
        K = K.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        # [batch_size, n_heads, seq_k, d_head]
        V = V.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        # [batch_size, n_heads, seq_k, d_head]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        # [batch_size, n_heads, seq_q, seq_k]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # [batch_size, n_heads, seq_q, d_head]
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch_size, seq_q, n_heads, d_head]
        attn_output = attn_output.view(batch_size, -1, self.d)
        # [batch_size, seq_q, d]
        
        output = self.out(attn_output)
        return output



class TransformerBlock(nn.Module):
    """Single Transformer block with self-attention, cross-attention, and FFN."""
    
    def __init__(self, d: int, n_heads: int, d_ff: int, u_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        
        # d_cond = d (t_emb) + u_dim (raw u); if u_dim=0 reverts to t-only conditioning
        d_cond = d + u_dim
        self.adalan1 = AdaLN(d, d_cond)
        self.adalan2 = AdaLN(d, d_cond)
        
        # Self-attention
        self.self_attn = MultiHeadAttention(d, n_heads)
        
        # Cross-attention (x attends to u)
        # self.cross_attn = MultiHeadAttention(d, n_heads)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        t_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: main input (vt) [batch_size, L, d]
            u: semantic anchor [batch_size, L, d]
            t_emb: timestep embedding [batch_size, d]
        
        Returns:
            output: [batch_size, L, d]
        
        """
        # fuse timestep and u into a single conditioning vector [B, d + u_dim]
        c = torch.cat([t_emb, u], dim=-1)

        # a. AdaLN + b. Self-Attention
        x_normalized = self.adalan1(x, c)
        x = x + self.dropout(self.self_attn(x_normalized, x_normalized, x_normalized))
        
        # c. Cross-Attention (x attends to u, no AdaLN before it)
        # x = x + self.dropout(self.cross_attn(x, u, u))
        
        # d. AdaLN + e. FFN
        x_normalized = self.adalan2(x, c)
        x = x + self.dropout(self.ffn(x_normalized))
        
        return x


class DiffusionModel(nn.Module):
    def __init__(self, d: int, num_slots: int, u_dim: int, hidden_dim: int):
        super().__init__()
        self.d = d
        self.num_slots = num_slots
        self.u_dim = u_dim
        self.hidden_dim = hidden_dim
        self.N_blocks = 4

        self.timestep_emb = TimestepEmbedding(d)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d=self.d,
                n_heads=4,
                d_ff=self.hidden_dim,
                u_dim=self.u_dim,
                dropout=0.1
            )
            for _ in range(self.N_blocks)
        ])

        self.output_norm = nn.LayerNorm(self.d)
        self.output_projection = nn.Linear(self.d, self.d)


    def forward(self, vt: torch.Tensor, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        This takes in degraded latent, gist and timestep.
        It predicts the latednt at previous timestep.
        """

        B, L, D = vt.shape
        t = t.to(vt.device)
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=vt.device)

        t_emb = self.timestep_emb(t)                 # [B, H]
        for block in self.transformer_blocks:
            h = block(vt, u, t_emb)                      # [B, L, D]
        v_prev_hat = self.output_projection(self.output_norm(h))              # [B, L, D]
        return v_prev_hat
