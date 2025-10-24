"""
改良されたトランスフォーマーモデル

PWA+PET Transformer with SU2Gate and RoPE
Peter-Weyl Attention (PWA) + Phase-Enriched Transformer (PET)
シンプル実装（条件分岐最小化）
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE)"""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        half = dim // 2
        inv_freq = base ** (-torch.arange(0, half, dtype=torch.float32) / half)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        half = dim // 2
        freqs = torch.einsum("l,f->lf", torch.arange(length, device=x.device, dtype=torch.float32), self.inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        cos = cos.unsqueeze(0).repeat(batch, 1, 1)
        sin = sin.unsqueeze(0).repeat(batch, 1, 1)
        x_even = x[..., :half]
        x_odd = x[..., half:]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        return torch.cat([rot_even, rot_odd], dim=-1)


class SU2Gate(nn.Module):
    """SU(2) Gate for molecular representations"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        half = dim // 2
        self.alpha = nn.Parameter(torch.zeros(half))
        self.beta = nn.Parameter(torch.zeros(half))
        self.gamma = nn.Parameter(torch.zeros(half))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        half = dim // 2
        x_even = x[..., :half]
        x_odd = x[..., half:]
        cos_a = torch.cos(self.alpha)
        sin_a = torch.sin(self.alpha)
        cos_b = torch.cos(self.beta)
        sin_b = torch.sin(self.beta)
        cos_g = torch.cos(self.gamma)
        sin_g = torch.sin(self.gamma)
        r11 = cos_a * cos_b
        r12 = -sin_a * cos_g + cos_a * sin_b * sin_g
        r21 = sin_a * cos_b
        r22 = cos_a * cos_g + sin_a * sin_b * sin_g
        rot_even = r11 * x_even + r12 * x_odd
        rot_odd = r21 * x_even + r22 * x_odd
        return torch.cat([rot_even, rot_odd], dim=-1)


class PeterWeylAttention(nn.Module):
    """Peter-Weyl Attention (multi-head self-attention)"""
    
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = math.sqrt(dim // heads)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        batch, length, dim = x.shape
        q = self.q_proj(x).view(batch, length, self.heads, dim // self.heads).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.heads, dim // self.heads).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.heads, dim // self.heads).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.out_proj(attended)


class PhaseEnrichedTransformerLayer(nn.Module):
    """Single PET layer combining PWA and SU2 gate"""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = PeterWeylAttention(dim, heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.su2 = SU2Gate(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        gated = self.su2(x)
        ffn_out = self.ffn(gated)
        return self.norm2(gated + self.dropout(ffn_out))


class TransformerRegressor(nn.Module):
    """改良されたPWA+PETトランスフォーマーモデル"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        num_targets: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = RotaryPositionalEncoding(hidden_dim)
        self.layers = nn.ModuleList(
            [PhaseEnrichedTransformerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, num_targets)
        self._init_weights()
    
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        mask = None  # Fix: Define mask as None if it is undefined
        for layer in self.layers:
            x = layer(x, mask)
        x = x.mean(dim=1)
        return self.output_proj(x)
    
    def get_attention_weights(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        states = self.input_proj(x)
        states = self.pos_encoding(states)
        weights = []
        for layer in self.layers:
            attn_out = layer.attention(states, mask)
            weights.append(attn_out)
            states = layer(states, mask)
        return weights

    def get_embeddings(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        states = self.input_proj(x)
        states = self.pos_encoding(states)
        for layer in self.layers:
            states = layer(states, mask)
        return states
    
    def get_feature_importance(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x.requires_grad_(True)
        preds = self.forward(x, mask)
        grad = torch.autograd.grad(preds.sum(), x, retain_graph=True)[0]
        return grad.abs().mean(dim=0)


def create_transformer_model(
    input_dim: int,
    hidden_dim: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    num_targets: int = 1,
    dropout: float = 0.1,
) -> TransformerRegressor:
    return TransformerRegressor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_targets=num_targets,
        dropout=dropout,
    )


if __name__ == "__main__":
    model = create_transformer_model(input_dim=128, hidden_dim=256, num_heads=8, num_layers=4, num_targets=3)
    x = torch.randn(16, 50, 128)
    outputs = model(x)
    print("outputs:", outputs.shape)
    embeddings = model.get_embeddings(x)
    print("embeddings:", embeddings.shape)
    try:
        importance = model.get_feature_importance(x)
    except Exception as e:
        print(f"Error getting feature importance: {e}")
        importance = None
    print("importance:", importance.shape)
