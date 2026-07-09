"""
model/encoder.py

Transformer encoder stack with Pre-LayerNorm (Pre-LN).

Pre-LN applies LayerNorm BEFORE each sublayer (not after, as in the
original "Attention Is All You Need" paper). This is more numerically
stable during training, especially in the early epochs when gradients
can be large.

Per-layer computation:
    x = x + Attention( LayerNorm(x), padding_mask )
    x = x + FFN( LayerNorm(x) )

The final output passes through one more LayerNorm for clean representations.
"""

import torch
import torch.nn as nn
from model.config import ModelConfig


class TransformerEncoderLayer(nn.Module):
    """A single Pre-LN Transformer encoder layer (attention + FFN)."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Multi-head self-attention — batch_first=True means input is [B, L, d]
        self.attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Position-wise feed-forward: expand to d_ff, then project back
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),                        # smoother than ReLU for transformers
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

        # Two LayerNorms: one before attention, one before FFN
        self.norm_attn = nn.LayerNorm(config.d_model)
        self.norm_ffn  = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x            : torch.Tensor,        # [B, L, d_model]
        padding_mask : torch.Tensor | None, # [B, L]  True=real event, False=pad
    ) -> torch.Tensor:                      # [B, L, d_model]

        # nn.MultiheadAttention uses True = IGNORE in key_padding_mask,
        # which is the opposite of our convention (True = real event).
        attn_key_mask = ~padding_mask if padding_mask is not None else None

        # Pre-LN attention sublayer with residual connection
        normed    = self.norm_attn(x)
        attn_out, _ = self.attention(
            query=normed,
            key=normed,
            value=normed,
            key_padding_mask=attn_key_mask,
            need_weights=False,             # skip attention weight storage
        )
        x = x + attn_out

        # Pre-LN feed-forward sublayer with residual connection
        x = x + self.ffn(self.norm_ffn(x))

        return x


class TransformerEncoder(nn.Module):
    """Stack of N identical encoder layers followed by a final LayerNorm."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config)
            for _ in range(config.n_layers)
        ])

        # Final norm ensures all output representations are on the same scale
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x            : torch.Tensor,        # [B, L, d_model]
        padding_mask : torch.Tensor | None, # [B, L]
    ) -> torch.Tensor:                      # [B, L, d_model]

        for layer in self.layers:
            x = layer(x, padding_mask)

        return self.final_norm(x)
