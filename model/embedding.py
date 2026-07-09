"""
model/embedding.py

ICUEventEmbedding: turns one raw ICU event into a d_model-dimensional vector.

Each event has four components — two discrete, two continuous:

    itemid      (discrete)   what was measured    → Embedding lookup  [V, d]
    source      (discrete)   where it came from   → Embedding lookup  [4, d]
    delta_hours (continuous) when it happened      → Linear(1 → d)
    value       (continuous) the Z-scored reading  → Linear(1 → d)

The four projected vectors are summed element-wise, then LayerNorm is applied.

NOTE: There is no sinusoidal positional encoding.
delta_hours already carries real, irregular time — using it directly via
time_proj is better than sinusoidal PE which assumes uniform spacing.
"""

import torch
import torch.nn as nn
from model.config import ModelConfig


class ICUEventEmbedding(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Lookup tables for discrete fields
        self.itemid_emb = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,   # [PAD] embeddings stay zero
        )
        self.source_emb = nn.Embedding(
            num_embeddings=config.source_size,  # 4 sources
            embedding_dim=config.d_model,
        )

        # Linear projections for scalar continuous fields
        # Each maps a single float → d_model-dimensional vector
        self.time_proj  = nn.Linear(1, config.d_model)   # delta_hours
        self.value_proj = nn.Linear(1, config.d_model)   # Z-scored value

        self.norm    = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        itemid      : torch.Tensor,   # [B, L]  LongTensor  — vocab token index
        source      : torch.Tensor,   # [B, L]  LongTensor  — 0/1/2/3
        delta_hours : torch.Tensor,   # [B, L]  FloatTensor — hours from ICU admission
        value       : torch.Tensor,   # [B, L]  FloatTensor — Z-scored measurement
    ) -> torch.Tensor:                # [B, L, d_model]

        # Expand scalar fields from [B, L] to [B, L, 1] for the Linear layers
        time_vec  = self.time_proj(delta_hours.unsqueeze(-1))   # [B, L, d_model]
        value_vec = self.value_proj(value.unsqueeze(-1))        # [B, L, d_model]

        # Sum all four contributions — each already lives in d_model space
        token = (
            self.itemid_emb(itemid)   # [B, L, d_model]
            + self.source_emb(source) # [B, L, d_model]
            + time_vec                # [B, L, d_model]
            + value_vec               # [B, L, d_model]
        )

        # LayerNorm stabilises the combined signal; dropout for regularisation
        return self.dropout(self.norm(token))   # [B, L, d_model]
