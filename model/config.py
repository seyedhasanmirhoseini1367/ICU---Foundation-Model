"""
model/config.py

Single source of truth for all model hyperparameters.
Pass a ModelConfig instance to every component — changing one
value here propagates through the entire model automatically.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:

    # ── Vocabulary ──────────────────────────────────────────────────────────
    # vocab_size is set after build_vocab.py runs (number of unique itemids + 3 special tokens)
    vocab_size    : int   = 5000
    pad_token_id  : int   = 0     # [PAD]  — ignored by attention and loss
    mask_token_id : int   = 1     # [MASK] — replaces events during MEM pretraining
    cls_token_id  : int   = 2     # [CLS]  — prepended; its output is the patient summary
    source_size   : int   = 4     # number of event sources: CHART / INPUT / OUTPUT / LAB

    # ── Transformer dimensions ──────────────────────────────────────────────
    d_model  : int = 256    # dimension of every token embedding
    n_heads  : int = 8      # attention heads (each head = d_model // n_heads = 32 dims)
    d_ff     : int = 1024   # inner dimension of the feed-forward sublayer
    n_layers : int = 6      # number of stacked encoder layers

    # ── Sequence ────────────────────────────────────────────────────────────
    max_len  : int = 512    # max events per ICU stay (including [CLS] at pos 0)

    # ── Regularisation ──────────────────────────────────────────────────────
    dropout  : float = 0.1

    # ── Pretraining (Masked Event Modeling) ─────────────────────────────────
    mask_prob         : float = 0.15  # fraction of real events masked each step
    value_loss_weight : float = 0.5   # λ: total = λ·CE(itemid) + (1-λ)·MSE(value)

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}). "
            f"Each head would have dimension {self.d_model / self.n_heads:.1f}."
        )
