"""
model/autoregressive.py

ICUAutoregressiveModel — Branch B: left-to-right autoregressive ICU decoder.

Reuses ICUEventEmbedding and TransformerEncoder from Branch A unchanged.
The only difference is a causal attention mask passed to the encoder so each
position can only attend to itself and earlier positions (left context).

Causal mask: torch.triu(ones(L, L, bool), diagonal=1)
    True at (i, j) means position i CANNOT attend to position j.
    diagonal=1 → i CAN attend to itself (j=i is False = allowed);
                  i CANNOT attend to j > i.

The NextEventHead then predicts the three components of the next event
(itemid, value_bin, delta_bin) from every position's hidden state.

Training alignment (done in pretrain_ar.py, not here):
    predictions:  model(inputs)[:, :-1]   from positions 0 … L-2
    itemid  tgt:  itemid[:, 1:]           next token's id
    value   tgt:  value_bins[:, 1:]       next token's value bin
    delta   tgt:  delta_bins[:, :-1]      gap FROM pos t TO t+1
"""

import torch
import torch.nn as nn
from model.config    import ModelConfig
from model.embedding import ICUEventEmbedding
from model.encoder   import TransformerEncoder
from model.heads     import NextEventHead


class ICUAutoregressiveModel(nn.Module):
    """
    Branch B causal autoregressive ICU model.

    forward() returns (itemid_logits, value_logits, delta_logits), each [B, L, *].
    The caller is responsible for slicing to form the causal shift alignment.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config    = config
        self.embedding = ICUEventEmbedding(config)
        self.encoder   = TransformerEncoder(config)
        self.head      = NextEventHead(config)

    @staticmethod
    def _make_causal_mask(L: int, device: torch.device) -> torch.Tensor:
        """[L, L] BoolTensor — True means 'do not attend to this position'.
        Upper-triangular (diagonal=1) → each pos sees itself + all left positions."""
        return torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1
        )

    def forward(
        self,
        itemid       : torch.Tensor,   # [B, L]  LongTensor
        source       : torch.Tensor,   # [B, L]  LongTensor
        delta_hours  : torch.Tensor,   # [B, L]  FloatTensor
        value        : torch.Tensor,   # [B, L]  FloatTensor
        padding_mask : torch.Tensor,   # [B, L]  BoolTensor  True=real, False=pad
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            itemid_logits : [B, L, vocab_size]
            value_logits  : [B, L, n_value_bins]
            delta_logits  : [B, L, n_time_bins]
        """
        L           = itemid.size(1)
        causal_mask = self._make_causal_mask(L, itemid.device)

        x = self.embedding(itemid, source, delta_hours, value)    # [B, L, d_model]
        x = self.encoder(x, padding_mask, attn_mask=causal_mask)  # [B, L, d_model]

        return self.head(x)   # (item_logits, val_logits, delta_logits)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = ModelConfig(vocab_size=200, d_model=64, n_heads=4, d_ff=128, n_layers=2,
                      max_len=32, n_value_bins=10, n_time_bins=10)
    model = ICUAutoregressiveModel(cfg)
    print(f"ICUAutoregressiveModel: {model.count_parameters():,} parameters")

    B, L = 2, 32
    itemid       = torch.randint(4, cfg.vocab_size, (B, L))
    source       = torch.randint(0, 4,              (B, L))
    delta_hours  = torch.cumsum(torch.rand(B, L) * 2, dim=1)
    value        = torch.randn(B, L)
    padding_mask = torch.ones(B, L, dtype=torch.bool)
    padding_mask[:, -4:] = False   # last 4 positions are padding

    item_logits, val_logits, dt_logits = model(
        itemid, source, delta_hours, value, padding_mask
    )
    print(f"itemid_logits : {list(item_logits.shape)}  (expect [{B},{L},{cfg.vocab_size}])")
    print(f"value_logits  : {list(val_logits.shape)}   (expect [{B},{L},{cfg.n_value_bins}])")
    print(f"delta_logits  : {list(dt_logits.shape)}    (expect [{B},{L},{cfg.n_time_bins}])")

    # Verify causal property: perturbing position k should not change logits at pos < k
    item_logits_2, _, _ = model(
        itemid.clone().fill_(0).index_fill_(1, torch.tensor([L-1]), 42),
        source, delta_hours, value, padding_mask
    )
    # Logits at positions 0..L-2 should be UNCHANGED (perturbation was at last position)
    unchanged = torch.allclose(item_logits[:, :-1], item_logits_2[:, :-1], atol=1e-5)
    print(f"Causal check  : {'PASS' if unchanged else 'FAIL'}")
    print("Self-test complete.")
