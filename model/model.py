"""
model/model.py

ICUFoundationModel — the complete model assembled from its parts.

Components (in order of data flow):
    ICUEventEmbedding   stage 2 — 4 projections → fused token [d_model]
    TransformerEncoder  stage 4 — 6-layer bidirectional encoder
    PretrainHead        phase 1 — predicts masked itemid + value
    MortalityHead       phase 2 — [CLS] → mortality logit
    LOSHead             phase 2 — [CLS] → LOS in days

Three operating modes controlled by the `mode` argument to forward():

    mode='pretrain'
        Applies Masked Event Modeling.
        Requires: masked_labels, value_labels (from apply_random_mask).
        Returns:  {'loss', 'itemid_loss', 'value_loss'}

    mode='finetune'
        Supervised downstream training (mortality + LOS + vital forecasting).
        Requires: mortality_labels, los_labels,
                  vital_labels [B, N_VITALS], vital_mask [B, N_VITALS].
        Returns:  {'loss', 'mortality_loss', 'los_loss', 'vital_loss',
                   'mortality_logits', 'los_preds', 'vital_preds'}
        Loss = 1.0*BCE(mort) + 0.05*MSE(LOS) + 0.5*masked_MSE(vitals)

    mode='encode'
        Inference / embedding extraction — no labels needed.
        Returns:  {'cls': [B, d_model], 'sequence': [B, L, d_model]}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config    import ModelConfig
from model.embedding import ICUEventEmbedding
from model.encoder   import TransformerEncoder
from model.heads     import PretrainHead, MortalityHead, LOSHead, ForecastHead


class ICUFoundationModel(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ── Core layers ──────────────────────────────────────────────────────
        self.embedding = ICUEventEmbedding(config)
        self.encoder   = TransformerEncoder(config)

        # ── Output heads ─────────────────────────────────────────────────────
        self.pretrain_head  = PretrainHead(config)
        self.mortality_head = MortalityHead(config)
        self.los_head       = LOSHead(config)
        self.forecast_head  = ForecastHead(config)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear layers; small normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ── Shared encode step ────────────────────────────────────────────────────

    def encode(
        self,
        itemid       : torch.Tensor,   # [B, L]  LongTensor
        source       : torch.Tensor,   # [B, L]  LongTensor
        delta_hours  : torch.Tensor,   # [B, L]  FloatTensor
        value        : torch.Tensor,   # [B, L]  FloatTensor
        padding_mask : torch.Tensor,   # [B, L]  BoolTensor  True=real event
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Shared forward pass: embedding → encoder.
        Returns:
            cls_output      [B, d_model]    — [CLS] token at position 0
            sequence_output [B, L, d_model] — all token representations
        """
        token_embs      = self.embedding(itemid, source, delta_hours, value)
        sequence_output = self.encoder(token_embs, padding_mask)
        cls_output      = sequence_output[:, 0, :]    # [CLS] is always index 0
        return cls_output, sequence_output

    # ── Mode-specific forward passes ──────────────────────────────────────────

    def _forward_pretrain(
        self,
        itemid        : torch.Tensor,   # [B, L]  — contains [MASK] tokens
        source        : torch.Tensor,
        delta_hours   : torch.Tensor,
        value         : torch.Tensor,
        padding_mask  : torch.Tensor,
        masked_labels : torch.Tensor,   # [B, L]  original itemid; -100 at non-masked
        value_labels  : torch.Tensor,   # [B, L]  original value;  0.0 at non-masked
    ) -> dict:
        """
        Masked Event Modeling loss.
        L = λ · CE(itemid) + (1-λ) · MSE(value)
        Both losses are computed only at masked positions.
        """
        _, sequence_output = self.encode(itemid, source, delta_hours, value, padding_mask)

        itemid_logits, value_preds = self.pretrain_head(sequence_output)

        # itemid loss — cross_entropy ignores positions where label == -100
        itemid_loss = F.cross_entropy(
            itemid_logits.view(-1, self.config.vocab_size),
            masked_labels.view(-1),
            ignore_index=-100,
        )

        # value loss — only at masked positions (where label != -100)
        masked_positions = (masked_labels != -100)
        if masked_positions.any():
            value_loss = F.mse_loss(
                value_preds[masked_positions],
                value_labels[masked_positions],
            )
        else:
            value_loss = torch.tensor(0.0, device=itemid_loss.device)

        lam        = self.config.value_loss_weight
        total_loss = lam * itemid_loss + (1.0 - lam) * value_loss

        return {
            "loss"        : total_loss,
            "itemid_loss" : itemid_loss.detach(),
            "value_loss"  : value_loss.detach(),
        }

    def _forward_finetune(
        self,
        itemid           : torch.Tensor,   # [B, L]
        source           : torch.Tensor,
        delta_hours      : torch.Tensor,
        value            : torch.Tensor,
        padding_mask     : torch.Tensor,
        mortality_labels : torch.Tensor,   # [B]      float  0 or 1
        los_labels       : torch.Tensor,   # [B]      float  days
        vital_labels     : torch.Tensor,   # [B, N_VITALS]  normalised values
        vital_mask       : torch.Tensor,   # [B, N_VITALS]  bool True=observed
    ) -> dict:
        """
        Multi-task fine-tuning loss.
        L = 1.0*BCE(mortality) + 0.05*MSE(LOS) + 0.5*masked_MSE(vitals)
        Loss weights balance the different scales of each task.
        """
        cls_output, _ = self.encode(itemid, source, delta_hours, value, padding_mask)

        mortality_logits = self.mortality_head(cls_output)   # [B]
        los_preds        = self.los_head(cls_output)         # [B]
        vital_preds      = self.forecast_head(cls_output)    # [B, N_VITALS]

        mortality_loss = F.binary_cross_entropy_with_logits(
            mortality_logits, mortality_labels.float(),
        )
        los_loss = F.mse_loss(los_preds, los_labels.float())

        if vital_mask.any():
            vital_loss = F.mse_loss(
                vital_preds[vital_mask],
                vital_labels[vital_mask],
            )
        else:
            vital_loss = torch.tensor(0.0, device=cls_output.device)

        total_loss = mortality_loss + 0.05 * los_loss + 0.5 * vital_loss

        return {
            "loss"             : total_loss,
            "mortality_loss"   : mortality_loss.detach(),
            "los_loss"         : los_loss.detach(),
            "vital_loss"       : vital_loss.detach(),
            "mortality_logits" : mortality_logits.detach(),
            "los_preds"        : los_preds.detach(),
            "vital_preds"      : vital_preds.detach(),
        }

    # ── Unified entry point ────────────────────────────────────────────────────

    def forward(self, mode: str = "pretrain", **kwargs) -> dict:
        """
        Unified forward pass.

        mode = 'pretrain' → _forward_pretrain(**kwargs)
        mode = 'finetune' → _forward_finetune(**kwargs)
        mode = 'encode'   → returns {'cls': ..., 'sequence': ...}
        """
        if mode == "pretrain":
            return self._forward_pretrain(**kwargs)
        elif mode == "finetune":
            return self._forward_finetune(**kwargs)
        elif mode == "encode":
            cls_out, seq_out = self.encode(**kwargs)
            return {"cls": cls_out, "sequence": seq_out}
        else:
            raise ValueError(
                f"Unknown mode '{mode}'. Choose: pretrain | finetune | encode"
            )

    # ── Utility ───────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_encoder(self):
        """Freeze embedding + encoder weights (used in fine-tune phase 2a)."""
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze everything (used after warm-up epoch in fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = ModelConfig(vocab_size=500)
    model  = ICUFoundationModel(config)

    print(f"Parameters : {model.count_parameters():,}")

    B, L = 2, 512
    dummy_inputs = dict(
        itemid       = torch.randint(3, 500, (B, L)),
        source       = torch.randint(0,   4, (B, L)),
        delta_hours  = torch.rand(B, L) * 72,
        value        = torch.randn(B, L),
        padding_mask = torch.ones(B, L, dtype=torch.bool),
    )

    # Pretrain forward
    masked_labels = torch.full((B, L), -100, dtype=torch.long)
    masked_labels[:, 5:10] = dummy_inputs['itemid'][:, 5:10]
    value_labels  = torch.zeros(B, L)
    value_labels[:, 5:10] = dummy_inputs['value'][:, 5:10]

    out = model.forward(
        mode='pretrain',
        masked_labels=masked_labels,
        value_labels=value_labels,
        **dummy_inputs,
    )
    print(f"Pretrain loss : {out['loss'].item():.4f}")

    # Finetune forward
    out = model.forward(
        mode='finetune',
        mortality_labels=torch.tensor([1.0, 0.0]),
        los_labels=torch.tensor([3.5, 7.2]),
        **dummy_inputs,
    )
    print(f"Finetune loss : {out['loss'].item():.4f}")
    print("All checks passed.")
