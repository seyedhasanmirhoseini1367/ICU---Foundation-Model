"""
model/model.py

ICUFoundationModel — the complete model assembled from its parts.

Components (in order of data flow):
    ICUEventEmbedding   stage 2 — 4 projections → fused token [d_model]
    TransformerEncoder  stage 4 — 6-layer bidirectional encoder
    PretrainHead        phase 1 — predicts masked itemid + value bin
    MortalityHead       phase 2 — [CLS] → mortality logit
    LOSHead             phase 2 — [CLS] → LOS in days
    ForecastHead        phase 2 — [CLS] → future vital sign values
    vicreg_projector    phase 1 — [CLS] → projection space for VICReg loss

Three operating modes controlled by the `mode` argument to forward():

    mode='pretrain'
        Applies Masked Event Modeling.
        Requires: masked_labels (itemid), value_bin_labels.
        Returns:  {'loss', 'itemid_loss', 'value_loss', 'cls'}
        Loss = λ·CE(itemid) + (1-λ)·CE(value_bins)
        'cls' is returned to support the optional VICReg objective in the
        training loop (two forward passes → two [CLS] → vicreg_loss).

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

        # ── VICReg projector — 3-layer MLP mapping [CLS] → projection space ──
        # Keeps the projector deeper than the encoder output to avoid losing
        # task-relevant information in the encoder itself (SimCLR finding).
        self.vicreg_projector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.BatchNorm1d(config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.BatchNorm1d(config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
        )

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
        Shared forward: embedding → encoder.
        Returns:
            cls_output      [B, d_model]    — [CLS] token at position 0
            sequence_output [B, L, d_model] — all token representations
        """
        token_embs      = self.embedding(itemid, source, delta_hours, value)
        sequence_output = self.encoder(token_embs, padding_mask)
        cls_output      = sequence_output[:, 0, :]
        return cls_output, sequence_output

    # ── Mode-specific forward passes ──────────────────────────────────────────

    def _forward_pretrain(
        self,
        itemid           : torch.Tensor,   # [B, L]  — contains [MASK] tokens
        source           : torch.Tensor,
        delta_hours      : torch.Tensor,
        value            : torch.Tensor,   # [B, L]  — zeroed at masked positions
        padding_mask     : torch.Tensor,
        masked_labels    : torch.Tensor,   # [B, L]  original itemid; -100 elsewhere
        value_bin_labels : torch.Tensor,   # [B, L]  original bin idx; -100 elsewhere
    ) -> dict:
        """
        Masked Event Modeling loss.
        L = λ·CE(itemid) + (1-λ)·CE(value_bins)
        Both losses computed only at masked positions (ignore_index=-100).

        Using CE for value bins (not MSE on the raw float) provides stronger
        gradient signal on heavy-tailed clinical measurements.
        """
        cls_output, sequence_output = self.encode(
            itemid, source, delta_hours, value, padding_mask
        )

        itemid_logits, value_logits = self.pretrain_head(sequence_output)
        # itemid_logits: [B, L, vocab_size]
        # value_logits:  [B, L, n_value_bins]

        # Itemid loss — CE ignores -100
        itemid_loss = F.cross_entropy(
            itemid_logits.view(-1, self.config.vocab_size),
            masked_labels.view(-1),
            ignore_index=-100,
        )

        # Value bin loss — CE ignores -100
        masked_pos = (value_bin_labels != -100)
        if masked_pos.any():
            value_loss = F.cross_entropy(
                value_logits.view(-1, self.config.n_value_bins)[masked_pos.view(-1)],
                value_bin_labels.view(-1)[masked_pos.view(-1)],
            )
        else:
            value_loss = torch.tensor(0.0, device=cls_output.device)

        lam        = self.config.value_loss_weight
        total_loss = lam * itemid_loss + (1.0 - lam) * value_loss

        return {
            "loss"        : total_loss,
            "itemid_loss" : itemid_loss.detach(),
            "value_loss"  : value_loss.detach(),
            "cls"         : cls_output,   # returned for optional VICReg in training loop
        }

    def _forward_finetune(
        self,
        itemid           : torch.Tensor,
        source           : torch.Tensor,
        delta_hours      : torch.Tensor,
        value            : torch.Tensor,
        padding_mask     : torch.Tensor,
        mortality_labels : torch.Tensor,   # [B]      float  0 or 1
        los_labels       : torch.Tensor,   # [B]      float  days
        vital_labels     : torch.Tensor,   # [B, N_VITALS]
        vital_mask       : torch.Tensor,   # [B, N_VITALS]  bool
    ) -> dict:
        """
        Multi-task fine-tuning loss.
        L = 1.0*BCE(mortality) + 0.05*MSE(LOS) + 0.5*masked_MSE(vitals)
        """
        cls_output, _ = self.encode(itemid, source, delta_hours, value, padding_mask)

        mortality_logits = self.mortality_head(cls_output)
        los_preds        = self.los_head(cls_output)
        vital_preds      = self.forecast_head(cls_output)

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
        """Freeze embedding + encoder weights (fine-tune phase 2a: heads only)."""
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze everything (fine-tune phase 2b: full model)."""
        for param in self.parameters():
            param.requires_grad = True


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from model.heads import N_VITALS
    config = ModelConfig(vocab_size=500)
    model  = ICUFoundationModel(config)

    print(f"Parameters : {model.count_parameters():,}")

    B, L = 2, 512
    dummy = dict(
        itemid       = torch.randint(4, 500, (B, L)),
        source       = torch.randint(0,   4, (B, L)),
        delta_hours  = torch.rand(B, L) * 72,
        value        = torch.randn(B, L),
        padding_mask = torch.ones(B, L, dtype=torch.bool),
    )

    masked_labels    = torch.full((B, L), -100, dtype=torch.long)
    masked_labels[:, 5:10] = dummy["itemid"][:, 5:10]
    value_bin_labels = torch.full((B, L), -100, dtype=torch.long)
    value_bin_labels[:, 5:10] = torch.randint(0, config.n_value_bins, (B, 5))

    out = model.forward(
        mode="pretrain",
        masked_labels=masked_labels,
        value_bin_labels=value_bin_labels,
        **dummy,
    )
    print(f"Pretrain loss : {out['loss'].item():.4f}  cls shape: {out['cls'].shape}")

    out = model.forward(
        mode="finetune",
        mortality_labels=torch.tensor([1.0, 0.0]),
        los_labels=torch.tensor([3.5, 7.2]),
        vital_labels=torch.zeros(B, N_VITALS),
        vital_mask=torch.zeros(B, N_VITALS, dtype=torch.bool),
        **dummy,
    )
    print(f"Finetune loss : {out['loss'].item():.4f}")
    print("All checks passed.")
