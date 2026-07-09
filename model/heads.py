"""
model/heads.py

Output heads that attach to the encoder output.

─────────────────────────────────────────────────────────────
Phase 1 — Pretraining  (Masked Event Modeling)
─────────────────────────────────────────────────────────────
PretrainHead
    Input  : full sequence output  [B, L, d_model]
    Outputs: itemid_logits         [B, L, vocab_size]  → CrossEntropy
             value_preds           [B, L]              → MSE
    Loss computed only at masked positions (training loop handles masking).

─────────────────────────────────────────────────────────────
Phase 2 — Fine-tuning  (supervised downstream tasks)
─────────────────────────────────────────────────────────────
Both heads take the [CLS] token representation as input [B, d_model].
The [CLS] token attends to the entire stay and becomes the patient summary.

MortalityHead   → binary classification  (did the patient die?)
LOSHead         → regression             (ICU length of stay in days)
"""

import torch
import torch.nn as nn
from model.config import ModelConfig


class PretrainHead(nn.Module):
    """
    Predicts the masked event's itemid (classification) and value (regression).
    Applied to every position in the sequence, but the training loop computes
    loss only at masked positions using the provided label tensors.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Itemid prediction: project to vocab logits with a hidden bottleneck
        self.itemid_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.vocab_size),
        )

        # Value prediction: single scalar output
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
        )

    def forward(
        self,
        sequence_output: torch.Tensor,  # [B, L, d_model]
    ) -> tuple[torch.Tensor, torch.Tensor]:

        itemid_logits = self.itemid_head(sequence_output)         # [B, L, vocab_size]
        value_preds   = self.value_head(sequence_output).squeeze(-1)  # [B, L]

        return itemid_logits, value_preds


class MortalityHead(nn.Module):
    """
    Binary classifier: did the patient die in hospital?
    (label = hospital_expire_flag from MIMIC admissions table)
    Output is a raw logit — apply sigmoid externally or use BCE-with-logits.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
        )

    def forward(self, cls_output: torch.Tensor) -> torch.Tensor:
        # cls_output: [B, d_model]
        return self.classifier(cls_output).squeeze(-1)  # [B]  raw logit


class LOSHead(nn.Module):
    """
    Regression: predicts ICU length of stay in days.
    Output is an unbounded scalar — apply ReLU externally if needed.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.regressor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
        )

    def forward(self, cls_output: torch.Tensor) -> torch.Tensor:
        # cls_output: [B, d_model]
        return self.regressor(cls_output).squeeze(-1)   # [B]
