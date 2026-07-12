"""
model/heads.py

Output heads that attach to the encoder output.

─────────────────────────────────────────────────────────────
Phase 1 — Pretraining  (Masked Event Modeling)
─────────────────────────────────────────────────────────────
PretrainHead
    Input  : full sequence output  [B, L, d_model]
    Outputs: itemid_logits         [B, L, vocab_size]     → CrossEntropy
             value_logits          [B, L, n_value_bins]   → CrossEntropy over bins
    Loss computed only at masked positions (training loop handles masking).

    Both objectives use CrossEntropy — itemid predicts which measurement was masked,
    value_bins predicts which decile bin the masked value fell into.
    This avoids MSE's heavy-tail dominance problem on clinical measurements.

─────────────────────────────────────────────────────────────
Phase 2 — Fine-tuning  (supervised downstream tasks)
─────────────────────────────────────────────────────────────
All heads take the [CLS] token representation as input [B, d_model].

MortalityHead   → binary classification  (did the patient die?)
LOSHead         → regression             (ICU length of stay in days)
ForecastHead    → regression per vital   (future vital sign values, normalised)
                  Targets: mean of each vital in the LAST 20% of the stay's events.
                  Vitals: HR, SpO2, RR, SBP, DBP, MAP, Temperature (7 total).
"""

# ── Vital sign constants (shared with dataloader) ─────────────────────────────
# MIMIC-IV chartevents itemids for the 7 primary ICU vitals
VITAL_ITEMIDS = [220045, 220277, 220210, 220179, 220180, 220052, 223762]
VITAL_NAMES   = ["heart_rate", "spo2", "resp_rate", "sbp", "dbp", "map", "temp"]
N_VITALS      = len(VITAL_ITEMIDS)

# ── Proxy-target constants (shared with dataloader) ───────────────────────────
# Self-supervised stay-level targets derived from the raw data — no human labels.
# Four targets (indices 0-3):
#   0  log_event_count  log(n_events + 1) / log(5001)   continuous float in [0,1]
#   1  mean_hr_zscore   mean Z-scored HR over the stay   continuous float (NaN → 0)
#   2  any_vasopressor  1 if any vasopressor was given   binary
#   3  any_critical_lab 1 if lactate > 4 or pH < 7.25   binary
N_PROXY_TARGETS = 4

# MIMIC-IV inputevents itemids for the 5 main vasopressors
VASOPRESSOR_ITEMIDS = {221906, 221289, 221662, 222315, 221749}
# norepinephrine, epinephrine, dopamine, vasopressin, phenylephrine

# Lactate itemids (arterial and venous) and arterial pH itemid
LACTATE_ITEMIDS = {225668, 225670}
PH_ITEMID       = 220274

import torch
import torch.nn as nn
from model.config import ModelConfig


class PretrainHead(nn.Module):
    """
    Predicts the masked event's itemid (classification) and value quantile bin
    (classification over decile bins). Applied to every position in the sequence,
    but the training loop computes loss only at masked positions.
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

        # Value bin prediction: classify into n_value_bins decile bins
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.n_value_bins),
        )

    def forward(
        self,
        sequence_output: torch.Tensor,  # [B, L, d_model]
    ) -> tuple[torch.Tensor, torch.Tensor]:

        itemid_logits = self.itemid_head(sequence_output)   # [B, L, vocab_size]
        value_logits  = self.value_head(sequence_output)    # [B, L, n_value_bins]

        return itemid_logits, value_logits


class MortalityHead(nn.Module):
    """
    Binary classifier: did the patient die in hospital?
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
        return self.regressor(cls_output).squeeze(-1)   # [B]


class ForecastHead(nn.Module):
    """
    Predicts normalised future vital sign values from the [CLS] token.

    The 7 vitals (HR, SpO2, RR, SBP, DBP, MAP, Temp) are the mean of each
    vital's z-scored values measured in the LAST 20% of the stay's events
    (i.e., events at raw-CSV index >= 80% of total stay length).
    Where a vital is not observed in that future window, vital_mask is False
    and the position is excluded from the MSE loss.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, N_VITALS),
        )

    def forward(self, cls_output: torch.Tensor) -> torch.Tensor:
        return self.regressor(cls_output)   # [B, N_VITALS]


class ProxyHead(nn.Module):
    """
    Predicts N_PROXY_TARGETS stay-level proxy targets from [CLS].

    These are self-supervised targets derived entirely from the raw event data —
    no human annotation required.  Directly trains the [CLS] representation
    without needing contrastive augmentation (unlike VICReg), so there is no
    risk of the model learning to be invariant to clinically relevant variation.

    Targets:
        0  log_event_count  (continuous) — log-normalised total event count
        1  mean_hr_zscore   (continuous) — mean Z-scored HR during stay
        2  any_vasopressor  (binary)     — any vasopressor was given
        3  any_critical_lab (binary)     — lactate > 4 mmol/L or pH < 7.25

    Loss per target:
        0, 1 → MSE (only where proxy_mask[i] is True)
        2, 3 → BCE-with-logits (proxy_mask is always True for these)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, N_PROXY_TARGETS),
        )

    def forward(self, cls_output: torch.Tensor) -> torch.Tensor:
        return self.head(cls_output)   # [B, N_PROXY_TARGETS]
