"""
training/utils.py

Shared utilities used by both pretrain.py and finetune.py:

    AverageMeter       — tracks running mean of a scalar (e.g. loss)
    EarlyStopping      — stops training when val loss stops improving
    save_checkpoint    — saves model + optimizer state to disk
    load_checkpoint    — loads a checkpoint back into model + optimizer
    apply_random_mask  — randomly masks events for MEM pretraining (BERT 80/10/10)
    vicreg_loss        — VICReg contrastive loss between two [CLS] projections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from model.config import ModelConfig


# ── Running statistics ────────────────────────────────────────────────────────

class AverageMeter:
    """Tracks a running mean of any scalar across multiple update() calls."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}" if self.name else f"{self.avg:.4f}"


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training if validation loss has not improved by at least
    `min_delta` for `patience` consecutive epochs.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_loss   = float("inf")
        self.wait        = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait      = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(
    model     : nn.Module,
    optimizer : torch.optim.Optimizer,
    epoch     : int,
    loss      : float,
    path      : Path,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch"               : epoch,
        "loss"                : loss,
        "model_state_dict"    : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"  Saved  {path.name}  (epoch={epoch}, loss={loss:.4f})")


def load_checkpoint(
    path      : Path,
    model     : nn.Module,
    optimizer : torch.optim.Optimizer | None = None,
    device    : torch.device = torch.device("cpu"),
) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss  = checkpoint["loss"]
    print(f"  Loaded {path.name}  (epoch={epoch}, loss={loss:.4f})")
    return epoch


# ── Masking for Masked Event Modeling ────────────────────────────────────────

def apply_random_mask(
    inputs : dict,
    config : ModelConfig,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """
    Randomly masks 15% of real events for Masked Event Modeling (MEM).

    Masking strategy — BERT 80/10/10 applied to selected positions:
        80%: replace itemid with [MASK] token
        10%: replace itemid with a random real token
        10%: keep itemid unchanged (the model must still reconstruct it)
    At ALL selected positions, the float value is zeroed out in the input
    to prevent the model from reconstructing itemid from the visible value.

    Rules:
        - [CLS] at position 0 is NEVER masked.
        - [PAD] positions are NEVER masked.
        - The integer value_bins field (if present) is used as the classification
          label; the float value field is zeroed at masked positions.

    Returns:
        masked_inputs      — dict with some itemids replaced and masked values zeroed
        itemid_labels      — [B, L] LongTensor; original itemid at masked positions,
                             -100 elsewhere (CrossEntropy ignores -100)
        value_bin_labels   — [B, L] LongTensor; original value bin at masked positions,
                             -100 elsewhere (CrossEntropy ignores -100)
    """
    itemid       = inputs["itemid"].clone()       # [B, L]
    value        = inputs["value"].clone()        # [B, L]  float Z-scored
    value_bins   = inputs.get("value_bins")       # [B, L]  int bin indices (may be absent)
    padding_mask = inputs["padding_mask"]         # [B, L]  True=real event

    # Positions eligible for masking: real events, not [CLS]
    eligible       = padding_mask.clone()
    eligible[:, 0] = False

    # Select 15% of eligible positions
    rand           = torch.rand_like(eligible.float())
    mask_positions = eligible & (rand < config.mask_prob)   # [B, L] bool

    # ── Targets (set before modifying itemid) ─────────────────────────────────
    itemid_labels    = torch.full_like(itemid, fill_value=-100)
    value_bin_labels = torch.full_like(itemid, fill_value=-100)

    itemid_labels[mask_positions] = itemid[mask_positions]

    if value_bins is not None:
        value_bin_labels[mask_positions] = value_bins[mask_positions]

    # ── BERT 80/10/10 replacement strategy ───────────────────────────────────
    rand2      = torch.rand_like(itemid.float())
    is_mask    = mask_positions & (rand2 < 0.80)            # 80%: → [MASK]
    is_random  = mask_positions & (rand2 >= 0.80) & (rand2 < 0.90)  # 10%: random token
    # remaining 10%: keep original itemid (but label still set, no value signal)

    itemid[is_mask] = config.mask_token_id

    if is_random.any():
        n_random = int(is_random.sum().item())
        # Draw from real tokens only (indices >= unk_token_id + 1)
        first_real = config.unk_token_id + 1
        random_ids = torch.randint(
            first_real, config.vocab_size, (n_random,), device=itemid.device
        )
        itemid[is_random] = random_ids

    # ── Zero out float value at ALL masked positions (close the leak) ─────────
    value[mask_positions] = 0.0

    masked_inputs = {**inputs, "itemid": itemid, "value": value}
    return masked_inputs, itemid_labels, value_bin_labels


# ── ICU-specific augmentation for VICReg ─────────────────────────────────────

# Matches PAD_TOKEN_ID in dataloader.py and ModelConfig.pad_token_id
_PAD_TOKEN_ID = 0


def augment(
    inputs     : dict,
    drop_p     : float = 0.15,   # event dropout probability
    val_sigma  : float = 0.10,   # std of value jitter (in Z-score units)
    time_sigma : float = 0.50,   # std of timing jitter (in hours)
) -> dict:
    """
    ICU-specific data augmentation for VICReg.

    Perturbs only NUISANCE factors while preserving the clinical state:

        Event dropout  — randomly removes events (pad_p=0.15).  Models the fact
                         that different nurses may chart different subsets of the
                         same underlying physiology.

        Value jitter   — adds small Gaussian noise to Z-scored values.  Models
                         routine measurement noise (e.g., HR ±3 bpm).

        Timing jitter  — adds small Gaussian noise to delta_hours.  Models minor
                         charting delays (e.g., a note entered 30 min late).

    What is intentionally NOT done:
        - Large temporal crops (different time windows of the same stay) would
          force the model to treat a patient at admission and near death as
          equivalent — destroying exactly the clinical trajectory signal that
          downstream tasks depend on.
        - Token replacement ([MASK]) would corrupt the sequence semantics.
          apply_random_mask() is for MEM pretraining, NOT for VICReg augmentation.

    [CLS] (position 0) is never modified.
    [PAD] positions are never modified.
    """
    itemid = inputs["itemid"].clone()
    source = inputs["source"].clone()
    delta  = inputs["delta_hours"].clone()
    value  = inputs["value"].clone()
    mask   = inputs["padding_mask"].clone()

    # Eligible: real events excluding [CLS]
    eligible       = mask.clone()
    eligible[:, 0] = False

    # Event dropout: remove event entirely (set to PAD, zero value)
    drop = eligible & (torch.rand_like(value) < drop_p)
    itemid[drop] = _PAD_TOKEN_ID
    value[drop]  = 0.0
    mask[drop]   = False

    # Value jitter on surviving real events
    keep = eligible & ~drop
    value[keep] = value[keep] + torch.randn_like(value[keep]) * val_sigma

    # Timing jitter on surviving real events (clamp to non-negative hours)
    delta[keep] = (
        delta[keep] + torch.randn_like(delta[keep]) * time_sigma
    ).clamp(min=0.0)

    out = dict(inputs)
    out.update({"itemid": itemid, "source": source,
                "delta_hours": delta, "value": value, "padding_mask": mask})
    return out


# ── VICReg contrastive loss ────────────────────────────────────────────────────

def vicreg_loss(
    z1      : torch.Tensor,  # [B, D] projected [CLS] from view 1
    z2      : torch.Tensor,  # [B, D] projected [CLS] from view 2
    lambda_ : float = 25.0,  # variance term weight
    mu      : float = 25.0,  # invariance term weight
    nu      : float = 1.0,   # covariance term weight
) -> torch.Tensor:
    """
    VICReg loss (Bardes et al. 2022) between two augmented views.

    Three terms:
      - Invariance: MSE between z1 and z2 — views of the same stay should be similar
      - Variance:   penalise per-dimension variance collapsing below 1
      - Covariance: penalise off-diagonal covariance — decorrelates dimensions

    Applied to [CLS] representations from two differently masked versions of
    the same batch, encouraging stay-level representations to be:
      (a) invariant to masking pattern (which clinical values are visible),
      (b) non-degenerate (each dimension carries information),
      (c) disentangled (dimensions are statistically independent).
    """
    N, D = z1.shape

    # Invariance: representations of the two views should be close
    sim_loss = F.mse_loss(z1, z2)

    # Variance: each dimension should have std >= 1 within the batch
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var_loss = (F.relu(1.0 - std_z1).mean() + F.relu(1.0 - std_z2).mean()) / 2

    # Covariance: off-diagonal elements of normalised covariance → 0
    # Divide by D (not D²) to keep gradient scale comparable across dimensions.
    z1_c = z1 - z1.mean(dim=0)
    z2_c = z2 - z2.mean(dim=0)
    cov_z1 = (z1_c.T @ z1_c) / (N - 1)
    cov_z2 = (z2_c.T @ z2_c) / (N - 1)

    def _off_diag_loss(cov):
        # Zero out diagonal, then penalise remaining elements
        off = cov - torch.diag(torch.diag(cov))
        return off.pow(2).sum() / D

    cov_loss = (_off_diag_loss(cov_z1) + _off_diag_loss(cov_z2)) / 2

    return mu * sim_loss + lambda_ * var_loss + nu * cov_loss
