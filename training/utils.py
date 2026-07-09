"""
training/utils.py

Shared utilities used by both pretrain.py and finetune.py:

    AverageMeter       — tracks running mean of a scalar (e.g. loss)
    EarlyStopping      — stops training when val loss stops improving
    save_checkpoint    — saves model + optimizer state to disk
    load_checkpoint    — loads a checkpoint back into model + optimizer
    apply_random_mask  — randomly masks events for MEM pretraining
"""

import torch
import torch.nn as nn
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
        """Add a new value computed over n samples."""
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
        """
        Call once per epoch with the current validation loss.
        Returns True when training should stop.
        """
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
    """Saves model weights, optimizer state, epoch, and loss to a .pt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch"               : epoch,
        "loss"                : loss,
        "model_state_dict"    : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"  ✓ Saved  {path.name}  (epoch={epoch}, loss={loss:.4f})")


def load_checkpoint(
    path      : Path,
    model     : nn.Module,
    optimizer : torch.optim.Optimizer | None = None,
    device    : torch.device = torch.device("cpu"),
) -> int:
    """
    Loads a checkpoint into model (and optionally optimizer).
    Returns the epoch at which the checkpoint was saved.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss  = checkpoint["loss"]
    print(f"  ✓ Loaded {path.name}  (epoch={epoch}, loss={loss:.4f})")
    return epoch


# ── Masking for Masked Event Modeling ────────────────────────────────────────

def apply_random_mask(
    inputs : dict,
    config : ModelConfig,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """
    Randomly masks 15% of real events for Masked Event Modeling (MEM).

    Rules:
        - [CLS] at position 0 is NEVER masked.
        - [PAD] positions are NEVER masked.
        - Selected events have their itemid replaced with [MASK] token id.

    Returns:
        masked_inputs   — same dict as inputs but with some itemids replaced
        itemid_labels   — [B, L] LongTensor; original itemid at masked positions,
                          -100 elsewhere (CrossEntropy ignores -100)
        value_labels    — [B, L] FloatTensor; original Z-scored value at masked
                          positions, 0.0 elsewhere
    """
    itemid       = inputs["itemid"].clone()       # [B, L]
    value        = inputs["value"].clone()        # [B, L]
    padding_mask = inputs["padding_mask"]         # [B, L]  True=real event

    # Build a boolean mask of positions eligible for masking:
    #   must be a real event (not padding) AND not the [CLS] token at index 0
    eligible          = padding_mask.clone()
    eligible[:, 0]    = False    # protect [CLS]

    # Sample: each eligible position is independently masked with prob=mask_prob
    rand           = torch.rand_like(eligible.float())
    mask_positions = eligible & (rand < config.mask_prob)   # [B, L] bool

    # Targets: true values at masked positions, sentinels elsewhere
    itemid_labels                  = torch.full_like(itemid, fill_value=-100)
    value_labels                   = torch.zeros_like(value)
    itemid_labels[mask_positions]  = itemid[mask_positions]
    value_labels[mask_positions]   = value[mask_positions]

    # Replace masked itemids with the [MASK] special token
    itemid[mask_positions] = config.mask_token_id

    masked_inputs = {**inputs, "itemid": itemid}
    return masked_inputs, itemid_labels, value_labels
