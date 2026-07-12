"""
training/pretrain.py

Phase 1: Masked Event Modeling (MEM) pretraining.

Each training step:
    1. Load a clean batch of ICU stay sequences (random window augmentation).
    2. Create two differently masked views for VICReg (if USE_VICREG=1).
    3. Randomly mask 15% of real events with BERT 80/10/10 strategy.
    4. Forward pass → encoder + PretrainHead.
    5. Loss = λ·CE(itemid) + (1-λ)·CE(value_bins)  [only at masked positions]
       + vicreg_weight * VICReg([CLS]_1, [CLS]_2)  [optional]
    6. Backprop + AdamW step + gradient clipping.

Early stopping uses HELD-OUT VALIDATION LOSS, not train loss.
The dataset is split 90/10 by patient at run time (no leakage into finetune split).

Run:
    python training/pretrain.py

Checkpoints saved to:
    checkpoints/pretrain/epoch_NNN.pt   — every epoch
    checkpoints/pretrain/best.pt        — best val loss

wandb:
    Set USE_WANDB=1 to enable live logging.

VICReg:
    Set USE_VICREG=1 to enable the contrastive CLS objective.
    Disabled by default (doubles compute per step — not suitable for Kaggle T4).
"""

import json
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.config          import ModelConfig
from model.model           import ICUFoundationModel
from dataloader.dataloader import make_dataloader
from training.utils        import (
    AverageMeter, EarlyStopping,
    save_checkpoint, load_checkpoint,
    apply_random_mask, vicreg_loss,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parent.parent
DATA_DIR        = ROOT / "dataloader" / "all_stays"
INDEX_PATH      = ROOT / "dataloader" / "index.csv"
VOCAB_PATH      = ROOT / "tokenizer" / "vocab.json"
NORM_PATH       = ROOT / "tokenizer" / "norm_stats.json"
BIN_EDGES_PATH  = ROOT / "tokenizer" / "bin_edges.json"
CKPT_DIR        = ROOT / "checkpoints" / "pretrain"

# ── GPU capability gate ───────────────────────────────────────────────────────
def _cuda_ok() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability(0)
    if cap[0] < 7:
        name = torch.cuda.get_device_name(0)
        print(f"WARNING: {name} has CUDA capability sm_{cap[0]}{cap[1]} "
              f"(PyTorch 2.x requires sm_70+) — using CPU instead.")
        return False
    return True

_CUDA_OK = _cuda_ok()

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE    = 32 if _CUDA_OK else 8
MAX_EPOCHS    = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-2
PATIENCE      = 5
VAL_FRACTION  = 0.10   # 10% of patients held out for pretrain validation

# ── Feature flags ─────────────────────────────────────────────────────────────
USE_WANDB  = os.getenv("USE_WANDB",  "0") == "1"
USE_VICREG = os.getenv("USE_VICREG", "0") == "1"

WANDB_ENTITY  = "seyedhasan-mirhoseini1367-tampere-university"
WANDB_PROJECT = "MIMIC-IV-ICU"


def _wlog(metrics: dict, step: int):
    if USE_WANDB:
        import wandb
        wandb.log(metrics, step=step)


# ── Train / val split by patient ──────────────────────────────────────────────

def pretrain_split(index_path: Path, val_fraction: float, seed: int):
    """
    Split index.csv 90/10 by patient (subject_id) for pretrain validation.
    Patient-level split prevents leakage when one patient has multiple stays.
    """
    index    = pd.read_csv(index_path)
    subjects = index["subject_id"].unique()

    rng      = np.random.default_rng(seed)
    n_val    = max(1, int(len(subjects) * val_fraction))
    val_subj = set(rng.choice(subjects, size=n_val, replace=False))

    train_df = index[~index["subject_id"].isin(val_subj)].reset_index(drop=True)
    val_df   = index[ index["subject_id"].isin(val_subj)].reset_index(drop=True)
    return train_df, val_df


# ── Training and evaluation ───────────────────────────────────────────────────

def train_one_epoch(
    model     : ICUFoundationModel,
    loader    : torch.utils.data.DataLoader,
    optimizer : torch.optim.Optimizer,
    config    : ModelConfig,
    device    : torch.device,
) -> dict:
    model.train()

    total_m  = AverageMeter("loss")
    itemid_m = AverageMeter("itemid_loss")
    value_m  = AverageMeter("value_loss")
    vicreg_m = AverageMeter("vicreg_loss")

    for step, (inputs, _) in enumerate(loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ── MEM: first masked view ─────────────────────────────────────────
        masked1, itemid_labels, bin_labels = apply_random_mask(inputs, config)
        itemid_labels = itemid_labels.to(device)
        bin_labels    = bin_labels.to(device)

        out = model.forward(
            mode             = "pretrain",
            itemid           = masked1["itemid"],
            source           = masked1["source"],
            delta_hours      = masked1["delta_hours"],
            value            = masked1["value"],
            padding_mask     = masked1["padding_mask"],
            masked_labels    = itemid_labels,
            value_bin_labels = bin_labels,
        )

        loss = out["loss"]

        # ── VICReg: second masked view → contrastive CLS loss ─────────────
        vcr_val = 0.0
        if USE_VICREG:
            masked2, _, _ = apply_random_mask(inputs, config)
            with torch.no_grad():
                # Encode second view; no need to backprop through MEM heads
                pass
            out2 = model.forward(
                mode             = "pretrain",
                itemid           = masked2["itemid"],
                source           = masked2["source"],
                delta_hours      = masked2["delta_hours"],
                value            = masked2["value"],
                padding_mask     = masked2["padding_mask"],
                masked_labels    = itemid_labels,    # same labels (both views = same stay)
                value_bin_labels = bin_labels,
            )
            z1 = model.vicreg_projector(out["cls"])
            z2 = model.vicreg_projector(out2["cls"])
            vcr = vicreg_loss(
                z1, z2,
                lambda_=config.vicreg_lambda,
                mu=config.vicreg_mu,
                nu=config.vicreg_nu,
            )
            loss = loss + config.vicreg_weight * vcr
            vcr_val = vcr.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        n = inputs["itemid"].size(0)
        total_m.update(loss.item(),               n)
        itemid_m.update(out["itemid_loss"].item(), n)
        value_m.update(out["value_loss"].item(),   n)
        vicreg_m.update(vcr_val,                  n)

        if (step + 1) % max(1, len(loader) // 10) == 0:
            print(f"    step {step+1:>4}/{len(loader)}"
                  f"  loss={loss.item():.4f}"
                  f"  itemid={out['itemid_loss'].item():.4f}"
                  f"  value={out['value_loss'].item():.4f}"
                  + (f"  vicreg={vcr_val:.4f}" if USE_VICREG else ""))

    return {
        "loss"       : total_m.avg,
        "itemid_loss": itemid_m.avg,
        "value_loss" : value_m.avg,
        "vicreg_loss": vicreg_m.avg,
    }


@torch.no_grad()
def evaluate_pretrain(
    model  : ICUFoundationModel,
    loader : torch.utils.data.DataLoader,
    config : ModelConfig,
    device : torch.device,
) -> dict:
    """Compute MEM loss on the held-out validation set."""
    model.eval()

    total_m  = AverageMeter("val_loss")
    itemid_m = AverageMeter("val_itemid_loss")
    value_m  = AverageMeter("val_value_loss")

    for inputs, _ in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        masked, itemid_labels, bin_labels = apply_random_mask(inputs, config)

        out = model.forward(
            mode             = "pretrain",
            itemid           = masked["itemid"],
            source           = masked["source"],
            delta_hours      = masked["delta_hours"],
            value            = masked["value"],
            padding_mask     = masked["padding_mask"],
            masked_labels    = itemid_labels.to(device),
            value_bin_labels = bin_labels.to(device),
        )

        n = inputs["itemid"].size(0)
        total_m.update(out["loss"].item(),         n)
        itemid_m.update(out["itemid_loss"].item(), n)
        value_m.update(out["value_loss"].item(),   n)

    return {
        "val_loss"       : total_m.avg,
        "val_itemid_loss": itemid_m.avg,
        "val_value_loss" : value_m.avg,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if _CUDA_OK else "cpu")
    print(f"Device      : {device}")
    if _CUDA_OK:
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"VICReg      : {'ON' if USE_VICREG else 'OFF'}")

    vocab  = json.loads(VOCAB_PATH.read_text())
    config = ModelConfig(vocab_size=len(vocab))
    print(f"Vocab size  : {config.vocab_size} tokens")

    # ── Pretrain 90/10 patient split ──────────────────────────────────────────
    print(f"\nSplitting index by patient ({int(VAL_FRACTION*100)}% val) ...")
    train_df, val_df = pretrain_split(INDEX_PATH, VAL_FRACTION, config.random_seed)
    print(f"  Train: {len(train_df)} stays ({train_df['subject_id'].nunique()} patients)")
    print(f"  Val  : {len(val_df)}   stays ({val_df['subject_id'].nunique()} patients)")

    # Write temp split CSVs for the dataloaders
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    _train_idx = CKPT_DIR / "_pretrain_train.csv"
    _val_idx   = CKPT_DIR / "_pretrain_val.csv"
    train_df.to_csv(_train_idx, index=False)
    val_df.to_csv(_val_idx,   index=False)

    def _make_loader(idx_path, shuffle, window_mode):
        return make_dataloader(
            index_path      = idx_path,
            data_dir        = DATA_DIR,
            vocab_path      = VOCAB_PATH,
            norm_stats_path = NORM_PATH,
            bin_edges_path  = BIN_EDGES_PATH if BIN_EDGES_PATH.exists() else None,
            max_len         = config.max_len,
            batch_size      = BATCH_SIZE,
            shuffle         = shuffle,
            window_mode     = window_mode,
        )

    train_loader = _make_loader(_train_idx, shuffle=True,  window_mode="random")
    val_loader   = _make_loader(_val_idx,   shuffle=False, window_mode="last")

    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Train batches: {len(train_loader)}   Val batches: {len(val_loader)}")

    model = ICUFoundationModel(config).to(device)
    print(f"Parameters  : {model.count_parameters():,}")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
    stopper   = EarlyStopping(patience=PATIENCE)
    best_loss = float("inf")

    # ── wandb init ────────────────────────────────────────────────────────────
    if USE_WANDB:
        import wandb
        wandb.init(
            entity  = WANDB_ENTITY,
            project = WANDB_PROJECT,
            name    = f"pretrain-{len(train_df)}train-{len(val_df)}val",
            config  = {
                "phase"        : "pretrain",
                "n_train"      : len(train_df),
                "n_val"        : len(val_df),
                "val_fraction" : VAL_FRACTION,
                "vocab_size"   : config.vocab_size,
                "d_model"      : config.d_model,
                "n_layers"     : config.n_layers,
                "n_heads"      : config.n_heads,
                "d_ff"         : config.d_ff,
                "max_len"      : config.max_len,
                "mask_prob"    : config.mask_prob,
                "n_value_bins" : config.n_value_bins,
                "batch_size"   : BATCH_SIZE,
                "lr"           : LEARNING_RATE,
                "weight_decay" : WEIGHT_DECAY,
                "patience"     : PATIENCE,
                "max_epochs"   : MAX_EPOCHS,
                "use_vicreg"   : USE_VICREG,
                "device"       : str(device),
            },
        )

    print(f"\n── Pretraining (MEM) for up to {MAX_EPOCHS} epochs ──\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, config, device)
        val_m   = evaluate_pretrain(model, val_loader, config, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        vicreg_str = f"  vicreg={train_m['vicreg_loss']:.4f}" if USE_VICREG else ""
        print(f"Epoch {epoch:02d}/{MAX_EPOCHS}"
              f"  train={train_m['loss']:.4f}"
              f"  val={val_m['val_loss']:.4f}"
              f"  itemid={train_m['itemid_loss']:.4f}"
              f"  value={train_m['value_loss']:.4f}"
              f"{vicreg_str}"
              f"  lr={lr:.2e}")

        wlog_dict = {
            "pretrain/train_loss"       : train_m["loss"],
            "pretrain/val_loss"         : val_m["val_loss"],
            "pretrain/itemid_loss"      : train_m["itemid_loss"],
            "pretrain/value_loss"       : train_m["value_loss"],
            "pretrain/val_itemid_loss"  : val_m["val_itemid_loss"],
            "pretrain/val_value_loss"   : val_m["val_value_loss"],
            "pretrain/lr"               : lr,
            "epoch"                     : epoch,
        }
        if USE_VICREG:
            wlog_dict["pretrain/vicreg_loss"] = train_m["vicreg_loss"]
        _wlog(wlog_dict, step=epoch)

        save_checkpoint(model, optimizer, epoch, val_m["val_loss"],
                        CKPT_DIR / f"epoch_{epoch:03d}.pt")

        # Best checkpoint and early stopping based on VALIDATION loss
        if val_m["val_loss"] < best_loss:
            best_loss = val_m["val_loss"]
            save_checkpoint(model, optimizer, epoch, val_m["val_loss"],
                            CKPT_DIR / "best.pt")
            if USE_WANDB:
                import wandb
                wandb.run.summary["best_pretrain_val_loss"]  = best_loss
                wandb.run.summary["best_pretrain_val_epoch"] = epoch
                artifact = wandb.Artifact("pretrain-best", type="model")
                artifact.add_file(str(CKPT_DIR / "best.pt"))
                wandb.log_artifact(artifact)

        if stopper.step(val_m["val_loss"]):
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    print(f"\nPretraining complete. Best val loss: {best_loss:.4f}")
    print(f"Best checkpoint: {CKPT_DIR / 'best.pt'}")

    if USE_WANDB:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
