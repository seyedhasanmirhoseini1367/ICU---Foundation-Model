"""
training/pretrain.py

Phase 1: Masked Event Modeling (MEM) pretraining.

Each training step:
    1. Load a clean batch of ICU stay sequences.
    2. Randomly mask 15% of real events (apply_random_mask).
    3. Forward pass → encoder + PretrainHead.
    4. Loss = λ·CE(itemid) + (1-λ)·MSE(value)   [only at masked positions]
    5. Backprop + AdamW step + gradient clipping.

Run:
    python training/pretrain.py

Checkpoints are saved to:
    checkpoints/pretrain/epoch_NNN.pt   — every epoch
    checkpoints/pretrain/best.pt        — best loss so far

wandb:
    Set env var USE_WANDB=1 to enable live logging.
    On Kaggle this is set automatically by kaggle_run.py.
"""

import json
import os
import torch
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.config          import ModelConfig
from model.model           import ICUFoundationModel
from dataloader.dataloader import make_dataloader
from training.utils        import (
    AverageMeter, EarlyStopping,
    save_checkpoint, apply_random_mask,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "dataloader" / "all_stays"
INDEX_PATH = ROOT / "dataloader" / "index.csv"
VOCAB_PATH = ROOT / "tokenizer" / "vocab.json"
NORM_PATH  = ROOT / "tokenizer" / "norm_stats.json"
CKPT_DIR   = ROOT / "checkpoints" / "pretrain"

# ── GPU capability gate ───────────────────────────────────────────────────────
def _cuda_ok() -> bool:
    """True only when the GPU supports PyTorch 2.x (CUDA capability >= sm_70)."""
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

# ── wandb ─────────────────────────────────────────────────────────────────────
USE_WANDB      = os.getenv("USE_WANDB", "0") == "1"
WANDB_ENTITY   = "seyedhasan-mirhoseini1367-tampere-university"
WANDB_PROJECT  = "MIMIC-IV-ICU"


def _wlog(metrics: dict, step: int):
    if USE_WANDB:
        import wandb
        wandb.log(metrics, step=step)


# ── Training logic ────────────────────────────────────────────────────────────

def train_one_epoch(
    model     : ICUFoundationModel,
    loader    : torch.utils.data.DataLoader,
    optimizer : torch.optim.Optimizer,
    config    : ModelConfig,
    device    : torch.device,
) -> dict:
    """One full pass over the dataset. Returns average losses."""
    model.train()

    total_meter  = AverageMeter("loss")
    itemid_meter = AverageMeter("itemid_loss")
    value_meter  = AverageMeter("value_loss")

    for step, (inputs, _) in enumerate(loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}

        masked_inputs, itemid_labels, value_labels = apply_random_mask(inputs, config)
        itemid_labels = itemid_labels.to(device)
        value_labels  = value_labels.to(device)

        out = model.forward(
            mode          = "pretrain",
            itemid        = masked_inputs["itemid"],
            source        = masked_inputs["source"],
            delta_hours   = masked_inputs["delta_hours"],
            value         = masked_inputs["value"],
            padding_mask  = masked_inputs["padding_mask"],
            masked_labels = itemid_labels,
            value_labels  = value_labels,
        )

        optimizer.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        n = inputs["itemid"].size(0)
        total_meter.update(out["loss"].item(),         n)
        itemid_meter.update(out["itemid_loss"].item(), n)
        value_meter.update(out["value_loss"].item(),   n)

        if (step + 1) % max(1, len(loader) // 10) == 0:
            print(f"    step {step+1:>4}/{len(loader)}"
                  f"  loss={out['loss'].item():.4f}"
                  f"  itemid={out['itemid_loss'].item():.4f}"
                  f"  value={out['value_loss'].item():.4f}")

    return {
        "loss"       : total_meter.avg,
        "itemid_loss": itemid_meter.avg,
        "value_loss" : value_meter.avg,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if _CUDA_OK else "cpu")
    print(f"Device      : {device}")
    if _CUDA_OK:
        print(f"GPU         : {torch.cuda.get_device_name(0)}")

    vocab  = json.loads(VOCAB_PATH.read_text())
    config = ModelConfig(vocab_size=len(vocab))
    print(f"Vocab size  : {config.vocab_size} tokens")

    loader = make_dataloader(
        index_path      = INDEX_PATH,
        data_dir        = DATA_DIR,
        vocab_path      = VOCAB_PATH,
        norm_stats_path = NORM_PATH,
        max_len         = config.max_len,
        batch_size      = BATCH_SIZE,
        shuffle         = True,
    )
    n_stays = len(loader.dataset)
    print(f"Stays       : {n_stays}")
    print(f"Batch size  : {BATCH_SIZE}  ({len(loader)} batches/epoch)")

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
            name    = f"pretrain-{n_stays}stays",
            config  = {
                "phase"        : "pretrain",
                "n_stays"      : n_stays,
                "vocab_size"   : config.vocab_size,
                "d_model"      : config.d_model,
                "n_layers"     : config.n_layers,
                "n_heads"      : config.n_heads,
                "d_ff"         : config.d_ff,
                "max_len"      : config.max_len,
                "mask_prob"    : config.mask_prob,
                "batch_size"   : BATCH_SIZE,
                "lr"           : LEARNING_RATE,
                "weight_decay" : WEIGHT_DECAY,
                "patience"     : PATIENCE,
                "max_epochs"   : MAX_EPOCHS,
                "device"       : str(device),
            },
        )

    print(f"\n── Pretraining (MEM) for up to {MAX_EPOCHS} epochs ──\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        metrics = train_one_epoch(model, loader, optimizer, config, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:02d}/{MAX_EPOCHS}"
              f"  loss={metrics['loss']:.4f}"
              f"  itemid={metrics['itemid_loss']:.4f}"
              f"  value={metrics['value_loss']:.4f}"
              f"  lr={lr:.2e}")

        _wlog({
            "pretrain/loss"       : metrics["loss"],
            "pretrain/itemid_loss": metrics["itemid_loss"],
            "pretrain/value_loss" : metrics["value_loss"],
            "pretrain/lr"         : lr,
            "epoch"               : epoch,
        }, step=epoch)

        save_checkpoint(model, optimizer, epoch, metrics["loss"],
                        CKPT_DIR / f"epoch_{epoch:03d}.pt")

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            save_checkpoint(model, optimizer, epoch, metrics["loss"],
                            CKPT_DIR / "best.pt")
            if USE_WANDB:
                import wandb
                wandb.run.summary["best_pretrain_loss"]  = best_loss
                wandb.run.summary["best_pretrain_epoch"] = epoch
                artifact = wandb.Artifact("pretrain-best", type="model")
                artifact.add_file(str(CKPT_DIR / "best.pt"))
                wandb.log_artifact(artifact)

        if stopper.step(metrics["loss"]):
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    print(f"\nPretraining complete. Best loss: {best_loss:.4f}")
    print(f"Best checkpoint: {CKPT_DIR / 'best.pt'}")

    if USE_WANDB:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
