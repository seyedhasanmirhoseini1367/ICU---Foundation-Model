"""
training/finetune.py

Phase 2: Supervised fine-tuning on mortality prediction and LOS regression.

Train / validation split is done by PATIENT (subject_id) so no patient
appears in both sets — prevents data leakage when one patient has multiple stays.
Default: 80% train / 20% val.

Two-stage strategy:
    Stage 2a — Epoch 1:  encoder frozen, heads only (lr=1e-3)
    Stage 2b — Epoch 2+: full model, lower lr (lr=1e-5)

Run:
    python training/finetune.py

Checkpoints saved to:
    checkpoints/finetune/epoch_NNN.pt
    checkpoints/finetune/best.pt

wandb:
    Set USE_WANDB=1 to enable. Kaggle script sets this automatically.
"""

import json
import os
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW

import pandas as pd

from model.config          import ModelConfig
from model.model           import ICUFoundationModel
from model.heads           import VITAL_NAMES, N_VITALS
from dataloader.dataloader import make_dataloader
from training.utils        import (
    AverageMeter, EarlyStopping,
    save_checkpoint, load_checkpoint,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT / "dataloader" / "all_stays"
INDEX_PATH    = ROOT / "dataloader" / "index.csv"
VOCAB_PATH    = ROOT / "tokenizer" / "vocab.json"
NORM_PATH     = ROOT / "tokenizer" / "norm_stats.json"
PRETRAIN_CKPT = ROOT / "checkpoints" / "pretrain" / "best.pt"
CKPT_DIR      = ROOT / "checkpoints" / "finetune"

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
MAX_EPOCHS    = 30
LR_HEADS_ONLY = 1e-3
LR_FULL_MODEL = 1e-5
WEIGHT_DECAY  = 1e-2
PATIENCE      = 5
VAL_FRACTION  = 0.20
RANDOM_SEED   = 42

# ── wandb ─────────────────────────────────────────────────────────────────────
USE_WANDB     = os.getenv("USE_WANDB", "0") == "1"
WANDB_ENTITY  = "seyedhasan-mirhoseini1367-tampere-university"
WANDB_PROJECT = "MIMIC-IV-ICU"


def _wlog(metrics: dict, step: int):
    if USE_WANDB:
        import wandb
        wandb.log(metrics, step=step)


# ── Train / val split by patient ──────────────────────────────────────────────

def patient_split(index_path: Path, val_fraction: float, seed: int):
    """
    Split index.csv into train/val by subject_id.

    Splitting by patient (not by stay) ensures a patient with multiple
    ICU stays never appears in both sets.
    """
    index    = pd.read_csv(index_path)
    subjects = index["subject_id"].unique()

    rng      = np.random.default_rng(seed)
    n_val    = max(1, int(len(subjects) * val_fraction))
    val_subj = set(rng.choice(subjects, size=n_val, replace=False))

    train_df = index[~index["subject_id"].isin(val_subj)].reset_index(drop=True)
    val_df   = index[ index["subject_id"].isin(val_subj)].reset_index(drop=True)

    # Persist splits alongside index.csv for reproducibility
    out_dir = index_path.parent
    train_df.to_csv(out_dir / "train_index.csv", index=False)
    val_df.to_csv(  out_dir / "val_index.csv",   index=False)

    return train_df, val_df


# ── Epoch helpers ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device) -> dict:
    model.train()
    total_m = AverageMeter("loss")
    mort_m  = AverageMeter("mortality_loss")
    los_m   = AverageMeter("los_loss")
    vital_m = AverageMeter("vital_loss")

    for inputs, labels in loader:
        inputs       = {k: v.to(device) for k, v in inputs.items()}
        mort_labels  = labels["hospital_expire_flag"].float().to(device)
        los_labels   = labels["los"].float().to(device)
        vital_labels = labels["vital_targets"].to(device)   # [B, N_VITALS]
        vital_mask   = labels["vital_mask"].to(device)      # [B, N_VITALS]

        out = model.forward(
            mode             = "finetune",
            itemid           = inputs["itemid"],
            source           = inputs["source"],
            delta_hours      = inputs["delta_hours"],
            value            = inputs["value"],
            padding_mask     = inputs["padding_mask"],
            mortality_labels = mort_labels,
            los_labels       = los_labels,
            vital_labels     = vital_labels,
            vital_mask       = vital_mask,
        )

        optimizer.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        n = inputs["itemid"].size(0)
        total_m.update(out["loss"].item(),          n)
        mort_m.update(out["mortality_loss"].item(), n)
        los_m.update(out["los_loss"].item(),        n)
        vital_m.update(out["vital_loss"].item(),    n)

    return {
        "loss"          : total_m.avg,
        "mortality_loss": mort_m.avg,
        "los_loss"      : los_m.avg,
        "vital_loss"    : vital_m.avg,
    }


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    loss_m      = AverageMeter("val_loss")
    all_labels  = []
    all_logits  = []
    # Per-vital: accumulate (pred, target) pairs for MAE
    vital_preds_all  = [[] for _ in range(N_VITALS)]
    vital_labels_all = [[] for _ in range(N_VITALS)]

    for inputs, labels in loader:
        inputs       = {k: v.to(device) for k, v in inputs.items()}
        mort_labels  = labels["hospital_expire_flag"].float().to(device)
        los_labels   = labels["los"].float().to(device)
        vital_labels = labels["vital_targets"].to(device)
        vital_mask   = labels["vital_mask"].to(device)

        out = model.forward(
            mode             = "finetune",
            itemid           = inputs["itemid"],
            source           = inputs["source"],
            delta_hours      = inputs["delta_hours"],
            value            = inputs["value"],
            padding_mask     = inputs["padding_mask"],
            mortality_labels = mort_labels,
            los_labels       = los_labels,
            vital_labels     = vital_labels,
            vital_mask       = vital_mask,
        )

        loss_m.update(out["loss"].item(), inputs["itemid"].size(0))
        all_labels.append(mort_labels.cpu())
        all_logits.append(out["mortality_logits"].cpu())

        # Collect per-vital predictions for MAE computation
        vp = out["vital_preds"].cpu()   # [B, N_VITALS]
        vl = vital_labels.cpu()         # [B, N_VITALS]
        vm = vital_mask.cpu()           # [B, N_VITALS]
        for i in range(N_VITALS):
            mask_i = vm[:, i]
            if mask_i.any():
                vital_preds_all[i].append(vp[mask_i, i])
                vital_labels_all[i].append(vl[mask_i, i])

    all_labels = torch.cat(all_labels).numpy()
    all_logits = torch.cat(all_logits).numpy()

    try:
        auroc = float(roc_auc_score(all_labels, all_logits))
    except ValueError:
        auroc = float("nan")

    # Per-vital MAE (in normalised space; lower = better)
    vital_mae = {}
    for i, name in enumerate(VITAL_NAMES):
        if vital_preds_all[i]:
            p = torch.cat(vital_preds_all[i])
            t = torch.cat(vital_labels_all[i])
            vital_mae[name] = float((p - t).abs().mean())
        else:
            vital_mae[name] = float("nan")

    return {"val_loss": loss_m.avg, "auroc": auroc, "vital_mae": vital_mae}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if _CUDA_OK else "cpu")
    print(f"Device      : {device}")
    if _CUDA_OK:
        print(f"GPU         : {torch.cuda.get_device_name(0)}")

    vocab  = json.loads(VOCAB_PATH.read_text())
    config = ModelConfig(vocab_size=len(vocab))

    # ── Split by patient ──────────────────────────────────────────────────────
    print(f"\nSplitting index by patient ({int(VAL_FRACTION*100)}% val) …")
    train_df, val_df = patient_split(INDEX_PATH, VAL_FRACTION, RANDOM_SEED)
    print(f"  Train: {len(train_df)} stays ({train_df['subject_id'].nunique()} patients)")
    print(f"  Val  : {len(val_df)}   stays ({val_df['subject_id'].nunique()} patients)")
    print(f"  Mortality — train: {train_df['hospital_expire_flag'].mean():.1%}"
          f"   val: {val_df['hospital_expire_flag'].mean():.1%}")

    def make_loader(df, shuffle):
        tmp = INDEX_PATH.parent / "_tmp_split.csv"
        df.to_csv(tmp, index=False)
        return make_dataloader(
            index_path      = tmp,
            data_dir        = DATA_DIR,
            vocab_path      = VOCAB_PATH,
            norm_stats_path = NORM_PATH,
            max_len         = config.max_len,
            batch_size      = BATCH_SIZE,
            shuffle         = shuffle,
        )

    train_loader = make_loader(train_df, shuffle=True)
    val_loader   = make_loader(val_df,   shuffle=False)
    print(f"\nBatch size  : {BATCH_SIZE}")
    print(f"Train batches: {len(train_loader)}   Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ICUFoundationModel(config).to(device)
    if PRETRAIN_CKPT.exists():
        load_checkpoint(PRETRAIN_CKPT, model, device=device)
        print(f"Loaded pretrain checkpoint: {PRETRAIN_CKPT}")
    else:
        print("⚠  No pretrain checkpoint — fine-tuning from random init.")

    # ── wandb init ────────────────────────────────────────────────────────────
    if USE_WANDB:
        import wandb
        wandb.init(
            entity  = WANDB_ENTITY,
            project = WANDB_PROJECT,
            name    = f"finetune-{len(train_df)}train-{len(val_df)}val",
            config  = {
                "phase"         : "finetune",
                "n_train"       : len(train_df),
                "n_val"         : len(val_df),
                "val_fraction"  : VAL_FRACTION,
                "vocab_size"    : config.vocab_size,
                "d_model"       : config.d_model,
                "n_layers"      : config.n_layers,
                "n_heads"       : config.n_heads,
                "batch_size"    : BATCH_SIZE,
                "lr_heads_only" : LR_HEADS_ONLY,
                "lr_full_model" : LR_FULL_MODEL,
                "weight_decay"  : WEIGHT_DECAY,
                "patience"      : PATIENCE,
                "max_epochs"    : MAX_EPOCHS,
                "device"        : str(device),
            },
        )

    # ── Stage 2a: frozen encoder, heads only ──────────────────────────────────
    print("\n── Stage 2a: heads only (encoder frozen) ──")
    model.freeze_encoder()
    head_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer   = AdamW(head_params, lr=LR_HEADS_ONLY, weight_decay=WEIGHT_DECAY)
    m = train_one_epoch(model, train_loader, optimizer, device)
    print(f"  Epoch 01  loss={m['loss']:.4f}  mort={m['mortality_loss']:.4f}"
          f"  los={m['los_loss']:.4f}  vital={m['vital_loss']:.4f}")
    _wlog({
        "finetune/train_loss"    : m["loss"],
        "finetune/mortality_loss": m["mortality_loss"],
        "finetune/los_loss"      : m["los_loss"],
        "finetune/vital_loss"    : m["vital_loss"],
        "epoch": 1,
    }, step=1)

    # ── Stage 2b: full model fine-tuning ──────────────────────────────────────
    print("\n── Stage 2b: full model fine-tuning ──")
    model.unfreeze_all()
    optimizer = AdamW(model.parameters(), lr=LR_FULL_MODEL, weight_decay=WEIGHT_DECAY)
    stopper   = EarlyStopping(patience=PATIENCE)
    best_loss = float("inf")

    for epoch in range(2, MAX_EPOCHS + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, device)
        eval_m  = evaluate(model, val_loader, device)

        mae_str = "  ".join(
            f"{n}={v:.3f}" for n, v in eval_m["vital_mae"].items()
            if not np.isnan(v)
        )
        print(f"  Epoch {epoch:02d}"
              f"  train={train_m['loss']:.4f}"
              f"  val={eval_m['val_loss']:.4f}"
              f"  AUROC={eval_m['auroc']:.3f}"
              f"  mort={train_m['mortality_loss']:.4f}"
              f"  los={train_m['los_loss']:.4f}"
              f"  vital={train_m['vital_loss']:.4f}")
        if mae_str:
            print(f"    vital MAE: {mae_str}")

        wlog_dict = {
            "finetune/train_loss"    : train_m["loss"],
            "finetune/val_loss"      : eval_m["val_loss"],
            "finetune/auroc"         : eval_m["auroc"],
            "finetune/mortality_loss": train_m["mortality_loss"],
            "finetune/los_loss"      : train_m["los_loss"],
            "finetune/vital_loss"    : train_m["vital_loss"],
            "epoch": epoch,
        }
        for name, mae in eval_m["vital_mae"].items():
            if not np.isnan(mae):
                wlog_dict[f"finetune/mae_{name}"] = mae
        _wlog(wlog_dict, step=epoch)

        save_checkpoint(model, optimizer, epoch, eval_m["val_loss"],
                        CKPT_DIR / f"epoch_{epoch:03d}.pt")

        if eval_m["val_loss"] < best_loss:
            best_loss = eval_m["val_loss"]
            save_checkpoint(model, optimizer, epoch, eval_m["val_loss"],
                            CKPT_DIR / "best.pt")
            if USE_WANDB:
                import wandb
                wandb.run.summary["best_val_loss"]  = best_loss
                wandb.run.summary["best_val_epoch"] = epoch
                wandb.run.summary["best_auroc"]     = eval_m["auroc"]
                artifact = wandb.Artifact("finetune-best", type="model")
                artifact.add_file(str(CKPT_DIR / "best.pt"))
                wandb.log_artifact(artifact)

        if stopper.step(eval_m["val_loss"]):
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    print(f"\nFine-tuning complete. Best val loss: {best_loss:.4f}")
    print(f"Best checkpoint: {CKPT_DIR / 'best.pt'}")

    if USE_WANDB:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
