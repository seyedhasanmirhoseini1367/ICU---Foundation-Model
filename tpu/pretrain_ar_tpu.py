"""
tpu/pretrain_ar_tpu.py

Branch B AR pretraining — TPU (torch_xla) version.
Runs across all 8 TPU v3-8 cores via xmp.spawn().

Key differences from training/pretrain_ar.py:
  device     : xm.xla_device() per core, not "cuda"
  optimizer  : xm.optimizer_step() syncs gradients across all 8 cores
  lazy exec  : xm.mark_step() flushes XLA graph each step
  checkpoints: xm.save() from master core (rank 0) only
  val loss   : xm.mesh_reduce() averages across cores
  data       : train index pre-sharded — each core sees 1/8 of patients
  batch size : per-core (effective = batch * 8)
  no early-stop: all cores must run same number of epochs (use --epochs)

Usage:
    python tpu/pretrain_ar_tpu.py \
        --index    checkpoints/ar/train_index.csv \
        --data_dir dataloader/all_stays \
        --vocab    tokenizer/vocab.json \
        --norm     tokenizer/norm_stats.json \
        --bins     tokenizer/bin_edges.json \
        --time_bins tokenizer/time_bin_edges.json \
        --epochs 20 --batch 64 --lr 3e-4 \
        --out checkpoints/ar
"""

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

from model.config          import ModelConfig
from model.autoregressive  import ICUAutoregressiveModel
from dataloader.dataloader import make_dataloader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

NUM_CORES = 8  # TPU v3-8


# ── Loss (identical to GPU version) ───────────────────────────────────────────

def _ar_loss(item_logits, val_logits, dt_logits,
             itemid_tgt, value_tgt, delta_tgt, valid, config, criterion):
    itemid_tgt = itemid_tgt.masked_fill(~valid, -1)
    value_tgt  = value_tgt.masked_fill(~valid, -1)
    delta_tgt  = delta_tgt.masked_fill(~valid, -1)
    B, Lm1 = valid.shape
    loss_id  = criterion(item_logits.reshape(B * Lm1, -1), itemid_tgt.reshape(-1))
    loss_val = criterion(val_logits.reshape(B * Lm1, -1),  value_tgt.reshape(-1))
    loss_dt  = criterion(dt_logits.reshape(B * Lm1, -1),   delta_tgt.reshape(-1))
    w_sum = config.ar_itemid_weight + config.ar_value_weight + config.ar_delta_weight
    total = (config.ar_itemid_weight * loss_id
             + config.ar_value_weight * loss_val
             + config.ar_delta_weight * loss_dt) / w_sum
    return total, loss_id, loss_val, loss_dt


# ── Per-epoch train / eval ─────────────────────────────────────────────────────

def _train_epoch(model, device_loader, optimizer, device, config,
                 step_offset, use_wandb):
    model.train()
    criterion  = nn.CrossEntropyLoss(ignore_index=-1)
    total_loss = 0.0
    n_steps    = 0

    for batch_idx, (inputs, _) in enumerate(device_loader):
        itemid       = inputs["itemid"]
        source       = inputs["source"]
        delta_hours  = inputs["delta_hours"]
        value        = inputs["value"]
        padding_mask = inputs["padding_mask"]
        value_bins   = inputs["value_bins"]
        delta_bins   = inputs.get("delta_bins", torch.zeros_like(itemid))

        item_logits, val_logits, dt_logits = model(
            itemid, source, delta_hours, value, padding_mask
        )
        item_logits = item_logits[:, :-1]
        val_logits  = val_logits[:, :-1]
        dt_logits   = dt_logits[:, :-1]
        itemid_tgt  = itemid[:, 1:]
        value_tgt   = value_bins[:, 1:]
        delta_tgt   = delta_bins[:, :-1]
        valid       = padding_mask[:, :-1] & padding_mask[:, 1:]

        loss, loss_id, loss_val, loss_dt = _ar_loss(
            item_logits, val_logits, dt_logits,
            itemid_tgt, value_tgt, delta_tgt,
            valid, config, criterion,
        )

        optimizer.zero_grad()
        loss.backward()
        xm.optimizer_step(optimizer)  # allreduce gradients across 8 cores
        xm.mark_step()               # flush XLA lazy graph

        total_loss += loss.item()
        n_steps    += 1

        if use_wandb and WANDB_AVAILABLE and xm.is_master_ordinal():
            wandb.log({
                "ar/train_loss"     : loss.item(),
                "ar/loss_itemid"    : loss_id.item(),
                "ar/loss_value_bin" : loss_val.item(),
                "ar/loss_delta_bin" : loss_dt.item(),
                "step"              : step_offset + batch_idx,
            })

    return total_loss / max(n_steps, 1), n_steps


@torch.no_grad()
def _eval_epoch(model, device_loader, device, config):
    model.eval()
    criterion  = nn.CrossEntropyLoss(ignore_index=-1)
    total_loss = 0.0
    n_steps    = 0

    for inputs, _ in device_loader:
        itemid       = inputs["itemid"]
        source       = inputs["source"]
        delta_hours  = inputs["delta_hours"]
        value        = inputs["value"]
        padding_mask = inputs["padding_mask"]
        value_bins   = inputs["value_bins"]
        delta_bins   = inputs.get("delta_bins", torch.zeros_like(itemid))

        item_logits, val_logits, dt_logits = model(
            itemid, source, delta_hours, value, padding_mask
        )
        item_logits = item_logits[:, :-1]
        val_logits  = val_logits[:, :-1]
        dt_logits   = dt_logits[:, :-1]
        itemid_tgt  = itemid[:, 1:]
        value_tgt   = value_bins[:, 1:]
        delta_tgt   = delta_bins[:, :-1]
        valid       = padding_mask[:, :-1] & padding_mask[:, 1:]

        loss, _, _, _ = _ar_loss(
            item_logits, val_logits, dt_logits,
            itemid_tgt, value_tgt, delta_tgt,
            valid, config, criterion,
        )
        total_loss += loss.item()
        n_steps    += 1
        xm.mark_step()

    local_avg = total_loss / max(n_steps, 1)
    # Average val_loss across all 8 cores
    return xm.mesh_reduce("val_loss", local_avg, lambda vals: sum(vals) / len(vals))


# ── Per-core worker ────────────────────────────────────────────────────────────

def _worker(rank, args):
    device = xm.xla_device()
    xm.master_print(f"All cores ready — device: {device}")

    # wandb: master core only
    use_wandb = False
    if xm.is_master_ordinal() and WANDB_AVAILABLE:
        key_path = Path("/kaggle/input/wandb-config/key.txt")
        if key_path.exists():
            os.environ["WANDB_API_KEY"] = key_path.read_text().strip()
            wandb.init(project=args.wandb_project, config=vars(args))
            use_wandb = True

    with open(args.vocab) as f:
        vocab = json.load(f)
    config = ModelConfig(vocab_size=len(vocab), max_len=args.max_len)
    model  = ICUAutoregressiveModel(config).to(device)
    xm.master_print(f"ICUAutoregressiveModel: {model.count_parameters():,} parameters")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Resume
    start_epoch   = 1
    best_val_loss = float("inf")
    resume_epoch  = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        resume_epoch  = ckpt["epoch"]
        start_epoch   = resume_epoch + 1
        best_val_loss = ckpt["val_loss"]
        xm.master_print(f"Resumed  : epoch {resume_epoch}  val_loss={ckpt['val_loss']:.4f}"
                        f"  → epoch {start_epoch} to {args.epochs}")

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs,
                                  last_epoch=start_epoch - 2)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Each core uses its own data shard (created in main() before spawn)
    common_kw = dict(
        data_dir            = args.data_dir,
        vocab_path          = args.vocab,
        norm_stats_path     = args.norm,
        bin_edges_path      = args.bins,
        time_bin_edges_path = args.time_bins,
        max_len             = args.max_len,
        batch_size          = args.batch,
        num_workers         = 0,  # must be 0 on TPU (fork issues)
    )
    train_loader = make_dataloader(
        index_path  = out_dir / f"train_shard_{rank:02d}.csv",
        shuffle     = True,
        window_mode = "random",
        **common_kw,
    )
    val_loader = make_dataloader(
        index_path  = out_dir / "val_index.csv",
        shuffle     = False,
        window_mode = "last",
        **common_kw,
    )

    # MpDeviceLoader moves batches to the TPU device asynchronously
    train_dl = pl.MpDeviceLoader(train_loader, device)
    val_dl   = pl.MpDeviceLoader(val_loader,   device)

    step_offset = 0

    for epoch in range(start_epoch, args.epochs + 1):
        xm.master_print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, n_steps = _train_epoch(
            model, train_dl, optimizer, device, config, step_offset, use_wandb,
        )
        step_offset += n_steps
        scheduler.step()

        val_loss = _eval_epoch(model, val_dl, device, config)
        xm.master_print(f"Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if use_wandb and WANDB_AVAILABLE and xm.is_master_ordinal():
            wandb.log({"ar/val_loss": val_loss, "epoch": epoch})

        # Save from master core only
        if xm.is_master_ordinal():
            xm.save({
                "epoch"     : epoch,
                "model"     : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "val_loss"  : val_loss,
                "config"    : vars(config),
            }, out_dir / f"ar_epoch{epoch:03d}.pt")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                xm.save(model.state_dict(), out_dir / "ar_best.pt")
                xm.master_print(f"  → new best {best_val_loss:.4f}  (saved ar_best.pt)")

        # All cores sync before next epoch
        xm.rendezvous(f"epoch_{epoch}_done")

    if use_wandb and WANDB_AVAILABLE and xm.is_master_ordinal():
        wandb.finish()

    xm.master_print(f"\nBest val loss : {best_val_loss:.4f}")
    xm.master_print(f"Best ckpt     : {out_dir / 'ar_best.pt'}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AR pretraining — TPU (torch_xla)")
    parser.add_argument("--index",     required=True)
    parser.add_argument("--data_dir",  required=True)
    parser.add_argument("--vocab",     required=True)
    parser.add_argument("--norm",      required=True)
    parser.add_argument("--bins",      required=True)
    parser.add_argument("--time_bins", default=None)
    parser.add_argument("--out",       default="checkpoints/ar")
    parser.add_argument("--epochs",    type=int,   default=10)
    parser.add_argument("--batch",     type=int,   default=64,
                        help="Per-core batch size. Effective = batch × 8 cores.")
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--max_len",   type=int,   default=512)
    parser.add_argument("--val_frac",  type=float, default=0.1)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--wandb_project", default="MIMIC-IV-ICU-AR-TPU")
    parser.add_argument("--resume",    default=None,
                        help="Path to ar_epochXXX.pt. --epochs = total (not additional).")
    args = parser.parse_args()

    import pandas as pd
    import numpy as np

    torch.manual_seed(args.seed)

    # Patient-level train/val split (done once before spawning cores)
    index_df = pd.read_csv(args.index)
    patients = index_df["subject_id"].unique() if "subject_id" in index_df.columns \
               else np.arange(len(index_df))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(patients)
    n_val    = max(1, int(len(patients) * args.val_frac))
    val_pats = set(patients[:n_val])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "subject_id" in index_df.columns:
        trn_idx = index_df[~index_df["subject_id"].isin(val_pats)]
        val_idx = index_df[index_df["subject_id"].isin(val_pats)]
    else:
        n_val_rows = int(len(index_df) * args.val_frac)
        val_idx = index_df.iloc[:n_val_rows]
        trn_idx = index_df.iloc[n_val_rows:]

    val_idx.to_csv(out_dir / "val_index.csv", index=False)

    # Shard train index across NUM_CORES — each core trains on its own 1/8
    trn_idx = trn_idx.reset_index(drop=True)
    for i in range(NUM_CORES):
        shard = trn_idx.iloc[i::NUM_CORES]
        shard.to_csv(out_dir / f"train_shard_{i:02d}.csv", index=False)

    print(f"Train : {len(trn_idx):,} stays → {NUM_CORES} shards of ~{len(trn_idx)//NUM_CORES:,}")
    print(f"Val   : {len(val_idx):,} stays")
    print(f"Effective batch: {args.batch} × {NUM_CORES} cores = {args.batch * NUM_CORES}")

    xmp.spawn(_worker, args=(args,), nprocs=NUM_CORES, start_method="fork")


if __name__ == "__main__":
    main()
