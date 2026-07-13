"""
training/pretrain_ar.py

Branch B pretraining: next-event prediction via causal autoregressive LM.

Each forward pass:
  1. Feed the full sequence through ICUAutoregressiveModel with a causal mask.
  2. Slice predictions from positions 0…L-2 and targets from positions 1…L-1.
  3. Compute weighted CE loss for itemid, value_bin, and delta_bin, masking
     positions where either the current or the next event is padding.

Causal shift alignment (critical — see model/autoregressive.py):
    itemid_tgt  = itemid[:, 1:]       ← next token id
    value_tgt   = value_bins[:, 1:]   ← next token value bin
    delta_tgt   = delta_bins[:, :-1]  ← gap FROM pos t TO t+1
    logits      = model(inputs)[:, :-1]
    valid       = padding_mask[:, :-1] & padding_mask[:, 1:]

Usage (Kaggle / command-line):
    python training/pretrain_ar.py \
        --index    dataloader/index.csv \
        --data_dir dataloader/all_stays \
        --vocab    tokenizer/vocab.json \
        --norm     tokenizer/norm_stats.json \
        --bins     tokenizer/bin_edges.json \
        --time_bins tokenizer/time_bin_edges.json \
        --epochs 10 --batch 32 --lr 3e-4 --patience 3 \
        --out checkpoints/ar

wandb API key: read from /kaggle/input/wandb-config/key.txt (never hardcoded).
"""

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model.config          import ModelConfig
from model.autoregressive  import ICUAutoregressiveModel
from dataloader.dataloader import make_dataloader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ── Loss helpers ──────────────────────────────────────────────────────────────

def _ar_loss(
    item_logits : torch.Tensor,   # [B, L-1, vocab_size]
    val_logits  : torch.Tensor,   # [B, L-1, n_value_bins]
    dt_logits   : torch.Tensor,   # [B, L-1, n_time_bins]
    itemid_tgt  : torch.Tensor,   # [B, L-1]
    value_tgt   : torch.Tensor,   # [B, L-1]
    delta_tgt   : torch.Tensor,   # [B, L-1]
    valid       : torch.Tensor,   # [B, L-1]  True=compute loss
    config      : ModelConfig,
    criterion   : nn.CrossEntropyLoss,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total_loss, loss_id, loss_val, loss_dt)."""
    itemid_tgt = itemid_tgt.masked_fill(~valid, -1)
    value_tgt  = value_tgt.masked_fill(~valid, -1)
    delta_tgt  = delta_tgt.masked_fill(~valid, -1)

    B, Lm1 = valid.shape
    loss_id  = criterion(item_logits.reshape(B * Lm1, -1), itemid_tgt.reshape(-1))
    loss_val = criterion(val_logits.reshape(B * Lm1, -1),  value_tgt.reshape(-1))
    loss_dt  = criterion(dt_logits.reshape(B * Lm1, -1),   delta_tgt.reshape(-1))

    w_sum  = config.ar_itemid_weight + config.ar_value_weight + config.ar_delta_weight
    total  = (
        config.ar_itemid_weight * loss_id
        + config.ar_value_weight * loss_val
        + config.ar_delta_weight * loss_dt
    ) / w_sum

    return total, loss_id, loss_val, loss_dt


# ── Training / evaluation ─────────────────────────────────────────────────────

def train_one_epoch(
    model      : ICUAutoregressiveModel,
    loader,
    optimizer  : torch.optim.Optimizer,
    device     : torch.device,
    config     : ModelConfig,
    step_offset: int  = 0,
    use_wandb  : bool = False,
) -> tuple[float, int]:
    """Returns (mean_loss, n_steps)."""
    model.train()
    criterion  = nn.CrossEntropyLoss(ignore_index=-1)
    total_loss = 0.0
    n_steps    = 0

    for batch_idx, (inputs, _) in enumerate(tqdm(loader, desc="AR-pretrain")):
        itemid       = inputs["itemid"].to(device)
        source       = inputs["source"].to(device)
        delta_hours  = inputs["delta_hours"].to(device)
        value        = inputs["value"].to(device)
        padding_mask = inputs["padding_mask"].to(device)
        value_bins   = inputs["value_bins"].to(device)
        # delta_bins may be absent if time_bin_edges_path was not provided
        delta_bins   = inputs.get("delta_bins", torch.zeros_like(itemid)).to(device)

        item_logits, val_logits, dt_logits = model(
            itemid, source, delta_hours, value, padding_mask
        )

        # Causal shift
        item_logits  = item_logits[:, :-1]
        val_logits   = val_logits[:, :-1]
        dt_logits    = dt_logits[:, :-1]
        itemid_tgt   = itemid[:, 1:]
        value_tgt    = value_bins[:, 1:]
        delta_tgt    = delta_bins[:, :-1]
        valid        = padding_mask[:, :-1] & padding_mask[:, 1:]

        loss, loss_id, loss_val, loss_dt = _ar_loss(
            item_logits, val_logits, dt_logits,
            itemid_tgt, value_tgt, delta_tgt,
            valid, config, criterion,
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_steps    += 1

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "ar/train_loss"     : loss.item(),
                "ar/loss_itemid"    : loss_id.item(),
                "ar/loss_value_bin" : loss_val.item(),
                "ar/loss_delta_bin" : loss_dt.item(),
                "step"              : step_offset + batch_idx,
            })

    return total_loss / max(n_steps, 1), n_steps


@torch.no_grad()
def evaluate_pretrain_ar(
    model  : ICUAutoregressiveModel,
    loader,
    device : torch.device,
    config : ModelConfig,
) -> float:
    """Returns mean validation loss."""
    model.eval()
    criterion  = nn.CrossEntropyLoss(ignore_index=-1)
    total_loss = 0.0
    n_steps    = 0

    for inputs, _ in loader:
        itemid       = inputs["itemid"].to(device)
        source       = inputs["source"].to(device)
        delta_hours  = inputs["delta_hours"].to(device)
        value        = inputs["value"].to(device)
        padding_mask = inputs["padding_mask"].to(device)
        value_bins   = inputs["value_bins"].to(device)
        delta_bins   = inputs.get("delta_bins", torch.zeros_like(itemid)).to(device)

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

    return total_loss / max(n_steps, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AR pretraining (Branch B)")
    parser.add_argument("--index",     required=True)
    parser.add_argument("--data_dir",  required=True)
    parser.add_argument("--vocab",     required=True)
    parser.add_argument("--norm",      required=True)
    parser.add_argument("--bins",      required=True)
    parser.add_argument("--time_bins", default=None,
                        help="Path to time_bin_edges.json (optional but recommended)")
    parser.add_argument("--out",       default="checkpoints/ar")
    parser.add_argument("--epochs",    type=int,   default=10)
    parser.add_argument("--batch",     type=int,   default=32)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--patience",  type=int,   default=3,
                        help="Early-stopping patience (val loss)")
    parser.add_argument("--max_len",   type=int,   default=512)
    parser.add_argument("--workers",   type=int,   default=0)
    parser.add_argument("--val_frac",  type=float, default=0.1,
                        help="Fraction of patients held out for validation")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--wandb_project", default="icu-ar-pretrain")
    parser.add_argument("--resume", default=None,
                        help="Path to ar_epochXXX.pt checkpoint to resume from. "
                             "--epochs is then the TOTAL epoch count (including already-done ones).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── wandb (optional) ─────────────────────────────────────────────────────
    use_wandb = False
    key_path  = Path("/kaggle/input/wandb-config/key.txt")
    if WANDB_AVAILABLE and key_path.exists():
        os.environ["WANDB_API_KEY"] = key_path.read_text().strip()
        wandb.init(project=args.wandb_project, config=vars(args))
        use_wandb = True

    # ── Vocab → config ────────────────────────────────────────────────────────
    with open(args.vocab) as f:
        vocab = json.load(f)
    config = ModelConfig(vocab_size=len(vocab), max_len=args.max_len)

    # ── Patient-level train/val split ─────────────────────────────────────────
    import pandas as pd, numpy as np
    index_df = pd.read_csv(args.index)
    patients = index_df["subject_id"].unique() if "subject_id" in index_df.columns \
               else np.arange(len(index_df))
    rng      = np.random.default_rng(args.seed)
    rng.shuffle(patients)
    n_val    = max(1, int(len(patients) * args.val_frac))
    val_pats = set(patients[:n_val])
    trn_pats = set(patients[n_val:])

    if "subject_id" in index_df.columns:
        trn_idx = index_df[index_df["subject_id"].isin(trn_pats)]
        val_idx = index_df[index_df["subject_id"].isin(val_pats)]
    else:
        n_val_rows = int(len(index_df) * args.val_frac)
        val_idx    = index_df.iloc[:n_val_rows]
        trn_idx    = index_df.iloc[n_val_rows:]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trn_idx_path = out_dir / "train_index.csv"
    val_idx_path = out_dir / "val_index.csv"
    trn_idx.to_csv(trn_idx_path, index=False)
    val_idx.to_csv(val_idx_path, index=False)

    common_dl_kwargs = dict(
        data_dir            = args.data_dir,
        vocab_path          = args.vocab,
        norm_stats_path     = args.norm,
        bin_edges_path      = args.bins,
        time_bin_edges_path = args.time_bins,
        max_len             = args.max_len,
        batch_size          = args.batch,
        num_workers         = args.workers,
    )
    train_loader = make_dataloader(
        index_path  = trn_idx_path,
        shuffle     = True,
        window_mode = "random",
        **common_dl_kwargs,
    )
    val_loader = make_dataloader(
        index_path  = val_idx_path,
        shuffle     = False,
        window_mode = "last",
        **common_dl_kwargs,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ICUAutoregressiveModel(config).to(device)
    print(f"ICUAutoregressiveModel: {model.count_parameters():,} parameters")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    best_val_loss  = float("inf")
    patience_count = 0
    step_offset    = 0
    start_epoch    = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch    = ckpt["epoch"] + 1
        best_val_loss  = ckpt["val_loss"]
        # Fast-forward cosine scheduler to the correct position
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        print(f"Resumed  : epoch {ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}  "
              f"→ continuing from epoch {start_epoch} to {args.epochs}")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, n_steps = train_one_epoch(
            model, train_loader, optimizer, device, config,
            step_offset=step_offset, use_wandb=use_wandb,
        )
        step_offset += n_steps
        scheduler.step()

        val_loss = evaluate_pretrain_ar(model, val_loader, device, config)
        print(f"Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({"ar/val_loss": val_loss, "epoch": epoch})

        # Checkpoint every epoch
        ckpt_path = out_dir / f"ar_epoch{epoch:03d}.pt"
        torch.save({
            "epoch"     : epoch,
            "model"     : model.state_dict(),
            "optimizer" : optimizer.state_dict(),
            "val_loss"  : val_loss,
            "config"    : vars(config),
        }, ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), out_dir / "ar_best.pt")
            print(f"  → new best val loss {best_val_loss:.4f} (saved ar_best.pt)")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping after {epoch} epochs (patience={args.patience})")
                break

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    print(f"\nBest val loss : {best_val_loss:.4f}")
    print(f"Best ckpt     : {out_dir / 'ar_best.pt'}")


if __name__ == "__main__":
    main()
