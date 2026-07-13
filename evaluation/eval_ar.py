"""
evaluation/eval_ar.py

Zero-shot evaluation for ICUAutoregressiveModel (Branch B).

Metrics:
  1. Next-event top-1 accuracy (split by new-onset vs. repeat events)
      — new-onset : the model has NOT seen this itemid in the seed context
      — repeat    : the itemid already appeared in the seed context
      This split matters because repeat routine vitals trivially inflate scores.

  2. Zero-shot mortality AUROC + Brier score
      — mortality_prob() over rollout trajectories; compare with
        hospital_expire_flag label.

  3. Optional Branch A comparison table
      — If a Branch A checkpoint is provided, its supervised MortalityHead
        logit is included alongside the AR zero-shot score.

Usage:
    python evaluation/eval_ar.py \
        --ar_ckpt   checkpoints/ar/ar_best.pt \
        --index     dataloader/index.csv \
        --data_dir  dataloader/all_stays \
        --vocab     tokenizer/vocab.json \
        --norm      tokenizer/norm_stats.json \
        --bins      tokenizer/bin_edges.json \
        --time_bins tokenizer/time_bin_edges.json \
        [--mem_ckpt checkpoints/finetune/best.pt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader as _DataLoader

from model.config          import ModelConfig
from model.autoregressive  import ICUAutoregressiveModel
from dataloader.dataloader import ICUDataset
from inference.rollout     import rollout, mortality_prob


# ── Next-event accuracy (single forward pass, no rollout needed) ──────────────

def _iter_stays(dataset, max_stays: int):
    """
    Yield (inputs_dict, labels_dict) one stay at a time from either an
    ICUDataset (preferred) or a DataLoader (unbatched automatically).

    Passing an ICUDataset is preferred because it avoids the collation overhead
    and makes `max_stays` exact.  If a DataLoader is passed the function
    iterates batches and unbatches them; `max_stays` is still respected.
    """
    if isinstance(dataset, _DataLoader):
        count = 0
        for batch_inputs, batch_labels in dataset:
            B = batch_inputs["itemid"].size(0)
            for b in range(B):
                inp = {k: v[b] for k, v in batch_inputs.items()}
                lbl = {
                    k: (v[b] if isinstance(v, torch.Tensor) else
                        (v[b] if hasattr(v, "__getitem__") else v))
                    for k, v in batch_labels.items()
                }
                yield inp, lbl
                count += 1
                if count >= max_stays:
                    return
    else:
        n = min(max_stays, len(dataset))
        for idx in range(n):
            yield dataset[idx]


@torch.no_grad()
def eval_next_event_accuracy(
    model     : ICUAutoregressiveModel,
    dataset,  # ICUDataset  OR  torch.utils.data.DataLoader
    device    : torch.device,
    max_stays : int = 500,
) -> dict:
    """
    Computes top-1 itemid accuracy split by new-onset vs. repeat events.

    For each stay we do ONE forward pass (not a rollout) and evaluate
    predictions at all valid causal positions.

    new-onset : itemid at position t+1 has NOT appeared in positions 0..t
    repeat    : itemid at position t+1 HAS appeared in positions 0..t

    `dataset` accepts either an ICUDataset (indexed directly) or a DataLoader
    (batches are automatically unbatched).  ICUDataset is preferred.
    """
    model.eval()

    new_onset_correct = 0;  new_onset_total = 0
    repeat_correct    = 0;  repeat_total    = 0

    for inputs, _ in _iter_stays(dataset, max_stays):
        itemid       = inputs["itemid"].unsqueeze(0).to(device)      # [1, L]
        source       = inputs["source"].unsqueeze(0).to(device)
        delta_hours  = inputs["delta_hours"].unsqueeze(0).to(device)
        value        = inputs["value"].unsqueeze(0).to(device)
        padding_mask = inputs["padding_mask"].unsqueeze(0).to(device)

        item_logits, _, _ = model(itemid, source, delta_hours, value, padding_mask)
        # [1, L, vocab_size]  predictions from 0..L-2
        item_logits = item_logits[:, :-1]   # [1, L-1, V]
        pred        = item_logits.argmax(-1).squeeze(0)   # [L-1]

        pm  = padding_mask.squeeze(0)               # [L]
        iid = itemid.squeeze(0)                     # [L]
        valid = pm[:-1] & pm[1:]                    # [L-1]

        for t in range(len(valid)):
            if not valid[t]:
                continue
            tgt = iid[t + 1].item()
            # seen set = itemids at positions 0..t (the current context)
            seen = set(iid[: t + 1].tolist())

            if tgt in seen:
                repeat_total    += 1
                repeat_correct  += int(pred[t].item() == tgt)
            else:
                new_onset_total  += 1
                new_onset_correct += int(pred[t].item() == tgt)

    return {
        "new_onset_acc"   : new_onset_correct / max(new_onset_total, 1),
        "new_onset_total" : new_onset_total,
        "repeat_acc"      : repeat_correct / max(repeat_total, 1),
        "repeat_total"    : repeat_total,
    }


# ── Zero-shot mortality AUROC + Brier score ───────────────────────────────────

def eval_mortality_zero_shot(
    model          : ICUAutoregressiveModel,
    dataset        : ICUDataset,
    vocab_inv      : dict[int, int],
    time_bin_edges : list[float],
    device         : torch.device,
    death_itemids  : set[int],
    n_rollout      : int   = 50,
    horizon_hours  : float = 6.0,
    max_stays      : int   = 200,
) -> dict:
    """Compute zero-shot mortality AUROC and Brier score via rollout."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("[eval_ar] sklearn not available — skipping mortality AUROC")
        return {}

    scores = []
    labels = []

    n = min(max_stays, len(dataset))
    for idx in range(n):
        inputs, lab = dataset[idx]
        label = int(lab["hospital_expire_flag"])

        trajs = rollout(
            model, inputs, vocab_inv, time_bin_edges, device,
            n_samples=n_rollout, horizon_hours=horizon_hours,
        )
        score = mortality_prob(trajs, death_itemids)
        scores.append(score)
        labels.append(label)

    scores = np.array(scores)
    labels = np.array(labels)

    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"note": "all labels same class — AUROC undefined", "brier": float(np.mean((scores - labels)**2))}

    auroc = float(roc_auc_score(labels, scores))
    brier = float(np.mean((scores - labels) ** 2))
    return {"mortality_auroc": auroc, "mortality_brier": brier, "n_stays": n}


# ── Optional Branch A comparison ──────────────────────────────────────────────

@torch.no_grad()
def eval_branch_a_mortality(
    mem_ckpt_path : str | Path,
    dataset       : ICUDataset,
    device        : torch.device,
    max_stays     : int = 200,
) -> dict:
    """Run Branch A MortalityHead on the same stays for a fair comparison."""
    try:
        from sklearn.metrics import roc_auc_score
        from model.model import ICUFoundationModel
    except ImportError:
        return {"note": "Branch A model not importable — skipping"}

    ckpt   = torch.load(mem_ckpt_path, map_location=device)
    config = ModelConfig(**ckpt.get("config", {}))
    model  = ICUFoundationModel(config).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    scores = []
    labels = []
    n      = min(max_stays, len(dataset))

    for idx in range(n):
        inputs, lab = dataset[idx]
        itemid       = inputs["itemid"].unsqueeze(0).to(device)
        source       = inputs["source"].unsqueeze(0).to(device)
        delta_hours  = inputs["delta_hours"].unsqueeze(0).to(device)
        value        = inputs["value"].unsqueeze(0).to(device)
        padding_mask = inputs["padding_mask"].unsqueeze(0).to(device)

        logit = model(
            itemid, source, delta_hours, value, padding_mask, mode="finetune"
        )["mortality_logit"]
        prob = float(torch.sigmoid(logit).item())
        scores.append(prob)
        labels.append(int(lab["hospital_expire_flag"]))

    labels = np.array(labels)
    scores = np.array(scores)

    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"note": "all labels same class"}

    return {
        "mem_mortality_auroc": float(roc_auc_score(labels, scores)),
        "mem_mortality_brier": float(np.mean((scores - labels) ** 2)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Branch B zero-shot evaluation")
    parser.add_argument("--ar_ckpt",     required=True)
    parser.add_argument("--index",       required=True)
    parser.add_argument("--data_dir",    required=True)
    parser.add_argument("--vocab",       required=True)
    parser.add_argument("--norm",        required=True)
    parser.add_argument("--bins",        required=True)
    parser.add_argument("--time_bins",   required=True)
    parser.add_argument("--mem_ckpt",    default=None,
                        help="Optional Branch A checkpoint for comparison")
    parser.add_argument("--n_rollout",   type=int,   default=50)
    parser.add_argument("--horizon",     type=float, default=6.0)
    parser.add_argument("--max_stays",   type=int,   default=200)
    parser.add_argument("--max_next_ev", type=int,   default=500,
                        help="Stays used for next-event accuracy eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load vocab ────────────────────────────────────────────────────────────
    with open(args.vocab) as f:
        vocab = json.load(f)
    vocab_inv = {v: int(k) for k, v in vocab.items() if k.isdigit()}
    config    = ModelConfig(vocab_size=len(vocab))

    with open(args.time_bins) as f:
        time_bin_edges: list[float] = json.load(f)

    # ── Load AR model ─────────────────────────────────────────────────────────
    ckpt = torch.load(args.ar_ckpt, map_location=device)
    if "config" in ckpt:
        config = ModelConfig(**ckpt["config"])
        config.vocab_size = len(vocab)
    model = ICUAutoregressiveModel(config).to(device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    print(f"Loaded AR model ({model.count_parameters():,} params) from {args.ar_ckpt}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = ICUDataset(
        index_path          = args.index,
        data_dir            = args.data_dir,
        vocab_path          = args.vocab,
        norm_stats_path     = args.norm,
        bin_edges_path      = args.bins,
        time_bin_edges_path = args.time_bins,
        max_len             = config.max_len,
    )

    # ── Next-event accuracy ───────────────────────────────────────────────────
    print("\n── Next-event accuracy ──────────────────────────────────────────")
    acc = eval_next_event_accuracy(model, dataset, device, max_stays=args.max_next_ev)
    print(f"  new-onset : {acc['new_onset_acc']:.4f}  (n={acc['new_onset_total']:,})")
    print(f"  repeat    : {acc['repeat_acc']:.4f}  (n={acc['repeat_total']:,})")

    # ── Zero-shot mortality ───────────────────────────────────────────────────
    # MIMIC-IV: itemid 223755 = "Patient Died" chartevents entry (example)
    # Replace with the actual itemids from your extract if different.
    DEATH_ITEMIDS: set[int] = {223755, 224085}

    print("\n── Zero-shot mortality (rollout) ────────────────────────────────")
    mort = eval_mortality_zero_shot(
        model, dataset, vocab_inv, time_bin_edges, device,
        death_itemids=DEATH_ITEMIDS,
        n_rollout=args.n_rollout,
        horizon_hours=args.horizon,
        max_stays=args.max_stays,
    )
    for k, v in mort.items():
        print(f"  {k:<30}: {v}")

    # ── Branch A comparison ───────────────────────────────────────────────────
    if args.mem_ckpt:
        print("\n── Branch A (MEM) mortality ─────────────────────────────────────")
        mem = eval_branch_a_mortality(args.mem_ckpt, dataset, device, args.max_stays)
        for k, v in mem.items():
            print(f"  {k:<30}: {v}")

        print("\n── Head-to-head ─────────────────────────────────────────────────")
        print(f"  {'Metric':<30} {'Branch A (MEM)':<20} {'Branch B (AR)'}")
        print(f"  {'Mortality AUROC':<30} {mem.get('mem_mortality_auroc','N/A'):<20} "
              f"{mort.get('mortality_auroc','N/A')}")
        print(f"  {'Mortality Brier':<30} {mem.get('mem_mortality_brier','N/A'):<20} "
              f"{mort.get('mortality_brier','N/A')}")


if __name__ == "__main__":
    main()
