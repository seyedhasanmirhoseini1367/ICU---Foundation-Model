"""
inference/rollout.py

Ancestral-sampling rollout for ICUAutoregressiveModel (Branch B).

Given a seed context (the events observed so far), the model generates
n_samples plausible future trajectories by sampling from its predicted
distributions at each step.

Rollout loop (per trajectory):
  1. Append the most recently generated event to the context.
  2. Run a forward pass (causal mask applied inside the model).
  3. Sample (itemid, value_bin, delta_bin) from the last position's logits.
  4. Convert sampled delta_bin to elapsed hours via bin midpoints.
  5. Stop when elapsed_hours > horizon_hours OR n_events > max_events.

Window sliding: if the context exceeds max_len, keep only the last
(max_len - 1) events plus [CLS] at position 0.

Zero-shot query helpers (no fine-tuning needed):
    mortality_prob()  — fraction of trajectories containing a death event
    vital_forecast()  — mean value_bin for a given vital itemid across trajectories
    event_prob()      — fraction of trajectories where an event from a set
                        occurs within `within_hours`
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model.autoregressive import ICUAutoregressiveModel

# Special token indices (must match tokenizer/build_vocab.py and ModelConfig)
PAD_TOKEN_ID = 0
CLS_TOKEN_ID = 2


# ── Sampling helpers ──────────────────────────────────────────────────────────

def _sample_token(
    logits      : torch.Tensor,   # [vocab_size]  raw logits
    temperature : float = 1.0,
    top_k       : int   = 0,
) -> int:
    """Sample one token index from logits with optional temperature + top-k."""
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-8)
    if top_k > 0:
        topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        threshold     = topk_vals[-1]
        logits        = logits.masked_fill(logits < threshold, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _bin_midpoints(edges: list[float]) -> list[float]:
    """Convert 9 quantile edges to 10 bin midpoints (in hours)."""
    pts: list[float] = []
    for i in range(len(edges) + 1):
        if len(edges) == 0:
            pts.append(0.5)           # fallback: no edges → uniform 0.5 h
        elif i == 0:
            pts.append(max(edges[0] / 2.0, 0.0))
        elif i == len(edges):
            pts.append(edges[-1] * 1.5)
        else:
            pts.append((edges[i - 1] + edges[i]) / 2.0)
    return pts


# ── Core rollout ──────────────────────────────────────────────────────────────

@torch.no_grad()
def rollout(
    model          : "ICUAutoregressiveModel",
    seed_inputs    : dict,          # single stay (not batched) — from ICUDataset
    vocab_inv      : dict[int, int], # {token_idx → raw_itemid}
    time_bin_edges : list[float],   # 9 floats (global gap quantile edges)
    device         : torch.device,
    n_samples      : int   = 50,
    horizon_hours  : float = 6.0,
    max_events     : int   = 200,
    top_k          : int   = 0,
    temperature    : float = 1.0,
) -> list[list[tuple[int, int, int]]]:
    """
    Ancestral-sampling rollout from a given seed context.

    Returns:
        List of n_samples trajectories.
        Each trajectory is a list of (raw_itemid, value_bin, delta_bin) tuples
        representing the GENERATED events (not the seed context).
    """
    model.eval()
    midpoints = _bin_midpoints(time_bin_edges)
    max_len   = model.config.max_len

    # Unpack seed context (add batch dim)
    def _to_dev(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0).to(device)

    itemid_seed       = _to_dev(seed_inputs["itemid"])        # [1, L]
    source_seed       = _to_dev(seed_inputs["source"])
    delta_hours_seed  = _to_dev(seed_inputs["delta_hours"])
    value_seed        = _to_dev(seed_inputs["value"])
    padding_mask_seed = _to_dev(seed_inputs["padding_mask"])

    # Count how many real positions are in the seed
    n_seed = int(padding_mask_seed.sum().item())

    trajectories: list[list[tuple[int, int, int]]] = []

    for _ in range(n_samples):
        # Working copies (we'll append generated events)
        itemid_ctx       = itemid_seed.clone()
        source_ctx       = source_seed.clone()
        delta_hours_ctx  = delta_hours_seed.clone()
        value_ctx        = value_seed.clone()
        padding_mask_ctx = padding_mask_seed.clone()

        trajectory: list[tuple[int, int, int]] = []
        elapsed   = 0.0
        last_t    = float(delta_hours_ctx[0, n_seed - 1].item())

        for _ in range(max_events):
            if elapsed >= horizon_hours:
                break

            # If context is full, slide window: keep [CLS] + last (max_len-1) real events
            L_ctx = int(padding_mask_ctx.sum().item())
            if L_ctx >= max_len:
                # Keep position 0 ([CLS]) and the last max_len-1 positions
                keep = torch.cat([
                    torch.tensor([0], device=device),
                    torch.arange(L_ctx - max_len + 1, L_ctx, device=device),
                ])
                itemid_ctx       = itemid_ctx[:, keep]
                source_ctx       = source_ctx[:, keep]
                delta_hours_ctx  = delta_hours_ctx[:, keep]
                value_ctx        = value_ctx[:, keep]
                padding_mask_ctx = padding_mask_ctx[:, keep]

            # Forward pass — predictions from the LAST real position
            item_logits, val_logits, dt_logits = model(
                itemid_ctx, source_ctx, delta_hours_ctx, value_ctx, padding_mask_ctx
            )

            # Last real position (not padding) for prediction
            last_real_idx = int(padding_mask_ctx.sum().item()) - 1

            # Sample from last position's logits
            sampled_itemid_tok = _sample_token(
                item_logits[0, last_real_idx], temperature, top_k
            )
            sampled_val_bin    = _sample_token(
                val_logits[0, last_real_idx], temperature, top_k
            )
            sampled_dt_bin     = _sample_token(
                dt_logits[0, last_real_idx], temperature, top_k
            )

            # Convert delta bin to elapsed hours
            dt_hours  = midpoints[sampled_dt_bin]
            elapsed  += dt_hours
            last_t   += dt_hours

            # Convert token index → raw itemid
            raw_itemid = vocab_inv.get(sampled_itemid_tok, -1)
            trajectory.append((raw_itemid, sampled_val_bin, sampled_dt_bin))

            if elapsed >= horizon_hours:
                break

            # Append generated event to context
            new_itemid       = torch.tensor([[sampled_itemid_tok]], dtype=torch.long,  device=device)
            new_source       = torch.tensor([[0]],                  dtype=torch.long,  device=device)
            new_delta_hours  = torch.tensor([[last_t]],             dtype=torch.float, device=device)
            new_value        = torch.tensor([[0.0]],                dtype=torch.float, device=device)
            new_mask         = torch.tensor([[True]],               dtype=torch.bool,  device=device)

            itemid_ctx       = torch.cat([itemid_ctx,       new_itemid],      dim=1)
            source_ctx       = torch.cat([source_ctx,       new_source],      dim=1)
            delta_hours_ctx  = torch.cat([delta_hours_ctx,  new_delta_hours], dim=1)
            value_ctx        = torch.cat([value_ctx,        new_value],       dim=1)
            padding_mask_ctx = torch.cat([padding_mask_ctx, new_mask],        dim=1)

        trajectories.append(trajectory)

    return trajectories


# ── Zero-shot query helpers ───────────────────────────────────────────────────

def mortality_prob(
    trajectories     : list[list[tuple[int, int, int]]],
    death_itemid_set : set[int],
) -> float:
    """Fraction of trajectories containing at least one death-related event."""
    if not trajectories:
        return 0.0
    hits = sum(
        1 for traj in trajectories
        if any(raw_iid in death_itemid_set for raw_iid, _, _ in traj)
    )
    return hits / len(trajectories)


def vital_forecast(
    trajectories   : list[list[tuple[int, int, int]]],
    target_itemid  : int,
    n_value_bins   : int = 10,
) -> float:
    """
    Mean value_bin (normalised to [0, 1] as bin / n_value_bins) for a given
    vital itemid across all generated events in all trajectories.
    Returns NaN if the vital is never generated.
    """
    vals = [
        vb / n_value_bins
        for traj in trajectories
        for raw_iid, vb, _ in traj
        if raw_iid == target_itemid
    ]
    return float(np.mean(vals)) if vals else float("nan")


def event_prob(
    trajectories    : list[list[tuple[int, int, int]]],
    itemid_set      : set[int],
    within_hours    : float,
    time_bin_edges  : list[float],
) -> float:
    """
    Fraction of trajectories where at least one event from `itemid_set`
    occurs within `within_hours` of the rollout start.
    """
    if not trajectories:
        return 0.0
    midpoints = _bin_midpoints(time_bin_edges)
    hits = 0
    for traj in trajectories:
        elapsed = 0.0
        for raw_iid, _, dt_bin in traj:
            if raw_iid in itemid_set and elapsed <= within_hours:
                hits += 1
                break
            elapsed += midpoints[dt_bin]
            if elapsed > within_hours:
                break
    return hits / len(trajectories)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from model.config         import ModelConfig
    from model.autoregressive import ICUAutoregressiveModel

    cfg   = ModelConfig(vocab_size=50, d_model=32, n_heads=2, d_ff=64, n_layers=2,
                        max_len=16, n_value_bins=5, n_time_bins=5)
    model = ICUAutoregressiveModel(cfg)
    model.eval()

    device = torch.device("cpu")
    model  = model.to(device)

    # Synthetic seed context (10 events + CLS)
    L = 16
    seed = {
        "itemid"       : torch.randint(4, cfg.vocab_size, (L,)),
        "source"       : torch.zeros(L, dtype=torch.long),
        "delta_hours"  : torch.arange(L, dtype=torch.float32),
        "value"        : torch.randn(L),
        "padding_mask" : torch.tensor([True]*11 + [False]*5, dtype=torch.bool),
    }

    vocab_inv      = {i: i + 1000 for i in range(cfg.vocab_size)}
    time_bin_edges = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0]

    trajs = rollout(
        model, seed, vocab_inv, time_bin_edges, device,
        n_samples=5, horizon_hours=4.0, max_events=20,
    )
    print(f"Generated {len(trajs)} trajectories")
    for i, t in enumerate(trajs):
        print(f"  traj[{i}]: {len(t)} events  first={t[0] if t else 'empty'}")

    # Zero-shot queries
    print(f"mortality_prob  : {mortality_prob(trajs, {1009, 1010}):.3f}")
    print(f"vital_forecast  : {vital_forecast(trajs, 1004)}")
    print(f"event_prob(3h)  : {event_prob(trajs, {1004, 1005}, 3.0, time_bin_edges):.3f}")
    print("rollout self-test complete.")
