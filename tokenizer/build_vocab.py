"""
tokenizer/build_vocab.py

Scans every extracted ICU stay CSV and produces three JSON files:

  vocab.json       — maps each raw MIMIC itemid to a contiguous token index
                     {str(itemid): int(token_idx), ...}
                     Special tokens occupy the first four indices:
                       0 → [PAD]   (padding, ignored by attention)
                       1 → [MASK]  (replaces events during MEM pretraining)
                       2 → [CLS]   (prepended; output = patient summary)
                       3 → [UNK]   (unknown itemid seen at inference time)
                     Real itemids start at index 4.

  norm_stats.json  — per-itemid mean and std for Z-score normalisation
                     {str(itemid): {"mean": float, "std": float}, ...}

  bin_edges.json   — per-itemid quantile bin edges for value discretisation
                     {str(itemid): [e1, e2, ..., e9], ...}
                     9 edges define 10 quantile bins (deciles).
                     Used by the pretrain loss (CE over bins instead of MSE).

Run once before any training:
    python tokenizer/build_vocab.py
"""

import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
STAYS_DIR     = ROOT / "dataloader" / "all_stays"
OUT_DIR       = Path(__file__).resolve().parent   # tokenizer/
VOCAB_PATH    = OUT_DIR / "vocab.json"
NORM_PATH     = OUT_DIR / "norm_stats.json"
BIN_EDGES_PATH = OUT_DIR / "bin_edges.json"

# Special tokens (must match ModelConfig)
SPECIAL_TOKENS = {"[PAD]": 0, "[MASK]": 1, "[CLS]": 2, "[UNK]": 3}
FIRST_REAL_IDX = len(SPECIAL_TOKENS)   # real itemids start at 4

# Quantile binning
N_VALUE_BINS  = 10     # decile bins per itemid
MAX_RESERVOIR = 10_000 # reservoir size per itemid (avoids OOM on large datasets)


# ── Core functions ────────────────────────────────────────────────────────────

def collect_stats(stay_files: list[Path]) -> tuple[set, dict, dict]:
    """
    Single pass over all stay CSVs.

    Returns:
        itemid_set    — set of all unique raw itemids (int)
        value_accum   — {itemid: [running_sum, running_sum_sq, count]}
        value_samples — {itemid: [sampled_values]}  (reservoir, max MAX_RESERVOIR)
    """
    itemid_set    = set()
    value_accum   = defaultdict(lambda: [0.0, 0.0, 0])
    value_samples = defaultdict(list)

    rng = random.Random(42)

    for path in stay_files:
        df = pd.read_csv(path, usecols=["itemid", "value"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        for itemid, val in zip(df["itemid"].values, df["value"].values):
            iid = int(itemid)
            itemid_set.add(iid)

            if np.isnan(val):
                continue

            acc = value_accum[iid]
            acc[0] += val
            acc[1] += val ** 2
            acc[2] += 1

            # Reservoir sampling: keep up to MAX_RESERVOIR values per itemid
            samples = value_samples[iid]
            n = acc[2]
            if len(samples) < MAX_RESERVOIR:
                samples.append(val)
            else:
                j = rng.randint(0, n - 1)
                if j < MAX_RESERVOIR:
                    samples[j] = val

    return itemid_set, value_accum, value_samples


def build_vocab(itemid_set: set) -> dict:
    """
    Creates a {str(itemid): token_idx} mapping with contiguous indices.
    Special tokens first (0-3); real itemids sorted after that.
    Sorting ensures reproducible vocab across runs.
    """
    vocab = dict(SPECIAL_TOKENS)
    for idx, itemid in enumerate(sorted(itemid_set), start=FIRST_REAL_IDX):
        vocab[str(itemid)] = idx
    return vocab


def build_norm_stats(value_accum: dict) -> dict:
    """
    Computes per-itemid mean and std from running accumulators.
    Falls back to mean=0, std=1 for itemids with < 2 observations.
    """
    norm_stats = {}
    for itemid, (s, sq, n) in value_accum.items():
        mean = s / n if n > 0 else 0.0
        var  = (sq / n - mean ** 2) if n > 1 else 1.0
        std  = float(np.sqrt(max(var, 0.0)))
        std  = max(std, 1e-6)
        norm_stats[str(itemid)] = {
            "mean": round(mean, 6),
            "std" : round(std,  6),
        }
    return norm_stats


def build_bin_edges(value_samples: dict) -> dict:
    """
    Computes per-itemid decile edges from the reservoir-sampled values.

    Returns {str(itemid): [e1, ..., e9]} where the 9 edges define 10 bins.
    Itemids with fewer than 10 distinct values get an empty edge list
    (the dataloader assigns bin 0 for those).
    """
    bin_edges = {}
    percentiles = np.linspace(0, 100, N_VALUE_BINS + 1)[1:-1]  # 10,20,...,90

    for itemid, samples in value_samples.items():
        arr = np.array(samples, dtype=np.float32)
        arr = arr[~np.isnan(arr)]
        if len(np.unique(arr)) < N_VALUE_BINS:
            bin_edges[str(itemid)] = []
            continue
        edges = np.percentile(arr, percentiles).tolist()
        bin_edges[str(itemid)] = [round(float(e), 6) for e in edges]

    return bin_edges


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    stay_files = sorted(STAYS_DIR.glob("*.csv"))
    if not stay_files:
        raise FileNotFoundError(
            f"No CSV files found in {STAYS_DIR}.\n"
            "Run  python dataloader/extract.py  first."
        )

    print(f"Scanning {len(stay_files)} stay files in {STAYS_DIR} ...")
    itemid_set, value_accum, value_samples = collect_stats(stay_files)

    vocab      = build_vocab(itemid_set)
    norm_stats = build_norm_stats(value_accum)
    bin_edges  = build_bin_edges(value_samples)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VOCAB_PATH.write_text(json.dumps(vocab,      indent=2))
    NORM_PATH.write_text(json.dumps(norm_stats,  indent=2))
    BIN_EDGES_PATH.write_text(json.dumps(bin_edges, indent=2))

    n_with_edges = sum(1 for v in bin_edges.values() if v)
    print(f"\nVocab      : {len(vocab)} tokens  "
          f"({len(itemid_set)} itemids + {FIRST_REAL_IDX} special)")
    print(f"Norm stats : {len(norm_stats)} itemids with mean/std")
    print(f"Bin edges  : {n_with_edges}/{len(bin_edges)} itemids with {N_VALUE_BINS} bins")
    print(f"\nSaved → {VOCAB_PATH}")
    print(f"Saved → {NORM_PATH}")
    print(f"Saved → {BIN_EDGES_PATH}")


if __name__ == "__main__":
    main()
