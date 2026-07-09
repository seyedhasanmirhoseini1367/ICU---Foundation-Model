"""
vocab/build_vocab.py

Scans every extracted ICU stay CSV and produces two JSON files:

  vocab.json       — maps each raw MIMIC itemid to a contiguous token index
                     {str(itemid): int(token_idx), ...}
                     Special tokens occupy the first three indices:
                       0 → [PAD]   (padding, ignored by attention)
                       1 → [MASK]  (replaces events during MEM pretraining)
                       2 → [CLS]   (prepended; output = patient summary)
                     Real itemids start at index 3.

  norm_stats.json  — per-itemid mean and std for Z-score normalisation
                     {str(itemid): {"mean": float, "std": float}, ...}
                     Used by ICUDataset to normalise the value field before
                     passing it into the model.

Run once before any training:
    python vocab/build_vocab.py
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
STAYS_DIR = ROOT / "dataloader" / "all_stays"
OUT_DIR   = Path(__file__).resolve().parent   # vocab/
VOCAB_PATH = OUT_DIR / "vocab.json"
NORM_PATH  = OUT_DIR / "norm_stats.json"

# Special tokens (must match ModelConfig)
SPECIAL_TOKENS = {"[PAD]": 0, "[MASK]": 1, "[CLS]": 2}
FIRST_REAL_IDX = len(SPECIAL_TOKENS)   # real itemids start at 3


# ── Core functions ────────────────────────────────────────────────────────────

def collect_itemids_and_values(stay_files: list[Path]) -> tuple[set, dict]:
    """
    Single pass over all stay CSVs.

    Returns:
        itemid_set    — set of all unique raw itemids (int)
        value_accum   — {itemid: [running_sum, running_sum_sq, count]}
                        used to compute mean/std without loading everything
    """
    itemid_set  = set()
    value_accum = defaultdict(lambda: [0.0, 0.0, 0])   # sum, sum_sq, count

    for path in stay_files:
        df = pd.read_csv(path, usecols=["itemid", "value"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        for itemid, val in zip(df["itemid"].values, df["value"].values):
            iid = int(itemid)
            itemid_set.add(iid)

            if not np.isnan(val):
                acc = value_accum[iid]
                acc[0] += val        # running sum
                acc[1] += val ** 2   # running sum of squares
                acc[2] += 1          # count

    return itemid_set, value_accum


def build_vocab(itemid_set: set) -> dict:
    """
    Creates a {str(itemid): token_idx} mapping with contiguous indices.
    Special tokens come first (indices 0-2); real itemids follow sorted.
    Sorting ensures the same vocab is produced on every run.
    """
    vocab = dict(SPECIAL_TOKENS)
    for idx, itemid in enumerate(sorted(itemid_set), start=FIRST_REAL_IDX):
        vocab[str(itemid)] = idx
    return vocab


def build_norm_stats(value_accum: dict) -> dict:
    """
    Computes per-itemid mean and standard deviation from running accumulators
    (Welford-style: avoids storing all values in memory).

    Falls back to mean=0, std=1 for itemids with fewer than 2 observations
    so the Z-score formula stays numerically safe.
    """
    norm_stats = {}
    for itemid, (s, sq, n) in value_accum.items():
        mean = s / n if n > 0 else 0.0
        # Var = E[x²] - (E[x])²
        var  = (sq / n - mean ** 2) if n > 1 else 1.0
        std  = float(np.sqrt(max(var, 0.0)))
        std  = max(std, 1e-6)           # never divide by zero
        norm_stats[str(itemid)] = {
            "mean": round(mean, 6),
            "std" : round(std,  6),
        }
    return norm_stats


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    stay_files = sorted(STAYS_DIR.glob("*.csv"))
    if not stay_files:
        raise FileNotFoundError(
            f"No CSV files found in {STAYS_DIR}.\n"
            "Run  python dataloader/extract.py  first."
        )

    print(f"Scanning {len(stay_files)} stay files in {STAYS_DIR} ...")
    itemid_set, value_accum = collect_itemids_and_values(stay_files)

    vocab      = build_vocab(itemid_set)
    norm_stats = build_norm_stats(value_accum)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VOCAB_PATH.write_text(json.dumps(vocab,      indent=2))
    NORM_PATH.write_text(json.dumps(norm_stats,  indent=2))

    print(f"\nVocab      : {len(vocab)} tokens  "
          f"({len(itemid_set)} itemids + {FIRST_REAL_IDX} special)")
    print(f"Norm stats : {len(norm_stats)} itemids with mean/std")
    print(f"\nSaved → {VOCAB_PATH}")
    print(f"Saved → {NORM_PATH}")


if __name__ == "__main__":
    main()
