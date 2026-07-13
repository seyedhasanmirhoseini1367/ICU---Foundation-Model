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

# Quantile binning — per-itemid (value) and global (time gaps)
N_VALUE_BINS      = 10       # decile bins per itemid for measurement values
MAX_RESERVOIR     = 10_000   # reservoir size per itemid
N_TIME_BINS       = 10       # decile bins for inter-event time gaps (global)
MAX_TIME_RESERVOIR = 200_000 # global gap reservoir (avoids OOM on very large datasets)

TIME_BIN_EDGES_PATH = OUT_DIR / "time_bin_edges.json"


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


def collect_time_gaps(stay_files: list[Path]) -> list:
    """
    Reservoir-sample inter-event time gaps across all stays.

    gap[i] = delta_hours[i+1] - delta_hours[i] for consecutive events i, i+1
    within the same stay.  Gaps are global (not per-itemid) because per-itemid
    gap distributions would be too sparse for reliable quantile estimation.
    """
    samples: list[float] = []
    rng   = random.Random(42)
    total = 0

    for path in stay_files:
        df = pd.read_csv(path, usecols=["delta_hours"])
        dh = pd.to_numeric(df["delta_hours"], errors="coerce").dropna().values
        if len(dh) < 2:
            continue
        gaps = np.diff(dh)
        gaps = gaps[gaps >= 0.0]   # guard against reversed timestamps
        for gap in gaps:
            total += 1
            if len(samples) < MAX_TIME_RESERVOIR:
                samples.append(float(gap))
            else:
                j = rng.randint(0, total - 1)
                if j < MAX_TIME_RESERVOIR:
                    samples[j] = float(gap)

    return samples


def build_time_bin_edges(gap_samples: list) -> list:
    """
    Compute 9 global decile edges from reservoir-sampled inter-event gaps.
    Returns a flat list of 9 floats: [e1, ..., e9].
    9 edges define 10 bins: [0, e1), [e1, e2), ..., [e9, ∞).
    """
    if len(gap_samples) < N_TIME_BINS:
        return []
    arr         = np.array(gap_samples, dtype=np.float32)
    percentiles = np.linspace(0, 100, N_TIME_BINS + 1)[1:-1]   # 10, 20, …, 90
    edges       = np.percentile(arr, percentiles).tolist()
    return [round(float(e), 6) for e in edges]


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

    print("Building time-gap statistics ...")
    gap_samples = collect_time_gaps(stay_files)

    vocab           = build_vocab(itemid_set)
    norm_stats      = build_norm_stats(value_accum)
    bin_edges       = build_bin_edges(value_samples)
    time_bin_edges  = build_time_bin_edges(gap_samples)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VOCAB_PATH.write_text(json.dumps(vocab,          indent=2))
    NORM_PATH.write_text(json.dumps(norm_stats,      indent=2))
    BIN_EDGES_PATH.write_text(json.dumps(bin_edges,  indent=2))
    TIME_BIN_EDGES_PATH.write_text(json.dumps(time_bin_edges))

    n_real_vocab = len(vocab) - FIRST_REAL_IDX
    n_with_edges = sum(1 for v in bin_edges.values() if v)
    n_no_edges   = n_real_vocab - n_with_edges
    pct_covered  = 100.0 * n_with_edges / max(1, n_real_vocab)

    print(f"\nVocab      : {len(vocab)} tokens  "
          f"({n_real_vocab} itemids + {FIRST_REAL_IDX} special)")
    print(f"Norm stats : {len(norm_stats)} itemids with mean/std")
    print(f"Bin edges  : {n_with_edges}/{n_real_vocab} real itemids have decile bins "
          f"({pct_covered:.1f}% covered)")
    if n_no_edges:
        print(f"           : {n_no_edges} itemids ({100-pct_covered:.1f}%) always map to "
              f"bin 0 — these have < {N_VALUE_BINS} distinct values in the dataset")
    print(f"Time bins  : {len(time_bin_edges)} edges from {len(gap_samples)} gap samples  "
          f"→ {N_TIME_BINS} global time bins")
    print(f"\nSaved → {VOCAB_PATH}")
    print(f"Saved → {NORM_PATH}")
    print(f"Saved → {BIN_EDGES_PATH}")
    print(f"Saved → {TIME_BIN_EDGES_PATH}")


if __name__ == "__main__":
    main()
