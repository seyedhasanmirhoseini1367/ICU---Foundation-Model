"""
dataloader/dataloader.py

ICUDataset — one training example = one ICU stay.

Each event in the stay timeline becomes one token in a sequence:

    [CLS]  e₁  e₂  ...  eN  [PAD]  ...  [PAD]
     pos0  pos1            posN    ...   pos511

The dataset handles these responsibilities:
    1. Stay windowing     — 80/20 temporal split: first 80% is the model input,
                            last 20% provides vital forecast targets (no leakage).
                            If the input portion exceeds max_len, window_mode
                            controls which events to keep ("last" or "random").
    2. Value normalisation — Z-scores each event's value using per-itemid stats
    3. Value binning       — maps each value to a per-itemid decile bin (0-9)
                             for use as the pretrain classification target
    4. Vocab mapping       — converts raw itemids to contiguous token indices;
                             unknowns map to [UNK] (not silently to [PAD])
    5. Sequence assembly   — prepends [CLS], pads/truncates to max_len

Masking for Masked Event Modeling is NOT applied here.
It is done per-batch in training/pretrain.py via apply_random_mask().
"""

import json
import math
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from model.heads import (
    VITAL_ITEMIDS, N_VITALS,
    VASOPRESSOR_ITEMIDS, LACTATE_ITEMIDS, PH_ITEMID, N_PROXY_TARGETS,
)

SOURCE_MAP = {"CHART": 0, "INPUT": 1, "OUTPUT": 2, "LAB": 3}

# Special token indices — must match ModelConfig and tokenizer/build_vocab.py
PAD_TOKEN_ID  = 0
MASK_TOKEN_ID = 1
CLS_TOKEN_ID  = 2
UNK_TOKEN_ID  = 3


class ICUDataset(Dataset):
    """
    Loads one ICU stay CSV per __getitem__ call and returns a ready-to-use
    (inputs, labels) pair.

    inputs — dict of tensors, all shape [max_len]:
        itemid        LongTensor    vocab-mapped token index
        source        LongTensor    0=CHART 1=INPUT 2=OUTPUT 3=LAB
        delta_hours   FloatTensor   hours since ICU admission (0 for [CLS]/[PAD])
        value         FloatTensor   Z-scored measurement value (0 for [CLS]/[PAD])
        value_bins    LongTensor    per-itemid decile bin index 0-9 (for pretrain)
        delta_bins    LongTensor    global time-gap bin for gap from pos i to i+1
                                    (only present when time_bin_edges_path is given)
        padding_mask  BoolTensor    True=real event, False=padding

    labels — plain Python dict (collated into tensors by DataLoader):
        hospital_expire_flag  int
        los                   float  ICU length of stay in days
        age                   int
        gender                int    0=Male, 1=Female
        vital_targets         FloatTensor [N_VITALS]        Z-scored future vital means
        vital_mask            BoolTensor  [N_VITALS]        True where vital was observed
        proxy_targets         FloatTensor [N_PROXY_TARGETS] self-supervised stay targets
        proxy_mask            BoolTensor  [N_PROXY_TARGETS] True where target is valid
    """

    def __init__(
        self,
        index_path           : str | Path,
        data_dir             : str | Path,
        vocab_path           : str | Path | None = None,
        norm_stats_path      : str | Path | None = None,
        bin_edges_path       : str | Path | None = None,
        time_bin_edges_path  : str | Path | None = None,  # Branch B: global gap bins
        max_len              : int  = 512,
        window_mode          : str  = "last",   # "last" or "random"
    ):
        self.index       = pd.read_csv(index_path)
        self.data_dir    = Path(data_dir)
        self.max_len     = max_len
        self.window_mode = window_mode

        self.vocab = None
        if vocab_path is not None:
            with open(vocab_path) as f:
                self.vocab = json.load(f)

        self.norm_stats = None
        if norm_stats_path is not None:
            with open(norm_stats_path) as f:
                self.norm_stats = json.load(f)

        self.bin_edges = None
        if bin_edges_path is not None:
            with open(bin_edges_path) as f:
                self.bin_edges = json.load(f)

        self.time_bin_edges = None   # flat list of 9 floats (global gap quantile edges)
        if time_bin_edges_path is not None:
            with open(time_bin_edges_path) as f:
                self.time_bin_edges = json.load(f)
            # Warn if many vocab items lack decile edges (will always get bin 0)
            if self.vocab is not None:
                n_real = sum(1 for k in self.vocab if k not in {"[PAD]","[MASK]","[CLS]","[UNK]"})
                n_covered = sum(
                    1 for k in self.vocab
                    if k not in {"[PAD]","[MASK]","[CLS]","[UNK]"}
                    and self.bin_edges.get(k)
                )
                pct = 100.0 * n_covered / max(1, n_real)
                if pct < 95.0:
                    print(f"[bin_edges] {n_covered}/{n_real} vocab itemids have decile bins "
                          f"({pct:.1f}% covered); {n_real - n_covered} itemids always map to bin 0")

    def __len__(self) -> int:
        return len(self.index)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _to_token_idx(self, raw_itemid: int) -> int:
        """Map a raw MIMIC itemid to its vocab token index.
        Unknown itemids return UNK (not PAD) so they don't look like padding."""
        if self.vocab is None:
            return int(raw_itemid)
        return self.vocab.get(str(raw_itemid), UNK_TOKEN_ID)

    def _zscore(self, raw_value: float, raw_itemid: int) -> float:
        if self.norm_stats is None:
            return float(raw_value)
        stats = self.norm_stats.get(str(raw_itemid))
        if stats is None:
            return 0.0
        return (raw_value - stats["mean"]) / stats["std"]

    def _to_delta_bin_vec(
        self,
        delta_hours_arr : np.ndarray,   # [L]  float64
        mask_arr        : np.ndarray,   # [L]  bool
    ) -> np.ndarray:                    # [L]  int64
        """Compute per-position global time-gap bin.

        gap[i] = delta_hours[i+1] - delta_hours[i]  where both i and i+1 are real.
        gap for the last real event and all PAD positions = 0 → bin 0.
        """
        L    = len(delta_hours_arr)
        bins = np.zeros(L, dtype=np.int64)
        if self.time_bin_edges is None:
            return bins

        edges     = np.asarray(self.time_bin_edges, dtype=np.float64)
        both_real = mask_arr[:-1] & mask_arr[1:]              # [L-1]
        raw_gaps  = np.diff(delta_hours_arr.astype(np.float64))  # [L-1]
        gaps      = np.where(both_real, np.maximum(raw_gaps, 0.0), 0.0)
        bins[:-1] = np.searchsorted(edges, gaps)
        return bins

    def _to_value_bin(self, raw_value: float, raw_itemid: int) -> int:
        """Map a raw measurement value to its per-itemid decile bin (0 to n_bins-1).
        Returns 0 for itemids with no bin edges (e.g., special tokens, rare items)."""
        if self.bin_edges is None:
            return 0
        edges = self.bin_edges.get(str(raw_itemid))
        if not edges:
            return 0
        # searchsorted on the 9 decile edges gives an index in [0, 9]
        return int(np.searchsorted(edges, raw_value))

    # ── Core logic ────────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> tuple[dict, dict]:
        row = self.index.iloc[idx]

        # ── 1. Load raw stay CSV ──────────────────────────────────────────────
        df_raw = pd.read_csv(self.data_dir / row["file_path"])
        n_events = len(df_raw)

        # ── 2. Temporal 80/20 split ───────────────────────────────────────────
        #    Input window  : first 80% of events (up to max_len-1 slots after CLS)
        #    Future window : last 20% of events  → vital forecast targets
        #
        #    This ensures ALL stays contribute vital forecast signal (even short ones),
        #    and that future vitals are truly unseen during the encoder forward pass.
        split_pos = max(n_events * 4 // 5, 1)
        df_input_events = df_raw.iloc[:split_pos].copy()
        df_future       = df_raw.iloc[split_pos:]

        # ── 3. Vital forecast targets from the future window ──────────────────
        vital_targets = np.zeros(N_VITALS, dtype=np.float32)
        vital_mask    = np.zeros(N_VITALS, dtype=bool)
        for i, iid in enumerate(VITAL_ITEMIDS):
            vals = df_future.loc[df_future["itemid"] == iid, "value"].dropna()
            if len(vals) > 0:
                vital_targets[i] = self._zscore(float(vals.mean()), iid)
                vital_mask[i]    = True

        # ── 4. Proxy targets from full stay (self-supervised, no labels needed) ─
        #    Computed on df_raw (ALL events) before the temporal split.
        #    Target 0: log-normalised total event count           — always valid
        #    Target 1: mean Z-scored HR (itemid 220045)          — valid only if HR present
        #    Target 2: any vasopressor administered               — always valid
        #    Target 3: any critical lab (lactate>4 or pH<7.25)   — always valid
        log_count = math.log(n_events + 1) / math.log(5001)

        hr_vals  = df_raw.loc[df_raw["itemid"] == 220045, "value"].dropna()
        hr_valid = len(hr_vals) > 0
        mean_hr  = self._zscore(float(hr_vals.mean()), 220045) if hr_valid else 0.0

        any_vaso = float(df_raw["itemid"].isin(VASOPRESSOR_ITEMIDS).any())

        lac      = df_raw.loc[df_raw["itemid"].isin(LACTATE_ITEMIDS), "value"].dropna()
        ph       = df_raw.loc[df_raw["itemid"] == PH_ITEMID,          "value"].dropna()
        any_crit = float(
            (len(lac) > 0 and float(lac.max()) > 4.0)
            or (len(ph) > 0 and float(ph.min()) < 7.25)
        )

        proxy_targets_arr = np.array([log_count, mean_hr, any_vaso, any_crit], dtype=np.float32)
        proxy_mask_arr    = np.array([True, hr_valid, True, True],             dtype=bool)

        # ── 6. Build model input from the input window ────────────────────────
        df = df_input_events.copy()
        df["source"] = df["source"].map(SOURCE_MAP).fillna(0).astype(int)
        df = df[["delta_hours", "source", "itemid", "value"]].fillna(0)

        # ── 7. Value bins (from raw values, before Z-scoring) ─────────────────
        df["value_bin"] = [
            self._to_value_bin(float(v), int(iid))
            for v, iid in zip(df["value"].values, df["itemid"].values)
        ]

        # ── 8. Z-score normalise values ───────────────────────────────────────
        df["value"] = [
            self._zscore(float(v), int(iid))
            for v, iid in zip(df["value"].values, df["itemid"].values)
        ]

        # ── 9. Map raw itemids to vocab token indices ─────────────────────────
        df["itemid"] = [self._to_token_idx(int(iid)) for iid in df["itemid"].values]

        # ── 10. Window selection for long input sequences ─────────────────────
        #    budget = max_len - 1  (one slot reserved for [CLS])
        budget = self.max_len - 1
        if len(df) > budget:
            if self.window_mode == "random":
                start = random.randint(0, len(df) - budget)
                df = df.iloc[start:start + budget].reset_index(drop=True)
            else:  # "last" — most recent events, closer to outcome
                df = df.iloc[-budget:].reset_index(drop=True)

        # ── 11. Prepend [CLS] at position 0 ──────────────────────────────────
        cls_row = pd.DataFrame(
            [[0.0, 0, CLS_TOKEN_ID, 0.0, 0]],
            columns=["delta_hours", "source", "itemid", "value", "value_bin"],
        )
        df = pd.concat([cls_row, df], ignore_index=True)

        # ── 12. Pad to max_len ────────────────────────────────────────────────
        n = len(df)
        if n < self.max_len:
            pad_count = self.max_len - n
            pad_rows  = pd.DataFrame(
                [[0.0, 0, PAD_TOKEN_ID, 0.0, 0]] * pad_count,
                columns=["delta_hours", "source", "itemid", "value", "value_bin"],
            )
            df   = pd.concat([df, pad_rows], ignore_index=True)
            mask = [True] * n + [False] * pad_count
        else:
            mask = [True] * self.max_len

        # ── 13. Build input tensors ───────────────────────────────────────────
        mask_arr = np.array(mask, dtype=bool)
        inputs = {
            "itemid"      : torch.tensor(df["itemid"].values,      dtype=torch.long),
            "source"      : torch.tensor(df["source"].values,      dtype=torch.long),
            "delta_hours" : torch.tensor(df["delta_hours"].values, dtype=torch.float32),
            "value"       : torch.tensor(df["value"].values,       dtype=torch.float32),
            "value_bins"  : torch.tensor(df["value_bin"].values,   dtype=torch.long),
            "padding_mask": torch.tensor(mask_arr,                 dtype=torch.bool),
        }

        # delta_bins: included only when time_bin_edges were loaded (Branch B)
        if self.time_bin_edges is not None:
            delta_bin_arr = self._to_delta_bin_vec(
                df["delta_hours"].values, mask_arr
            )
            inputs["delta_bins"] = torch.tensor(delta_bin_arr, dtype=torch.long)

        # ── 14. Build label dict ──────────────────────────────────────────────
        labels = {
            "hospital_expire_flag": int(row["hospital_expire_flag"]),
            "los"                 : float(row["los"]),
            "age"                 : int(row["age"]),
            "gender"              : 0 if row["gender"] == "M" else 1,
            "vital_targets"       : torch.tensor(vital_targets,     dtype=torch.float32),
            "vital_mask"          : torch.tensor(vital_mask,        dtype=torch.bool),
            "proxy_targets"       : torch.tensor(proxy_targets_arr, dtype=torch.float32),
            "proxy_mask"          : torch.tensor(proxy_mask_arr,    dtype=torch.bool),
        }

        return inputs, labels


# ── Convenience factory ───────────────────────────────────────────────────────

def make_dataloader(
    index_path           : str | Path,
    data_dir             : str | Path,
    vocab_path           : str | Path | None = None,
    norm_stats_path      : str | Path | None = None,
    bin_edges_path       : str | Path | None = None,
    time_bin_edges_path  : str | Path | None = None,
    max_len              : int  = 512,
    batch_size           : int  = 16,
    shuffle              : bool = True,
    num_workers          : int  = 0,
    window_mode          : str  = "last",
) -> DataLoader:
    dataset = ICUDataset(
        index_path=index_path,
        data_dir=data_dir,
        vocab_path=vocab_path,
        norm_stats_path=norm_stats_path,
        bin_edges_path=bin_edges_path,
        time_bin_edges_path=time_bin_edges_path,
        max_len=max_len,
        window_mode=window_mode,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    ROOT = BASE.parent

    dataset = ICUDataset(
        index_path       = BASE / "index.csv",
        data_dir         = BASE / "all_stays",
        vocab_path       = ROOT / "tokenizer" / "vocab.json",
        norm_stats_path  = ROOT / "tokenizer" / "norm_stats.json",
        bin_edges_path   = ROOT / "tokenizer" / "bin_edges.json",
        max_len          = 512,
        window_mode      = "last",
    )

    print(f"Total stays  : {len(dataset)}")
    inputs, labels = dataset[0]
    for key, tensor in inputs.items():
        print(f"  {key:<14}: shape={list(tensor.shape)}  dtype={tensor.dtype}")
    print(f"  labels       : {labels}")
    print(f"  real events  : {inputs['padding_mask'].sum().item()} / 512")
