"""
dataloader/dataloader.py

ICUDataset — one training example = one ICU stay.

Each event in the stay timeline becomes one token in a sequence:

    [CLS]  e₁  e₂  ...  eN  [PAD]  ...  [PAD]
     pos0  pos1            posN    ...   pos511

The dataset handles three responsibilities:
    1. Value normalisation  — Z-scores each event's value using per-itemid
                              mean and std from norm_stats.json
    2. Vocab mapping        — converts raw MIMIC itemids (e.g. 220045) to
                              contiguous token indices (e.g. 47)
    3. Sequence assembly    — prepends [CLS], pads/truncates to max_len

Masking for Masked Event Modeling is NOT done here.
It is applied per-batch in training/pretrain.py via apply_random_mask()
so this dataset can be reused for both pretrain and fine-tune without changes.

Usage:
    dataset = ICUDataset(
        index_path      = 'dataloader/index.csv',
        data_dir        = 'dataloader/all_stays',
        vocab_path      = 'vocab/vocab.json',
        norm_stats_path = 'vocab/norm_stats.json',
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from model.heads import VITAL_ITEMIDS, N_VITALS

# Maps the string source column in stay CSVs to an integer for source_emb
SOURCE_MAP = {"CHART": 0, "INPUT": 1, "OUTPUT": 2, "LAB": 3}

# Special token indices — must match ModelConfig and vocab/build_vocab.py
PAD_TOKEN_ID  = 0
MASK_TOKEN_ID = 1
CLS_TOKEN_ID  = 2


class ICUDataset(Dataset):
    """
    Loads one ICU stay CSV per __getitem__ call and returns a ready-to-use
    (inputs, labels) pair.

    inputs — dict of tensors, all shape [max_len]:
        itemid        LongTensor    vocab-mapped token index
        source        LongTensor    0=CHART  1=INPUT  2=OUTPUT  3=LAB
        delta_hours   FloatTensor   hours since ICU admission (0 for [CLS]/[PAD])
        value         FloatTensor   Z-scored measurement value (0 for [CLS]/[PAD])
        padding_mask  BoolTensor    True=real event, False=padding

    labels — plain Python dict (collated into tensors by DataLoader):
        hospital_expire_flag  int    1=died, 0=survived
        los                   float  ICU length of stay in days
        age                   int
        gender                int    0=Male, 1=Female
    """

    def __init__(
        self,
        index_path      : str | Path,
        data_dir        : str | Path,
        vocab_path      : str | Path | None = None,
        norm_stats_path : str | Path | None = None,
        max_len         : int = 512,
    ):
        self.index    = pd.read_csv(index_path)
        self.data_dir = Path(data_dir)
        self.max_len  = max_len

        # Vocab: str(raw_itemid) → token_idx
        # If not provided, raw itemid integers are used directly (for quick tests)
        self.vocab = None
        if vocab_path is not None:
            with open(vocab_path) as f:
                self.vocab = json.load(f)

        # Norm stats: str(raw_itemid) → {"mean": float, "std": float}
        self.norm_stats = None
        if norm_stats_path is not None:
            with open(norm_stats_path) as f:
                self.norm_stats = json.load(f)

    def __len__(self) -> int:
        return len(self.index)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _to_token_idx(self, raw_itemid: int) -> int:
        """Map a raw MIMIC itemid to its contiguous vocab token index."""
        if self.vocab is None:
            return int(raw_itemid)
        return self.vocab.get(str(raw_itemid), PAD_TOKEN_ID)

    def _zscore(self, raw_value: float, raw_itemid: int) -> float:
        """Z-score a single value using the itemid's precomputed mean and std."""
        if self.norm_stats is None:
            return float(raw_value)
        stats = self.norm_stats.get(str(raw_itemid))
        if stats is None:
            return 0.0
        return (raw_value - stats["mean"]) / stats["std"]

    # ── Core logic ────────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> tuple[dict, dict]:
        row = self.index.iloc[idx]

        # ── 1. Load the raw stay CSV (kept intact for vital target extraction)
        df_raw = pd.read_csv(self.data_dir / row["file_path"])

        # ── 2. Vital forecast targets — events AFTER the input window ─────────
        #    Input window = first (max_len - 1) events (CLS takes slot 0).
        #    Future window = everything from index max_len-1 onwards in df_raw.
        #    For each vital: mean normalised value in the future window.
        #    vital_mask is False where a vital was never measured there.
        df_future = df_raw.iloc[self.max_len - 1:]
        vital_targets = np.zeros(N_VITALS, dtype=np.float32)
        vital_mask    = np.zeros(N_VITALS, dtype=bool)
        for i, iid in enumerate(VITAL_ITEMIDS):
            vals = df_future.loc[df_future["itemid"] == iid, "value"].dropna()
            if len(vals) > 0:
                vital_targets[i] = self._zscore(float(vals.mean()), iid)
                vital_mask[i]    = True

        # ── 3. Build model input from the full df_raw ─────────────────────────
        df = df_raw.copy()
        df["source"] = df["source"].map(SOURCE_MAP).fillna(0).astype(int)
        df = df[["delta_hours", "source", "itemid", "value"]].fillna(0)

        # ── 4. Normalise values BEFORE vocab mapping ──────────────────────────
        df["value"] = [
            self._zscore(v, iid)
            for v, iid in zip(df["value"].values, df["itemid"].values)
        ]

        # ── 5. Map raw itemids to vocab token indices ─────────────────────────
        df["itemid"] = [self._to_token_idx(int(iid)) for iid in df["itemid"].values]

        # ── 6. Prepend [CLS] at position 0 ───────────────────────────────────
        cls_row = pd.DataFrame(
            [[0.0, 0, CLS_TOKEN_ID, 0.0]],
            columns=df.columns,
        )
        df = pd.concat([cls_row, df], ignore_index=True)

        # ── 7. Truncate or pad to max_len ─────────────────────────────────────
        n = len(df)
        if n >= self.max_len:
            df   = df.iloc[: self.max_len].reset_index(drop=True)
            mask = [True] * self.max_len
        else:
            pad_count = self.max_len - n
            pad_rows  = pd.DataFrame(
                [[0.0, 0, PAD_TOKEN_ID, 0.0]] * pad_count,
                columns=df.columns,
            )
            df   = pd.concat([df, pad_rows], ignore_index=True)
            mask = [True] * n + [False] * pad_count

        # ── 8. Build input tensors ────────────────────────────────────────────
        inputs = {
            "itemid"      : torch.tensor(df["itemid"].values,      dtype=torch.long),
            "source"      : torch.tensor(df["source"].values,      dtype=torch.long),
            "delta_hours" : torch.tensor(df["delta_hours"].values, dtype=torch.float32),
            "value"       : torch.tensor(df["value"].values,       dtype=torch.float32),
            "padding_mask": torch.tensor(mask,                     dtype=torch.bool),
        }

        # ── 9. Build label dict ───────────────────────────────────────────────
        labels = {
            "hospital_expire_flag": int(row["hospital_expire_flag"]),
            "los"                 : float(row["los"]),
            "age"                 : int(row["age"]),
            "gender"              : 0 if row["gender"] == "M" else 1,
            "vital_targets"       : torch.tensor(vital_targets, dtype=torch.float32),
            "vital_mask"          : torch.tensor(vital_mask,    dtype=torch.bool),
        }

        return inputs, labels


# ── Convenience factory ───────────────────────────────────────────────────────

def make_dataloader(
    index_path      : str | Path,
    data_dir        : str | Path,
    vocab_path      : str | Path | None = None,
    norm_stats_path : str | Path | None = None,
    max_len         : int  = 512,
    batch_size      : int  = 16,
    shuffle         : bool = True,
    num_workers     : int  = 0,
) -> DataLoader:
    """Creates an ICUDataset wrapped in a PyTorch DataLoader."""
    dataset = ICUDataset(
        index_path=index_path,
        data_dir=data_dir,
        vocab_path=vocab_path,
        norm_stats_path=norm_stats_path,
        max_len=max_len,
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
        index_path      = BASE / "index.csv",
        data_dir        = BASE / "all_stays",
        vocab_path      = ROOT / "tokenizer" / "vocab.json",
        norm_stats_path = ROOT / "tokenizer" / "norm_stats.json",
        max_len         = 512,
    )

    print(f"Total stays  : {len(dataset)}")

    inputs, labels = dataset[0]
    for key, tensor in inputs.items():
        print(f"  {key:<14}: shape={list(tensor.shape)}  dtype={tensor.dtype}")
    print(f"  labels       : {labels}")
    print(f"  real events  : {inputs['padding_mask'].sum().item()} / 512  "
          f"(including [CLS])")
