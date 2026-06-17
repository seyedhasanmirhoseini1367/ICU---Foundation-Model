import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

SOURCE_MAP = {'CHART': 0, 'INPUT': 1, 'OUTPUT': 2, 'LAB': 3}


class ICUDataset(Dataset):
    """
    One training example = one ICU stay.

    __getitem__ always returns (inputs, labels).

    inputs — dict of tensors, shape [max_len]:
        itemid       LongTensor   — embedding lookup (vocab index)
        source       LongTensor   — embedding lookup (0-3)
        delta_hours  FloatTensor  — continuous time from ICU admission
        value        FloatTensor  — measurement value
        padding_mask BoolTensor   — True=real event, False=padding

    labels — dict:
        hospital_expire_flag  int   — 1 if patient died (pretrain: ignore)
        los                   float — ICU length of stay in days
        age                   int
        gender                int   — 0=M, 1=F

    pretrain=True  → training loop uses only inputs, ignores labels
    pretrain=False → training loop uses inputs + labels
    """

    def __init__(self, index_path, data_dir, max_len=512, pretrain=True):
        self.index    = pd.read_csv(index_path)
        self.data_dir = data_dir
        self.max_len  = max_len
        self.pretrain = pretrain

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row = self.index.iloc[idx]

        # ── load stay timeline ──────────────────────────
        df = pd.read_csv(f'{self.data_dir}/{row["file_path"]}')
        df['source'] = df['source'].map(SOURCE_MAP).fillna(0).astype(int)
        df = df[['delta_hours', 'source', 'itemid', 'value']].fillna(0)

        # ── truncate or pad to max_len ──────────────────
        n = len(df)
        if n >= self.max_len:
            df   = df.iloc[:self.max_len].reset_index(drop=True)
            mask = [True] * self.max_len
        else:
            pad  = pd.DataFrame([[0.0, 0, 0, 0.0]] * (self.max_len - n),
                                 columns=df.columns)
            df   = pd.concat([df, pad], ignore_index=True)
            mask = [True] * n + [False] * (self.max_len - n)

        # ── build inputs dict ───────────────────────────
        inputs = {
            'itemid'      : torch.tensor(df['itemid'].values,      dtype=torch.long),
            'source'      : torch.tensor(df['source'].values,      dtype=torch.long),
            'delta_hours' : torch.tensor(df['delta_hours'].values, dtype=torch.float32),
            'value'       : torch.tensor(df['value'].values,       dtype=torch.float32),
            'padding_mask': torch.tensor(mask,                     dtype=torch.bool),
        }

        # ── build labels dict ───────────────────────────
        labels = {
            'hospital_expire_flag': int(row['hospital_expire_flag']),
            'los'                 : float(row['los']),
            'age'                 : int(row['age']),
            'gender'              : 0 if row['gender'] == 'M' else 1,
        }

        return inputs, labels


# ── test ─────────────────────────────────────────────────
if __name__ == '__main__':
    import os
    BASE     = r'D:\datasets\HOSP&ICU-datasets(100000 med-data from 2001-2019）\dataloader'
    IDX      = os.path.join(BASE, 'index.csv')
    DATA_DIR = os.path.join(BASE, 'all_stays')

    dataset = ICUDataset(index_path=IDX, data_dir=DATA_DIR, max_len=512, pretrain=True)
    loader  = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Total stays : {len(dataset)}")

    inputs, labels = dataset[0]
    print(f"\nSingle sample:")
    print(f"  itemid       : {inputs['itemid'].shape}  dtype={inputs['itemid'].dtype}")
    print(f"  source       : {inputs['source'].shape}  dtype={inputs['source'].dtype}")
    print(f"  delta_hours  : {inputs['delta_hours'].shape}  dtype={inputs['delta_hours'].dtype}")
    print(f"  value        : {inputs['value'].shape}  dtype={inputs['value'].dtype}")
    print(f"  padding_mask : {inputs['padding_mask'].shape}  dtype={inputs['padding_mask'].dtype}")
    print(f"  real events  : {inputs['padding_mask'].sum().item()} / 512")
    print(f"  labels       : {labels}")

    print(f"\nFirst batch:")
    for batch_inputs, batch_labels in loader:
        print(f"  itemid shape : {batch_inputs['itemid'].shape}")   # [4, 512]
        print(f"  mortality    : {batch_labels['hospital_expire_flag']}")
        break
