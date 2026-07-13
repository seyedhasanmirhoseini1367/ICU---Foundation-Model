# MIMIC-IV ICU Foundation Model

A BERT-style foundation model for ICU event sequences, pretrained on 94,458 ICU stays from MIMIC-IV via Masked Event Modeling (MEM), then fine-tuned jointly on three clinical tasks.

---

## Architecture

```
Raw ICU stay (events sorted by charttime)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  ICUEventEmbedding                                    │
│  ┌──────────┐ ┌────────┐ ┌──────────┐ ┌───────────┐  │
│  │itemid_emb│ │src_emb │ │time_proj │ │value_proj │  │
│  │[V, 256]  │ │[4, 256]│ │Linear(1→d│ │Linear(1→d)│  │
│  └────┬─────┘ └───┬────┘ └────┬─────┘ └─────┬─────┘  │
│       └───────────┴───────────┴─────────────┘         │
│                   SUM + LayerNorm                     │
└───────────────────────┬───────────────────────────────┘
                        │ [B, L, 256]
                        ▼
┌───────────────────────────────────────────────────────┐
│  TransformerEncoder  (Pre-LN, 6 layers, 8 heads)      │
│  [CLS] e₁ e₂ ... eN [PAD] ... [PAD]                  │
└──────────┬────────────────────────────────────────────┘
           │
    ┌──────┴──────┐
    │  cls_output │  [B, 256]
    └──────┬──────┘
           │
    ┌──────┴──────────────────────────┐
    │  Phase 1 — Pretraining (MEM)    │
    │  PretrainHead                   │
    │    itemid_logits [B,L,vocab]    │ → CE loss
    │    value_logits  [B,L,10 bins]  │ → CE loss (per-itemid deciles)
    │  VICReg expander  [B,256→512]   │ → VICReg loss (optional)
    └─────────────────────────────────┘
           │
    ┌──────┴──────────────────────────┐
    │  Phase 2 — Fine-tuning          │
    │  MortalityHead  → logit [B]     │ → BCE
    │  LOSHead        → days [B]      │ → MSE
    │  ForecastHead   → vitals [B,7]  │ → MSE (masked)
    └─────────────────────────────────┘
```

**Model size:** ~7M parameters  
**d_model:** 256, **n_layers:** 6, **n_heads:** 8, **max_len:** 512

---

## Pipeline

```
1. Extract stays      python dataloader/extract.py  --root <MIMIC_ROOT> --out dataloader/all_stays
2. Build vocab        python tokenizer/build_vocab.py
3. Pretrain (MEM)     python training/pretrain.py
4. Fine-tune          python training/finetune.py
```

### Kaggle (automated)
```
exec(open('/kaggle/working/kaggle_run.py').read())
```

---

## Pretraining — Masked Event Modeling

- **Masking**: BERT 80/10/10 — 15% of events selected; 80% → `[MASK]`, 10% → random token, 10% unchanged
- **Value leak fix**: the float value is zeroed at masked positions so the model cannot reconstruct itemid from the visible measurement
- **Loss**: `λ·CE(itemid) + (1-λ)·CE(value_bins)` where value_bins are per-itemid decile bins (10 bins); CE over discrete bins outperforms MSE on heavy-tailed clinical measurements
- **Validation**: 10% of patients held out as a pretrain val set; early stopping on val loss (not train loss)
- **VICReg** (optional, `USE_VICREG=1`): two differently masked views of each batch → VICReg loss on projected `[CLS]` representations; strengthens the stay-level embedding

---

## Fine-tuning — Three Joint Tasks

| Task | Head | Loss | Weight |
|---|---|---|---|
| Hospital mortality | MortalityHead (1 logit) | BCE | 1.0 |
| ICU length of stay | LOSHead (scalar) | MSE | 0.05 |
| Vital forecasting | ForecastHead (7 vitals) | MSE | 0.5 |

**Vital targets**: mean Z-scored value of HR / SpO2 / RR / SBP / DBP / MAP / Temp in the **last 20% of the stay's events** (true future, no leakage).

**Two-stage strategy**:
1. Stage 2a (epoch 1): encoder frozen, heads only at lr=1e-3
2. Stage 2b (epoch 2+): full model at lr=1e-5 with linear warmup + cosine decay

---

## Key Design Decisions

| Decision | Why |
|---|---|
| `time_proj` (Linear, not sinusoidal PE) | ICU events are irregular in time; sinusoidal PE assumes uniform spacing |
| 80/20 temporal split per stay | All stays (short or long) get vital forecast signal; future window is always truly unseen |
| Random window (`window_mode="random"`) during training | Augmentation that also prevents always seeing the same prefix for long stays |
| `[UNK]` token (index 3) | Unknown itemids at inference return UNK, not silent PAD (which would look like padding to attention) |
| Patient-level train/val split | Prevents leakage when one patient has multiple ICU stays |
| VICReg on `[CLS]` | Directly trains the stay-level representation used by all downstream heads |

---

## Pretrained Results (94k stays, Kaggle T4)

| Metric | Value |
|---|---|
| Pretrain best val MEM loss | — (run in progress) |
| Finetune mortality AUROC | 0.867 (50-patient pilot) |

---

## Branch B — Autoregressive Decoder (AR)

A second model trained in parallel for head-to-head comparison.

```
Raw ICU stay (events sorted by charttime)
        │
        ▼  (same ICUEventEmbedding as Branch A — shared class, separate weights)
┌───────────────────────────────────────────────────────┐
│  TransformerEncoder  (Pre-LN, 6 layers, 8 heads)      │
│  ← causal mask: each position attends only to itself  │
│    and earlier positions (left-to-right AR)           │
│  [CLS=BOS] e₁  e₂  ...  eN [PAD] ... [PAD]           │
└──────────┬────────────────────────────────────────────┘
           │  [B, L, 256]
           ▼
┌──────────────────────────────────────────────────────────┐
│  NextEventHead  (predicts event t+1 from hidden state t) │
│    itemid_logits   [B, L, vocab_size]    → CE            │
│    value_logits    [B, L, 10 bins]       → CE            │
│    delta_logits    [B, L, 10 time bins]  → CE            │
└──────────────────────────────────────────────────────────┘
```

**Causal shift alignment** (critical):

| Tensor | Slice | Meaning |
|---|---|---|
| `logits` | `model(inputs)[:, :-1]` | predictions from positions 0…L-2 |
| `itemid_tgt` | `itemid[:, 1:]` | next event's itemid |
| `value_tgt` | `value_bins[:, 1:]` | next event's value bin |
| `delta_tgt` | `delta_bins[:, :-1]` | gap **from** position t **to** t+1 |
| `valid` | `mask[:, :-1] & mask[:, 1:]` | both t and t+1 are real events |

**Time-gap bins** (`time_bin_edges.json`): global (not per-itemid) decile bins computed from all inter-event gaps across all stays. 9 edges → 10 bins.

**AR Pipeline:**

```
1. Synthetic test    python scripts/create_synthetic_stays.py --out dataloader/synthetic_stays
2. Build vocab       python tokenizer/build_vocab.py     (adds time_bin_edges.json)
3. AR pretrain       python training/pretrain_ar.py \
                         --index dataloader/index.csv --data_dir dataloader/all_stays \
                         --vocab tokenizer/vocab.json --norm tokenizer/norm_stats.json \
                         --bins  tokenizer/bin_edges.json \
                         --time_bins tokenizer/time_bin_edges.json \
                         --epochs 10 --batch 32 --out checkpoints/ar
4. Evaluate          python evaluation/eval_ar.py \
                         --ar_ckpt checkpoints/ar/ar_best.pt \
                         --index dataloader/index.csv ... \
                         [--mem_ckpt checkpoints/finetune/best.pt]  # Branch A comparison
```

**Rollout compute cost:** each trajectory makes ~`horizon_hours / mean_gap` forward passes
(≈ 6 h / 1 h ≈ 6 passes for typical ICU stays). With `n_samples=50` that is ~300 forward passes
per stay.  On a T4 this takes ~2 s per stay.  Cap horizon at 6 h (default) to limit error
accumulation from auto-regressive drift over long horizons.

**Zero-shot queries** (no fine-tuning needed):
- `mortality_prob(trajectories, death_itemid_set)` — fraction of runs containing a death event
- `vital_forecast(trajectories, itemid)` — mean predicted value bin for a vital sign
- `event_prob(trajectories, itemid_set, within_hours)` — onset probability within a time window

---

## Repository Structure — Which File Belongs to Which Model

### Branch A — MEM (Masked Event Modeling, bidirectional encoder)

| File | Role |
|---|---|
| `model/embedding.py` | `ICUEventEmbedding` — shared embedding class (also used by Branch B) |
| `model/encoder.py` | `TransformerEncoder` — shared encoder class; Branch A calls it with `attn_mask=None` (full bidirectional attention) |
| `model/heads.py` | `PretrainHead`, `MortalityHead`, `LOSHead`, `ForecastHead`, `ProxyHead` |
| `model/model.py` | **`ICUFoundationModel`** — the Branch A model class |
| `training/pretrain.py` | Branch A pretraining (MEM + optional VICReg + proxy targets) |
| `training/finetune.py` | Branch A supervised fine-tuning (mortality, LOS, vital forecast) |
| `training/utils.py` | `apply_random_mask()`, `augment()`, `vicreg_loss()` — Branch A helpers |
| `checkpoints/mem_best.pt` | Branch A best checkpoint |

### Branch B — AR (Autoregressive Decoder, causal left-to-right)

| File | Role |
|---|---|
| `model/autoregressive.py` | **`ICUAutoregressiveModel`** — the Branch B model class; calls `TransformerEncoder` with a causal triu mask |
| `model/heads.py` | `NextEventHead` — added here (predicts next itemid + value_bin + delta_bin) |
| `training/pretrain_ar.py` | Branch B pretraining (next-event prediction with correct causal shift) |
| `inference/rollout.py` | Ancestral-sampling rollout; `mortality_prob()`, `vital_forecast()`, `event_prob()` |
| `evaluation/eval_ar.py` | Zero-shot eval; new-onset vs. repeat accuracy split; Branch A vs B comparison table |
| `tokenizer/time_bin_edges.json` | 9 global inter-event gap decile edges (10 time bins) — Branch B only |
| `checkpoints/ar/ar_best.pt` | Branch B best checkpoint |

### Shared (both branches read these)

| File | Role |
|---|---|
| `model/config.py` | Single `ModelConfig` dataclass for all hyperparameters |
| `model/embedding.py` | `ICUEventEmbedding` (both branches instantiate their own copy) |
| `model/encoder.py` | `TransformerEncoder` (both branches instantiate their own copy; mask differs) |
| `dataloader/dataloader.py` | `ICUDataset` / `make_dataloader` — `delta_bins` field only present when `time_bin_edges_path` is given (Branch B) |
| `tokenizer/vocab.json` | Token vocabulary |
| `tokenizer/norm_stats.json` | Per-itemid Z-score stats |
| `tokenizer/bin_edges.json` | Per-itemid value decile edges |
| `scripts/create_synthetic_stays.py` | Synthetic data generator for end-to-end testing |

```
model/
  config.py          ← shared hyperparameters
  embedding.py       ← shared embedding (A and B each instantiate their own)
  encoder.py         ← shared encoder (A: attn_mask=None, B: causal triu mask)
  heads.py           ← A: PretrainHead/MortalityHead/LOSHead/ForecastHead/ProxyHead
                       B: NextEventHead
  model.py           ← BRANCH A:  ICUFoundationModel
  autoregressive.py  ← BRANCH B:  ICUAutoregressiveModel

training/
  pretrain.py        ← BRANCH A pretraining  (MEM + VICReg + proxy)
  finetune.py        ← BRANCH A fine-tuning  (mortality + LOS + vitals)
  utils.py           ← BRANCH A helpers      (masking, augmentation, VICReg loss)
  pretrain_ar.py     ← BRANCH B pretraining  (next-event causal LM)

inference/
  rollout.py         ← BRANCH B only  (rollout + zero-shot query helpers)

evaluation/
  eval_ar.py         ← BRANCH B eval (also loads Branch A for comparison)
```

---

## Environment

```
pip install torch wandb duckdb tqdm scikit-learn pandas numpy
```

wandb API key: stored in Kaggle private dataset `seyedhasanmirhoseini/wandb-config` as `key.txt`.  
**Never hardcode the key in any file.**
