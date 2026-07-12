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

## Repository Structure

```
model/          config.py, embedding.py, encoder.py, heads.py, model.py
dataloader/     extract.py, dataloader.py
tokenizer/      build_vocab.py  →  vocab.json, norm_stats.json, bin_edges.json
training/       pretrain.py, finetune.py, utils.py
scripts/        build_patient_timeline.py, fix_cell.py
kaggle_run.py   full pipeline runner for Kaggle
runner.ipynb    Kaggle notebook wrapper
```

---

## Environment

```
pip install torch wandb duckdb tqdm scikit-learn pandas numpy
```

wandb API key: stored in Kaggle private dataset `seyedhasanmirhoseini/wandb-config` as `key.txt`.  
**Never hardcode the key in any file.**
