"""
kaggle_run.py - Full ICU Foundation Model pipeline for Kaggle

Copy this file into a Kaggle notebook and run each cell block in order,
OR run the whole file at once:

    exec(open('/kaggle/working/kaggle_run.py').read())

Prerequisites in Kaggle:
  1. Attach dataset:  luciadam/icu-datasets  (Add data -> Your datasets)
  2. Add secret:      WANDB_API_KEY           (Notebook -> Add-ons -> Secrets)
  3. Enable GPU:      Accelerator -> GPU T4 x1

Expected total runtime: ~4-6 hours on T4 GPU
"""

import os, sys, subprocess
from pathlib import Path

# ==============================================================================
# -- CELL 1 . Environment
# ==============================================================================

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "wandb", "duckdb", "tqdm"],
    check=True,
)

import torch
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU     : {props.name}  ({props.total_memory/1e9:.1f} GB VRAM)")

# ==============================================================================
# -- CELL 2 . wandb authentication
# ==============================================================================

import wandb

try:
    from kaggle_secrets import UserSecretsClient
    _key = UserSecretsClient().get_secret("WANDB_API_KEY")
    wandb.login(key=_key)
    os.environ["USE_WANDB"] = "1"
    print("wandb   : authenticated OK")
except Exception as e:
    os.environ["USE_WANDB"] = "0"
    print(f"wandb   : skipped ({e})")

# ==============================================================================
# -- CELL 3 . Clone repo
# ==============================================================================

WORK = Path("/kaggle/working")
os.chdir(WORK)

REPO_URL = "https://github.com/seyedhasanmirhoseini1367/ICU---Foundation-Model.git"

if (WORK / ".git").exists():
    subprocess.run(["git", "pull"], check=True)
    print("Repo    : up to date OK")
else:
    subprocess.run(["git", "init"], check=True)
    subprocess.run(["git", "remote", "add", "origin", REPO_URL], check=True)
    subprocess.run(["git", "pull", "origin", "main"], check=True)
    print("Repo    : cloned OK")

sys.path.insert(0, str(WORK))

# ==============================================================================
# -- CELL 4 . Inspect dataset & detect MIMIC root
# ==============================================================================

DATA_BASE = Path("/kaggle/input/icu-datasets")

print(f"\nDataset root: {DATA_BASE}")
print("Top-level structure:")
for item in sorted(DATA_BASE.iterdir()):
    size = ""
    if item.is_file():
        size = f"  {item.stat().st_size/1e6:.0f} MB"
    kind = "[DIR]" if item.is_dir() else "[FILE]"
    print(f"  {kind} {item.name}{size}")


def find_mimic_root(base: Path) -> Path:
    """Search for the folder that has both icu/ and hosp/ subdirectories."""
    for root, dirs, _ in os.walk(base):
        if "icu" in dirs and "hosp" in dirs:
            return Path(root)
    raise FileNotFoundError(
        f"Could not find icu/ + hosp/ under {base}.\n"
        "Check the dataset structure above and set MIMIC_ROOT manually."
    )


MIMIC_ROOT = find_mimic_root(DATA_BASE)
print(f"\nMIMIC root detected: {MIMIC_ROOT}")

# Verify key files exist
_required = [
    MIMIC_ROOT / "icu" / "icustays.csv",
    MIMIC_ROOT / "icu" / "chartevents.csv",
    MIMIC_ROOT / "hosp" / "admissions.csv",
    MIMIC_ROOT / "hosp" / "patients.csv",
]
for f in _required:
    status = "OK" if f.exists() else "MISSING"
    print(f"  {status}  {f.relative_to(DATA_BASE)}")

# ==============================================================================
# -- CELL 5 . Extract all stays
# ==============================================================================
#
# First run:  ~30-60 min  (reads chartevents.csv which is several GB)
# Resumable:  already-extracted stays are skipped automatically.
#
# To test with 50 patients first, add:  --n 50
# ------------------------------------------------------------------------------

STAYS_DIR = WORK / "dataloader" / "all_stays"
STAYS_DIR.mkdir(parents=True, exist_ok=True)

_already = len(list(STAYS_DIR.glob("*.csv")))
print(f"Already extracted: {_already} stays  (will be skipped)")

subprocess.run([
    sys.executable, str(WORK / "dataloader" / "extract.py"),
    "--root",  str(MIMIC_ROOT),
    "--out",   str(STAYS_DIR),
    "--index", str(WORK / "dataloader" / "index.csv"),
    # "--n", "50",      <- uncomment to test with 50 patients only
], check=True)

# ==============================================================================
# -- CELL 6 . Build vocabulary & normalisation stats
# ==============================================================================

subprocess.run(
    [sys.executable, str(WORK / "tokenizer" / "build_vocab.py")],
    check=True,
)

# ==============================================================================
# -- CELL 7 . Pre-training (Masked Event Modeling)
# ==============================================================================
#
# ~1-2 hours on T4 for 65k stays (50-epoch budget, early stop at ~43).
# Watch live at:  https://wandb.ai/seyedhasan-mirhoseini1367-tampere-university/MIMIC-IV-ICU
# ------------------------------------------------------------------------------

subprocess.run(
    [sys.executable, str(WORK / "training" / "pretrain.py")],
    env={**os.environ, "USE_WANDB": os.environ.get("USE_WANDB", "0")},
    check=True,
)

# ==============================================================================
# -- CELL 8 . Fine-tuning (mortality + LOS)
# ==============================================================================
#
# ~45-90 min on T4.  Best checkpoint -> checkpoints/finetune/best.pt
# ------------------------------------------------------------------------------

subprocess.run(
    [sys.executable, str(WORK / "training" / "finetune.py")],
    env={**os.environ, "USE_WANDB": os.environ.get("USE_WANDB", "0")},
    check=True,
)

# ==============================================================================
# -- CELL 9 . Summary
# ==============================================================================

import json

pretrain_ckpt = WORK / "checkpoints" / "pretrain" / "best.pt"
finetune_ckpt = WORK / "checkpoints" / "finetune" / "best.pt"

for ckpt in [pretrain_ckpt, finetune_ckpt]:
    if ckpt.exists():
        meta = torch.load(ckpt, map_location="cpu")
        print(f"\n{ckpt.parent.name}/best.pt")
        print(f"  epoch : {meta['epoch']}")
        print(f"  loss  : {meta['loss']:.4f}")
    else:
        print(f"\n{ckpt} - not found")

print(f"\nAll done OK")
print(f"wandb dashboard: https://wandb.ai/{os.environ.get('WANDB_ENTITY','(check your account)')}/MIMIC-IV-ICU")
