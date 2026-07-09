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
    try:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU     : {props.name}  ({props.total_memory/1e9:.1f} GB VRAM)")
    except Exception:
        print("GPU     : available (properties unavailable)")

# ==============================================================================
# -- CELL 2 . wandb authentication
# ==============================================================================

# Must be set BEFORE importing wandb to avoid subprocess service startup failure
os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_SERVICE_WAIT"] = "300"

import wandb

try:
    from kaggle_secrets import UserSecretsClient
    _key = UserSecretsClient().get_secret("WANDB_API_KEY")
    wandb.login(key=_key, relogin=True)
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

KAGGLE_INPUT = Path("/kaggle/input")

print(f"\nSearching for MIMIC data under {KAGGLE_INPUT}")
print("Available dataset directories:")
available = []
for item in sorted(KAGGLE_INPUT.iterdir()):
    if item.is_dir():
        available.append(item)
        print(f"  [DIR] {item.name}")
        for sub in sorted(item.iterdir()):
            print(f"        {'[DIR]' if sub.is_dir() else '[FILE]'} {sub.name}")

if not available:
    raise RuntimeError(
        "No datasets found under /kaggle/input. "
        "Please attach the dataset luciadam/icu-datasets via "
        "Notebook -> Add Data -> Datasets -> luciadam/icu-datasets"
    )


def find_mimic_root(base: Path) -> Path:
    """Search for the folder that contains both icu/ and hosp/ subdirectories."""
    for root, dirs, _ in os.walk(base):
        if "icu" in dirs and "hosp" in dirs:
            return Path(root)
    raise FileNotFoundError(
        f"Could not find icu/ + hosp/ under {base}.\n"
        "Directories found: " + ", ".join(str(d) for d in available)
    )


MIMIC_ROOT = find_mimic_root(KAGGLE_INPUT)
print(f"\nMIMIC root detected: {MIMIC_ROOT}")

# Verify key files exist
_required = [
    MIMIC_ROOT / "icu" / "icustays.csv",
    MIMIC_ROOT / "icu" / "chartevents.csv",
    MIMIC_ROOT / "hosp" / "admissions.csv",
    MIMIC_ROOT / "hosp" / "patients.csv",
]
all_ok = True
for f in _required:
    status = "OK" if f.exists() else "MISSING"
    if status == "MISSING":
        all_ok = False
    print(f"  {status}  {f.relative_to(MIMIC_ROOT)}")

if not all_ok:
    raise FileNotFoundError("Some required MIMIC files are missing. Check paths above.")

# ==============================================================================
# -- CELL 5 . Extract all stays
# ==============================================================================
#
# First run:  ~30-60 min  (reads chartevents.csv which is several GB)
# Resumable:  already-extracted stays are skipped automatically.
#
# To test with 50 patients first, uncomment: "--n", "50"
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
    # "--n", "50",
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
print(f"wandb: https://wandb.ai/seyedhasan-mirhoseini1367-tampere-university/MIMIC-IV-ICU")
