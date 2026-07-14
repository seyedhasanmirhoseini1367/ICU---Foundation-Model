"""
tpu/kaggle_run_ar_tpu.py — Branch B (AR) pipeline for Kaggle TPU

Differences from kaggle_run_ar.py:
  - No CUDA/P100 compatibility check
  - Calls tpu/pretrain_ar_tpu.py (8-core xmp.spawn)
  - Per-core batch = 64  →  effective batch = 512
  - No slow-path extraction fallback: requires mimic-iv-icu-stays attached

Prerequisites (add via Kaggle UI — API cannot attach private datasets):
  1. seyedhasanmirhoseini/mimic-iv-icu-stays   (fast path stays zip)
  2. seyedhasanmirhoseini/icu-ar-checkpoint     (optional resume)
  3. seyedhasanmirhoseini/wandb-config          (optional wandb key)

Kernel metadata: enable_tpu=true, enable_gpu=false
"""

import os, sys, subprocess, re as _re
from pathlib import Path

_SKIP_INIT = os.environ.get("PIPELINE_SKIP_INIT", "0") == "1"

if not _SKIP_INIT:
    # ── CELL 1 — Environment ──────────────────────────────────────────────────
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "wandb", "duckdb", "tqdm"],
        check=True,
    )

    try:
        import torch_xla.core.xla_model as _xm
        print(f"TPU     : {_xm.xla_device()}")
    except Exception as _e:
        print(f"TPU     : not available ({_e})")

    # ── CELL 2 — wandb authentication ─────────────────────────────────────────
    os.environ["WANDB_START_METHOD"] = "thread"
    _key = os.environ.get("WANDB_API_KEY", "").strip()

    if not _key:
        _key_file = Path("/kaggle/input/wandb-config/key.txt")
        if _key_file.exists():
            _key = _key_file.read_text().strip()
            print("wandb   : key loaded from wandb-config dataset")

    if not _key:
        try:
            from kaggle_secrets import UserSecretsClient
            _key = UserSecretsClient().get_secret("WANDB_API_KEY")
        except Exception as _e:
            print(f"wandb   : UserSecretsClient failed ({_e})")

    if _key:
        os.environ["WANDB_API_KEY"] = _key
        os.environ["USE_WANDB"]     = "1"
        print("wandb   : API key set OK")
    else:
        os.environ["USE_WANDB"] = "0"
        print("wandb   : skipped")

# ── CELL 3 — Clone / update repo ──────────────────────────────────────────────

WORK = Path("/kaggle/working")
os.chdir(WORK)

if not _SKIP_INIT:
    REPO_URL = "https://github.com/seyedhasanmirhoseini1367/ICU---Foundation-Model.git"
    if (WORK / ".git").exists():
        subprocess.run(["git", "pull", "origin", "main"], check=True)
        print("Repo    : up to date OK")
    else:
        subprocess.run(["git", "init"], check=True)
        subprocess.run(["git", "remote", "add", "origin", REPO_URL], check=True)
        subprocess.run(["git", "pull", "origin", "main"], check=True)
        print("Repo    : cloned OK")

sys.path.insert(0, str(WORK))

# ── CELL 4 — Load stays (fast path only — TPU time is precious) ───────────────

import zipfile as _zipfile, shutil as _shutil

KAGGLE_INPUT   = Path("/kaggle/input")
STAYS_DIR      = WORK / "dataloader" / "all_stays"
INDEX_PATH     = WORK / "dataloader" / "index.csv"
PRE_EXTRACTED  = Path("/kaggle/input/mimic-iv-icu-stays")
STAYS_ZIP_NAME = "extracted_stays.zip"

STAYS_DIR.mkdir(parents=True, exist_ok=True)

_ZIP_PATH = PRE_EXTRACTED / STAYS_ZIP_NAME if (PRE_EXTRACTED / STAYS_ZIP_NAME).exists() else None

if _ZIP_PATH is None:
    raise RuntimeError(
        "seyedhasanmirhoseini/mimic-iv-icu-stays not found.\n"
        "Attach it via Notebook → Add Data before running on TPU.\n"
        "(Slow extraction is disabled in the TPU runner — TPU time is limited.)"
    )

print(f"Pre-extracted stays found — unzipping ...")
with _zipfile.ZipFile(_ZIP_PATH) as zf:
    for member in zf.namelist():
        if member.startswith("stays/"):
            fname = Path(member).name
            with zf.open(member) as src, open(STAYS_DIR / fname, "wb") as dst:
                _shutil.copyfileobj(src, dst)
        elif member == "index.csv":
            with zf.open(member) as src, open(INDEX_PATH, "wb") as dst:
                _shutil.copyfileobj(src, dst)

print(f"Loaded {len(list(STAYS_DIR.glob('*.csv'))):,} stays OK")

# ── CELL 5 — Build vocabulary ─────────────────────────────────────────────────

subprocess.run(
    [sys.executable, str(WORK / "tokenizer" / "build_vocab.py")],
    env={**os.environ, "PYTHONPATH": str(WORK)},
    check=True,
)

# ── CELL 6 — Auto-detect resume checkpoint ────────────────────────────────────

_RESUME_DATASET = Path("/kaggle/input/icu-ar-checkpoint")
_resume_args    = []
_start_epoch    = 0

if _RESUME_DATASET.exists():
    _ckpts = sorted(_RESUME_DATASET.glob("ar_epoch*.pt"))
    if _ckpts:
        _ckpt_path = _ckpts[-1]
        _m = _re.search(r"ar_epoch(\d+)\.pt", _ckpt_path.name)
        if _m:
            _start_epoch = int(_m.group(1))
        _resume_args = ["--resume", str(_ckpt_path)]
        print(f"Resume   : found {_ckpt_path.name}  (epoch {_start_epoch})")

_total_epochs = _start_epoch + 10

# ── CELL 7 — Branch B AR pretraining (8-core TPU) ────────────────────────────
# Effective batch = 64 per core × 8 cores = 512
# ~5-8x faster than single T4 GPU

AR_CKPT_DIR = WORK / "checkpoints" / "ar"

subprocess.run(
    [
        sys.executable, str(WORK / "tpu" / "pretrain_ar_tpu.py"),
        "--index",     str(INDEX_PATH),
        "--data_dir",  str(STAYS_DIR),
        "--vocab",     str(WORK / "tokenizer" / "vocab.json"),
        "--norm",      str(WORK / "tokenizer" / "norm_stats.json"),
        "--bins",      str(WORK / "tokenizer" / "bin_edges.json"),
        "--time_bins", str(WORK / "tokenizer" / "time_bin_edges.json"),
        "--out",       str(AR_CKPT_DIR),
        "--epochs",    str(_total_epochs),
        "--batch",     "64",
        "--lr",        "3e-4",
        "--wandb_project", "MIMIC-IV-ICU-AR-TPU",
    ] + _resume_args,
    env={**os.environ, "PYTHONPATH": str(WORK)},
    check=True,
)

# ── CELL 8 — Summary ──────────────────────────────────────────────────────────

import torch as _torch

_ar_best = AR_CKPT_DIR / "ar_best.pt"
print("\n── Branch B (AR/TPU) results ─────────────────────────────────────────")
if _ar_best.exists():
    print(f"  ar/ar_best.pt  ✓")
else:
    print("  ar/ar_best.pt — not found")

print(f"\nDone.")
print(f"wandb : https://wandb.ai/seyedhasan-mirhoseini1367-tampere-university/MIMIC-IV-ICU-AR-TPU")
