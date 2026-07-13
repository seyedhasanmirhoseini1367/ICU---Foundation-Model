"""
kaggle_run.py - Full ICU Foundation Model pipeline for Kaggle

Runs both branches end-to-end:
  Branch A (MEM) : bidirectional masked encoder → pretrain → fine-tune
  Branch B (AR)  : causal autoregressive decoder → pretrain → zero-shot eval

Copy this file into a Kaggle notebook and run each cell block in order,
OR run the whole file at once:

    exec(open('/kaggle/working/kaggle_run.py').read())

Prerequisites in Kaggle:
  1. Attach dataset:  luciadam/icu-datasets  (Add data -> Your datasets)
  2. Add secret:      WANDB_API_KEY           (Notebook -> Add-ons -> Secrets)
  3. Enable GPU:      Accelerator -> GPU T4 x1

Expected total runtime: ~3-4 hours on T4 GPU
  Branch A pretrain  : ~50-90 min
  Branch A fine-tune : ~20-40 min
  Branch B pretrain  : ~50-90 min
  Branch B eval      : ~10-15 min
"""

import os, sys, subprocess
from pathlib import Path

# When called from runner.ipynb, install/auth/clone are already done.
_SKIP_INIT = os.environ.get("PIPELINE_SKIP_INIT", "0") == "1"

if not _SKIP_INIT:
    # ===========================================================================
    # -- CELL 1 . Environment
    # ===========================================================================
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

    # ===========================================================================
    # -- CELL 2 . wandb authentication
    # ===========================================================================
    os.environ["WANDB_START_METHOD"] = "thread"
    _key = os.environ.get("WANDB_API_KEY", "").strip()
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

# ==============================================================================
# -- CELL 3 . Clone repo
# ==============================================================================

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
# -- CELL 5 . Extract all stays  (or load from pre-extracted dataset)
# ==============================================================================
#
# If the pre-extracted dataset (seyedhasanmirhoseini/mimic-iv-icu-stays) is
# attached, we unzip it (~5 min) instead of re-extracting (~9 hours).
# On first run: extraction runs, then auto-saves a private Kaggle dataset.
# ------------------------------------------------------------------------------

import json as _json, zipfile as _zipfile, shutil as _shutil

STAYS_DIR      = WORK / "dataloader" / "all_stays"
INDEX_PATH     = WORK / "dataloader" / "index.csv"
PRE_EXTRACTED  = Path("/kaggle/input/mimic-iv-icu-stays")   # our saved dataset
STAYS_ZIP_NAME = "extracted_stays.zip"

STAYS_DIR.mkdir(parents=True, exist_ok=True)


def _save_extracted_as_dataset():
    """Zip the extracted stays and publish as a private Kaggle dataset."""
    zip_path = WORK / STAYS_ZIP_NAME
    print("\nZipping extracted stays for future runs ...")
    with _zipfile.ZipFile(zip_path, "w", _zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in sorted(STAYS_DIR.glob("*.csv")):
            zf.write(f, f"stays/{f.name}")
        if INDEX_PATH.exists():
            zf.write(INDEX_PATH, "index.csv")
    size_mb = zip_path.stat().st_size / 1e6
    print(f"  Zip size : {size_mb:.0f} MB")

    ds_dir = WORK / "_ds_upload"
    ds_dir.mkdir(exist_ok=True)
    # Move (not copy) to avoid needing an extra 1.7 GB of free space
    _shutil.move(str(zip_path), str(ds_dir / STAYS_ZIP_NAME))

    meta = {
        "title"    : "MIMIC-IV ICU Extracted Stays",
        "id"       : "seyedhasanmirhoseini/mimic-iv-icu-stays",
        "licenses" : [{"name": "other"}],
        "isPrivate": True,
    }
    (ds_dir / "dataset-metadata.json").write_text(_json.dumps(meta, indent=2))

    # Try create first; if dataset exists already, push a new version
    r = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "create",
         "-p", str(ds_dir), "--dir-mode", "zip"],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        print("  Saved as NEW dataset : seyedhasanmirhoseini/mimic-iv-icu-stays")
    else:
        r2 = subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "version",
             "-p", str(ds_dir), "-m", "updated extraction", "--dir-mode", "zip"],
            capture_output=True, text=True,
        )
        if r2.returncode == 0:
            print("  Updated dataset version : seyedhasanmirhoseini/mimic-iv-icu-stays")
        else:
            print(f"  Dataset save failed (non-fatal): {r2.stderr[:300]}")


if (PRE_EXTRACTED / STAYS_ZIP_NAME).exists():
    # ── Fast path: unzip pre-extracted stays (~5 min) ────────────────────────
    print("Pre-extracted dataset found — unzipping instead of re-extracting ...")
    with _zipfile.ZipFile(PRE_EXTRACTED / STAYS_ZIP_NAME) as zf:
        members = zf.namelist()
        for member in members:
            if member.startswith("stays/"):
                fname = Path(member).name
                with zf.open(member) as src, open(STAYS_DIR / fname, "wb") as dst:
                    _shutil.copyfileobj(src, dst)
            elif member == "index.csv":
                with zf.open(member) as src, open(INDEX_PATH, "wb") as dst:
                    _shutil.copyfileobj(src, dst)
    n_loaded = len(list(STAYS_DIR.glob("*.csv")))
    print(f"Loaded {n_loaded:,} pre-extracted stays OK")

else:
    # ── Slow path: full extraction (~9 hours) ────────────────────────────────
    _already = len(list(STAYS_DIR.glob("*.csv")))
    print(f"Already extracted: {_already} stays  (will be skipped)")

    N_PATIENTS = 0    # 0 = full run (all 65k patients); set to 50 for a quick test

    cmd = [
        sys.executable, str(WORK / "dataloader" / "extract.py"),
        "--root",  str(MIMIC_ROOT),
        "--out",   str(STAYS_DIR),
        "--index", str(INDEX_PATH),
    ]
    if N_PATIENTS > 0:
        cmd += ["--n", str(N_PATIENTS)]
        print(f"TEST MODE: extracting {N_PATIENTS} patients only")

    # Redirect extraction output to a log file to avoid notebook JSON overflow
    # (65k tqdm lines + 41GB Parquet conversion output would corrupt the notebook)
    _extract_log = WORK / "extraction.log"
    print(f"Extraction log → {_extract_log}")
    print("Progress updates every 60 s ...")
    import time as _time
    with open(_extract_log, "w") as _log:
        _proc = subprocess.Popen(cmd, stdout=_log, stderr=_log)
        while _proc.poll() is None:
            _time.sleep(60)
            # Print a brief progress line from the last checkpoint line in the log
            try:
                _lines = _extract_log.read_text(errors="replace").splitlines()
                _ckpt  = [l for l in _lines if "Checkpoint:" in l or "done:" in l or "converting" in l]
                if _ckpt:
                    print(f"  [{int(_time.time()%86400)//3600:02d}h] {_ckpt[-1].strip()}")
            except Exception:
                pass
        if _proc.returncode != 0:
            raise subprocess.CalledProcessError(_proc.returncode, cmd)

    # Only save dataset on full run (not test)
    if N_PATIENTS == 0:
        _save_extracted_as_dataset()

# ==============================================================================
# -- CELL 6 . Build vocabulary & normalisation stats
# ==============================================================================

subprocess.run(
    [sys.executable, str(WORK / "tokenizer" / "build_vocab.py")],
    env={**os.environ, "PYTHONPATH": str(WORK)},
    check=True,
)

# ==============================================================================
# -- CELL 7 . Pre-training (Masked Event Modeling)  [Branch A]
# ==============================================================================
#
# Set SKIP_BRANCH_A=1 to skip cells 7-8 when Branch A checkpoints already exist.
# ~1-2 hours on T4 for 65k stays (50-epoch budget, early stop at ~43).
# Watch live at:  https://wandb.ai/seyedhasan-mirhoseini1367-tampere-university/MIMIC-IV-ICU
# ------------------------------------------------------------------------------

SKIP_BRANCH_A = os.environ.get("SKIP_BRANCH_A", "0") == "1"

if SKIP_BRANCH_A:
    print("SKIP_BRANCH_A=1 — skipping Branch A pretrain + finetune (using existing checkpoints)")
else:
    subprocess.run(
        [sys.executable, str(WORK / "training" / "pretrain.py")],
        env={**os.environ,
             "USE_WANDB":   os.environ.get("USE_WANDB", "0"),
             "PYTHONPATH":  str(WORK)},
        check=True,
    )

# ==============================================================================
# -- CELL 8 . Fine-tuning (mortality + LOS)  [Branch A]
# ==============================================================================
#
# ~45-90 min on T4.  Best checkpoint -> checkpoints/finetune/best.pt
# ------------------------------------------------------------------------------

if not SKIP_BRANCH_A:
    subprocess.run(
        [sys.executable, str(WORK / "training" / "finetune.py")],
        env={**os.environ,
             "USE_WANDB":   os.environ.get("USE_WANDB", "0"),
             "PYTHONPATH":  str(WORK)},
        check=True,
    )

# ==============================================================================
# -- CELL 9 . Branch B — AR pretraining
# ==============================================================================
#
# Trains ICUAutoregressiveModel with next-event prediction (causal LM).
# Uses the same stays, vocab, and norm stats as Branch A.
# time_bin_edges.json was built in CELL 6 alongside the other tokenizer files.
# ~50-90 min on T4; early stopping with patience=3.
# ------------------------------------------------------------------------------

AR_CKPT_DIR = WORK / "checkpoints" / "ar"

subprocess.run(
    [
        sys.executable, str(WORK / "training" / "pretrain_ar.py"),
        "--index",     str(INDEX_PATH),
        "--data_dir",  str(STAYS_DIR),
        "--vocab",     str(WORK / "tokenizer" / "vocab.json"),
        "--norm",      str(WORK / "tokenizer" / "norm_stats.json"),
        "--bins",      str(WORK / "tokenizer" / "bin_edges.json"),
        "--time_bins", str(WORK / "tokenizer" / "time_bin_edges.json"),
        "--out",       str(AR_CKPT_DIR),
        "--epochs",    "10",
        "--batch",     "32",
        "--lr",        "3e-4",
        "--patience",  "3",
        "--workers",   "2",          # Kaggle runs Linux — parallel data loading is safe
        "--wandb_project", "MIMIC-IV-ICU-AR",
    ],
    env={**os.environ,
         "USE_WANDB":  os.environ.get("USE_WANDB", "0"),
         "PYTHONPATH": str(WORK)},
    check=True,
)

# ==============================================================================
# -- CELL 10 . Branch B — zero-shot evaluation
# ==============================================================================
#
# Evaluates the AR model without any fine-tuning (zero-shot):
#   • Next-event top-1 accuracy  split by new-onset vs. repeat events
#   • Mortality AUROC + Brier score via rollout (n_rollout=50 trajectories)
#   • Head-to-head comparison table with Branch A (MEM) fine-tuned model
#
# Uses the validation split created by pretrain_ar.py so there is no leakage.
# ~10-15 min on T4 (200 stays × 50 rollout trajectories).
# ------------------------------------------------------------------------------

import torch as _torch   # may not be imported yet if SKIP_INIT was set

_ar_best  = AR_CKPT_DIR / "ar_best.pt"
_val_idx  = AR_CKPT_DIR / "val_index.csv"   # written by pretrain_ar.py
_mem_best = WORK / "checkpoints" / "finetune" / "best.pt"

if _ar_best.exists():
    _eval_cmd = [
        sys.executable, str(WORK / "evaluation" / "eval_ar.py"),
        "--ar_ckpt",   str(_ar_best),
        "--index",     str(_val_idx if _val_idx.exists() else INDEX_PATH),
        "--data_dir",  str(STAYS_DIR),
        "--vocab",     str(WORK / "tokenizer" / "vocab.json"),
        "--norm",      str(WORK / "tokenizer" / "norm_stats.json"),
        "--bins",      str(WORK / "tokenizer" / "bin_edges.json"),
        "--time_bins", str(WORK / "tokenizer" / "time_bin_edges.json"),
        "--n_rollout", "50",
        "--horizon",   "6.0",
        "--max_stays", "200",
    ]
    if _mem_best.exists():
        _eval_cmd += ["--mem_ckpt", str(_mem_best)]

    subprocess.run(
        _eval_cmd,
        env={**os.environ, "PYTHONPATH": str(WORK)},
        check=True,
    )
else:
    print("AR best checkpoint not found — skipping eval_ar.py")

# ==============================================================================
# -- CELL 11 . Summary
# ==============================================================================

pretrain_ckpt = WORK / "checkpoints" / "pretrain" / "best.pt"
finetune_ckpt = WORK / "checkpoints" / "finetune" / "best.pt"
ar_ckpt       = AR_CKPT_DIR / "ar_best.pt"

print("\n── Branch A (MEM) ───────────────────────────────────────────────────")
for label, ckpt in [("pretrain/best.pt", pretrain_ckpt), ("finetune/best.pt", finetune_ckpt)]:
    if ckpt.exists():
        meta = _torch.load(ckpt, map_location="cpu")
        print(f"  {label}")
        print(f"    epoch : {meta.get('epoch', '?')}")
        print(f"    loss  : {meta.get('loss', float('nan')):.4f}")
    else:
        print(f"  {label} — not found")

print("\n── Branch B (AR) ────────────────────────────────────────────────────")
if ar_ckpt.exists():
    meta = _torch.load(ar_ckpt, map_location="cpu")
    print(f"  ar/ar_best.pt")
    print(f"    epoch    : {meta.get('epoch', '?')}")
    print(f"    val_loss : {meta.get('val_loss', float('nan')):.4f}")
else:
    print("  ar/ar_best.pt — not found")

print(f"\nAll done.")
print(f"Branch A wandb : https://wandb.ai/seyedhasan-mirhoseini1367-tampere-university/MIMIC-IV-ICU")
print(f"Branch B wandb : https://wandb.ai/seyedhasan-mirhoseini1367-tampere-university/MIMIC-IV-ICU-AR")
