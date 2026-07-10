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
    _shutil.copy(zip_path, ds_dir / STAYS_ZIP_NAME)

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

    subprocess.run(cmd, check=True)

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
# -- CELL 7 . Pre-training (Masked Event Modeling)
# ==============================================================================
#
# ~1-2 hours on T4 for 65k stays (50-epoch budget, early stop at ~43).
# Watch live at:  https://wandb.ai/seyedhasan-mirhoseini1367-tampere-university/MIMIC-IV-ICU
# ------------------------------------------------------------------------------

subprocess.run(
    [sys.executable, str(WORK / "training" / "pretrain.py")],
    env={**os.environ,
         "USE_WANDB":   os.environ.get("USE_WANDB", "0"),
         "PYTHONPATH":  str(WORK)},
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
    env={**os.environ,
         "USE_WANDB":   os.environ.get("USE_WANDB", "0"),
         "PYTHONPATH":  str(WORK)},
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
