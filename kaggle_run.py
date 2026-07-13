"""
kaggle_run.py — Branch A (MEM) pipeline for Kaggle

Branch A: bidirectional masked encoder
  pretrain  →  Masked Event Modeling (MEM) + optional VICReg + proxy targets
  finetune  →  mortality (BCE) + LOS (MSE) + vital forecast (MSE)

Usage:
    exec(open('/kaggle/working/kaggle_run.py').read())

To run Branch B instead, use kaggle_run_ar.py.

Prerequisites:
  1. Attach dataset:  luciadam/icu-datasets
  2. Add secret:      WANDB_API_KEY
  3. Enable GPU:      T4 x1

Expected runtime: ~1.5-2.5 hours on T4
  Extract / load stays : ~5 min  (or ~9 h on first run without pre-extracted dataset)
  Build vocab          : ~2 min
  Branch A pretrain    : ~50-90 min
  Branch A fine-tune   : ~20-40 min
"""

import os, sys, subprocess
from pathlib import Path

_SKIP_INIT = os.environ.get("PIPELINE_SKIP_INIT", "0") == "1"

if not _SKIP_INIT:
    # ==========================================================================
    # CELL 1 — Environment
    # ==========================================================================
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

    # ==========================================================================
    # CELL 2 — wandb authentication
    # ==========================================================================
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
# CELL 3 — Clone / update repo
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
# CELL 4 & 5 — Load stays (pre-extracted fast path) or extract from raw MIMIC
# ==============================================================================

import json as _json, zipfile as _zipfile, shutil as _shutil

KAGGLE_INPUT   = Path("/kaggle/input")
STAYS_DIR      = WORK / "dataloader" / "all_stays"
INDEX_PATH     = WORK / "dataloader" / "index.csv"
PRE_EXTRACTED  = Path("/kaggle/input/mimic-iv-icu-stays")
STAYS_ZIP_NAME = "extracted_stays.zip"

STAYS_DIR.mkdir(parents=True, exist_ok=True)

if (PRE_EXTRACTED / STAYS_ZIP_NAME).exists():
    # ── Fast path: unzip pre-extracted stays (~5 min) ────────────────────────
    # Raw MIMIC dataset does NOT need to be attached.
    print("Pre-extracted dataset found — unzipping ...")
    with _zipfile.ZipFile(PRE_EXTRACTED / STAYS_ZIP_NAME) as zf:
        for member in zf.namelist():
            if member.startswith("stays/"):
                fname = Path(member).name
                with zf.open(member) as src, open(STAYS_DIR / fname, "wb") as dst:
                    _shutil.copyfileobj(src, dst)
            elif member == "index.csv":
                with zf.open(member) as src, open(INDEX_PATH, "wb") as dst:
                    _shutil.copyfileobj(src, dst)
    print(f"Loaded {len(list(STAYS_DIR.glob('*.csv'))):,} pre-extracted stays OK")

else:
    # ── Slow path: full extraction from raw MIMIC (~9 hours) ─────────────────
    # Requires luciadam/icu-datasets attached as an additional dataset.
    print(f"\nSearching for MIMIC data under {KAGGLE_INPUT}")
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
            "Attach luciadam/icu-datasets via Notebook -> Add Data."
        )

    def find_mimic_root(base: Path) -> Path:
        for root, dirs, _ in os.walk(base):
            if "icu" in dirs and "hosp" in dirs:
                return Path(root)
        raise FileNotFoundError(
            f"Could not find icu/ + hosp/ under {base}.\n"
            "Directories found: " + ", ".join(str(d) for d in available)
        )

    MIMIC_ROOT = find_mimic_root(KAGGLE_INPUT)
    print(f"\nMIMIC root: {MIMIC_ROOT}")

    for f in [
        MIMIC_ROOT / "icu"  / "icustays.csv",
        MIMIC_ROOT / "icu"  / "chartevents.csv",
        MIMIC_ROOT / "hosp" / "admissions.csv",
        MIMIC_ROOT / "hosp" / "patients.csv",
    ]:
        status = "OK" if f.exists() else "MISSING"
        print(f"  {status}  {f.relative_to(MIMIC_ROOT)}")
        if status == "MISSING":
            raise FileNotFoundError(f"Required MIMIC file missing: {f}")

    N_PATIENTS = 0
    cmd = [
        sys.executable, str(WORK / "dataloader" / "extract.py"),
        "--root",  str(MIMIC_ROOT),
        "--out",   str(STAYS_DIR),
        "--index", str(INDEX_PATH),
    ]
    if N_PATIENTS > 0:
        cmd += ["--n", str(N_PATIENTS)]
        print(f"TEST MODE: {N_PATIENTS} patients")

    import time as _time
    _log_path = WORK / "extraction.log"
    print(f"Extraction log → {_log_path}")
    with open(_log_path, "w") as _log:
        _proc = subprocess.Popen(cmd, stdout=_log, stderr=_log)
        while _proc.poll() is None:
            _time.sleep(60)
            try:
                _lines = _log_path.read_text(errors="replace").splitlines()
                _ckpt = [l for l in _lines if any(k in l for k in ("Checkpoint:", "done:", "converting"))]
                if _ckpt:
                    print(f"  [{int(_time.time()%86400)//3600:02d}h] {_ckpt[-1].strip()}")
            except Exception:
                pass
        if _proc.returncode != 0:
            raise subprocess.CalledProcessError(_proc.returncode, cmd)

    # Zip and save as private Kaggle dataset for future fast-path runs
    zip_path = WORK / STAYS_ZIP_NAME
    print("\nZipping extracted stays for future runs ...")
    with _zipfile.ZipFile(zip_path, "w", _zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in sorted(STAYS_DIR.glob("*.csv")):
            zf.write(f, f"stays/{f.name}")
        if INDEX_PATH.exists():
            zf.write(INDEX_PATH, "index.csv")
    print(f"  Zip size : {zip_path.stat().st_size/1e6:.0f} MB")

    ds_dir = WORK / "_ds_upload"
    ds_dir.mkdir(exist_ok=True)
    _shutil.move(str(zip_path), str(ds_dir / STAYS_ZIP_NAME))
    (ds_dir / "dataset-metadata.json").write_text(_json.dumps({
        "title": "MIMIC-IV ICU Extracted Stays",
        "id": "seyedhasanmirhoseini/mimic-iv-icu-stays",
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }, indent=2))

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
        msg = "Updated" if r2.returncode == 0 else f"Failed: {r2.stderr[:200]}"
        print(f"  {msg}")

# ==============================================================================
# CELL 6 — Build vocabulary & tokenizer artifacts
# ==============================================================================

subprocess.run(
    [sys.executable, str(WORK / "tokenizer" / "build_vocab.py")],
    env={**os.environ, "PYTHONPATH": str(WORK)},
    check=True,
)

# ==============================================================================
# CELL 7 — Branch A: MEM pretraining
# ==============================================================================
# ~50-90 min on T4.
# wandb: https://wandb.ai/seyedhasan-mirhoseini1367-tampere-university/MIMIC-IV-ICU

subprocess.run(
    [sys.executable, str(WORK / "training" / "pretrain.py")],
    env={**os.environ,
         "USE_WANDB":  os.environ.get("USE_WANDB", "0"),
         "PYTHONPATH": str(WORK)},
    check=True,
)

# ==============================================================================
# CELL 8 — Branch A: fine-tuning (mortality + LOS + vital forecast)
# ==============================================================================
# ~20-40 min on T4.  Best checkpoint → checkpoints/finetune/best.pt

subprocess.run(
    [sys.executable, str(WORK / "training" / "finetune.py")],
    env={**os.environ,
         "USE_WANDB":  os.environ.get("USE_WANDB", "0"),
         "PYTHONPATH": str(WORK)},
    check=True,
)

# ==============================================================================
# CELL 9 — Summary
# ==============================================================================

import torch as _torch

print("\n── Branch A (MEM) results ───────────────────────────────────────────")
for label, path in [
    ("pretrain/best.pt", WORK / "checkpoints" / "pretrain" / "best.pt"),
    ("finetune/best.pt", WORK / "checkpoints" / "finetune" / "best.pt"),
]:
    if path.exists():
        meta = _torch.load(path, map_location="cpu")
        print(f"  {label}")
        print(f"    epoch : {meta.get('epoch', '?')}")
        print(f"    loss  : {meta.get('loss', float('nan')):.4f}")
    else:
        print(f"  {label} — not found")

print(f"\nDone.")
print(f"wandb : https://wandb.ai/seyedhasan-mirhoseini1367-tampere-university/MIMIC-IV-ICU")
