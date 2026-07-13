"""
kaggle_run_ar.py — Branch B (AR) pipeline for Kaggle

Branch B: causal autoregressive decoder
  pretrain  →  next-event prediction (itemid + value_bin + delta_bin)
  eval      →  zero-shot mortality AUROC + next-event accuracy (new-onset vs repeat)
               optional head-to-head table vs Branch A if finetune/best.pt exists

Usage:
    exec(open('/kaggle/working/kaggle_run_ar.py').read())

To run Branch A instead, use kaggle_run.py.

Prerequisites:
  1. Attach dataset:  luciadam/icu-datasets  (and optionally mimic-iv-icu-stays)
  2. Add secret:      WANDB_API_KEY
  3. Enable GPU:      T4 x1

Expected runtime: ~1.5-2 hours on T4
  Extract / load stays : ~5 min  (fast if stays are already extracted)
  Build vocab          : ~2 min  (also builds time_bin_edges.json)
  Branch B pretrain    : ~50-90 min
  Branch B eval        : ~10-15 min
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
            cap = torch.cuda.get_device_capability(0)
            print(f"CUDA sm : {cap[0]}.{cap[1]}")
            if cap[0] < 7:
                # P100 / K80 etc. are sm_60 or below — PyTorch 2.2+ dropped support.
                # Install a compatible build so training subprocesses use the GPU.
                print(f"  sm_{cap[0]}{cap[1]} < sm_70: installing P100-compatible PyTorch ...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.0.1+cu118",
                    "--extra-index-url", "https://download.pytorch.org/whl/cu118",
                ], check=True)
                print("  PyTorch 2.0.1+cu118 installed — training will use the P100")
        except Exception as _gpu_e:
            print(f"GPU     : check failed ({_gpu_e})")

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

# Also check the previous kernel run's output (v10 produced a full 94k-stay zip)
_KERNEL_OUT_ZIP = (
    Path("/kaggle/input/icu-foundation-model-full-pipeline/_ds_upload") / STAYS_ZIP_NAME
)
_ZIP_PATH = (
    PRE_EXTRACTED / STAYS_ZIP_NAME if (PRE_EXTRACTED / STAYS_ZIP_NAME).exists()
    else _KERNEL_OUT_ZIP if _KERNEL_OUT_ZIP.exists()
    else None
)

STAYS_DIR.mkdir(parents=True, exist_ok=True)

if _ZIP_PATH is not None:
    # ── Fast path: unzip pre-extracted stays (~5 min) ────────────────────────
    # Raw MIMIC dataset does NOT need to be attached.
    print(f"Pre-extracted stays found at {_ZIP_PATH} — unzipping ...")
    with _zipfile.ZipFile(_ZIP_PATH) as zf:
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
# CELL 6 — Build vocabulary & tokenizer artifacts (including time_bin_edges.json)
# ==============================================================================

subprocess.run(
    [sys.executable, str(WORK / "tokenizer" / "build_vocab.py")],
    env={**os.environ, "PYTHONPATH": str(WORK)},
    check=True,
)

# ==============================================================================
# CELL 7 — Branch B: AR pretraining
# ==============================================================================
# ~50-90 min on T4.
# wandb: https://wandb.ai/seyedhasan-mirhoseini1367-tampere-university/MIMIC-IV-ICU-AR

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
        "--workers",   "2",
        "--wandb_project", "MIMIC-IV-ICU-AR",
    ],
    env={**os.environ,
         "USE_WANDB":  os.environ.get("USE_WANDB", "0"),
         "PYTHONPATH": str(WORK)},
    check=True,
)

# ==============================================================================
# CELL 8 — Branch B: zero-shot evaluation
# ==============================================================================
# ~10-15 min on T4  (200 stays × 50 rollout trajectories).
# Includes head-to-head table vs Branch A if finetune/best.pt exists.

_ar_best  = AR_CKPT_DIR / "ar_best.pt"
_val_idx  = AR_CKPT_DIR / "val_index.csv"
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
        print("Branch A finetune checkpoint found — head-to-head comparison included.")

    subprocess.run(
        _eval_cmd,
        env={**os.environ, "PYTHONPATH": str(WORK)},
        check=True,
    )
else:
    print("AR best checkpoint not found — skipping eval.")

# ==============================================================================
# CELL 9 — Summary
# ==============================================================================

import torch as _torch

print("\n── Branch B (AR) results ────────────────────────────────────────────")
if _ar_best.exists():
    meta = _torch.load(_ar_best, map_location="cpu")
    print(f"  ar/ar_best.pt")
    print(f"    epoch    : {meta.get('epoch', '?')}")
    print(f"    val_loss : {meta.get('val_loss', float('nan')):.4f}")
else:
    print("  ar/ar_best.pt — not found")

print(f"\nDone.")
print(f"wandb : https://wandb.ai/seyedhasan-mirhoseini1367-tampere-university/MIMIC-IV-ICU-AR")
