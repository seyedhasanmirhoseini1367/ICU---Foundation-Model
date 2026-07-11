"""
dataloader/extract.py

Extracts per-stay timeline CSVs from raw MIMIC-IV tables and writes index.csv.

Each stay → one CSV:  dataloader/all_stays/{subject_id}_{stay_id}.csv
Columns: delta_hours, time, source, itemid, label, value, unit

Output:
    dataloader/
        index.csv               -- one row per stay with labels
        all_stays/
            {subject_id}_{stay_id}.csv

Usage:
    # All patients (Kaggle / full run):
    python dataloader/extract.py --root /kaggle/input/datasets/luciadam/icu-datasets

    # Quick smoke-test with 50 patients:
    python dataloader/extract.py --root /path/to/mimic --n 50

    # Override output location:
    python dataloader/extract.py --root /path/to/mimic --out /kaggle/working/dataloader/all_stays
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# Patients per memory batch. 1000 uses ~1-2 GB peak, well within Kaggle limits.
BATCH_SIZE = 1000


# -- CLI -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root", required=True,
        help="Path that contains icu/ and hosp/ (auto-searched recursively).",
    )
    p.add_argument(
        "--out", default=None,
        help="Directory for per-stay CSVs. Default: <repo>/dataloader/all_stays",
    )
    p.add_argument(
        "--index", default=None,
        help="Path for index.csv. Default: <repo>/dataloader/index.csv",
    )
    p.add_argument(
        "--n", type=int, default=0,
        help="Number of PATIENTS to extract (0 = all). Use 50 for a quick test.",
    )
    return p.parse_args()


# -- Path helpers --------------------------------------------------------------

def find_mimic_root(base: str) -> Path:
    """Walk from base until we find a directory that has both icu/ and hosp/."""
    for root, dirs, _ in os.walk(base):
        if "icu" in dirs and "hosp" in dirs:
            return Path(root)
    raise FileNotFoundError(
        f"Could not find a folder containing both icu/ and hosp/ under '{base}'.\n"
        "Pass the correct --root path."
    )


def p(path: Path) -> str:
    """DuckDB needs forward-slash paths even on Windows."""
    return str(path).replace("\\", "/")


# -- DuckDB queries ------------------------------------------------------------

def load_stays(con, icu: Path, n_patients: int) -> pd.DataFrame:
    limit_clause = f"LIMIT {n_patients}" if n_patients > 0 else ""
    return con.execute(f"""
        SELECT s.subject_id, s.stay_id, s.hadm_id,
               s.intime, s.outtime, ROUND(s.los, 2) AS los,
               s.first_careunit
        FROM read_csv_auto('{p(icu)}/icustays.csv') s
        WHERE s.subject_id IN (
            SELECT DISTINCT subject_id
            FROM read_csv_auto('{p(icu)}/icustays.csv')
            ORDER BY subject_id
            {limit_clause}
        )
        ORDER BY s.subject_id, s.intime
    """).df()


def load_labels(con, hosp: Path, sid_str: str) -> pd.DataFrame:
    return con.execute(f"""
        SELECT a.hadm_id,
               a.hospital_expire_flag,
               a.admission_type,
               a.discharge_location,
               p.anchor_age AS age,
               p.gender
        FROM read_csv_auto('{p(hosp)}/admissions.csv') a
        JOIN read_csv_auto('{p(hosp)}/patients.csv') p ON a.subject_id = p.subject_id
        WHERE a.subject_id IN ({sid_str})
    """).df()


def preload_parquet(con, icu: Path, hosp: Path) -> dict:
    """
    Convert the four large event CSVs to Parquet once, then register them as
    DuckDB views.  Subsequent batch queries read from fast columnar Parquet
    instead of rescanning the raw CSV each time — cuts extraction from ~9h to ~1h.
    Returns a dict mapping table name → path string for use in queries.
    """
    work = Path("/kaggle/working")
    if not work.exists():          # local dev fallback
        work = icu.parent.parent / "parquet_cache"
    work.mkdir(parents=True, exist_ok=True)

    sources = {
        "chartevents" : (icu  / "chartevents.csv",  work / "chartevents.parquet"),
        "inputevents" : (icu  / "inputevents.csv",   work / "inputevents.parquet"),
        "outputevents": (icu  / "outputevents.csv",  work / "outputevents.parquet"),
        "labevents"   : (hosp / "labevents.csv",     work / "labevents.parquet"),
    }

    # Use most available RAM for the one-time conversion pass
    con.execute("PRAGMA memory_limit='20GB'")

    paths = {}
    for name, (csv_path, pq_path) in sources.items():
        if pq_path.exists():
            size_mb = pq_path.stat().st_size / 1e6
            print(f"  {name}: cached ({size_mb:.0f} MB)", flush=True)
        elif csv_path.exists():
            size_gb = csv_path.stat().st_size / 1e9
            print(f"  {name}: converting {size_gb:.1f} GB CSV → Parquet ...", flush=True)
            con.execute(f"""
                COPY (SELECT * FROM read_csv_auto('{p(csv_path)}', ignore_errors=true))
                TO '{p(pq_path)}' (FORMAT PARQUET, COMPRESSION 'zstd')
            """)
            size_mb = pq_path.stat().st_size / 1e6
            print(f"    done: {size_mb:.0f} MB", flush=True)
        else:
            print(f"  {name}: CSV not found — skipping", flush=True)
            paths[name] = None
            continue
        paths[name] = p(pq_path)

    # Back to a conservative limit for per-batch queries
    con.execute("PRAGMA memory_limit='6GB'")
    return paths


def load_events_batch(con, pq: dict, icu: Path, hosp: Path, sid_str: str) -> tuple:
    """Load all 4 event tables for a batch of subjects from Parquet (fast path)
    or CSV fallback when a Parquet file is unavailable."""

    def src(name, csv_path):
        if pq.get(name):
            return f"read_parquet('{pq[name]}')"
        return f"read_csv_auto('{p(csv_path)}', ignore_errors=true)"

    chart = con.execute(f"""
        SELECT c.subject_id, c.stay_id,
               c.charttime AS time, 'CHART' AS source,
               c.itemid, d.label, c.valuenum AS value, c.valueuom AS unit
        FROM {src('chartevents', icu / 'chartevents.csv')} c
        JOIN read_csv_auto('{p(icu)}/d_items.csv') d ON c.itemid = d.itemid
        WHERE c.subject_id IN ({sid_str})
          AND c.valuenum IS NOT NULL AND c.warning = 0
    """).df()

    inp = con.execute(f"""
        SELECT i.subject_id, i.stay_id,
               i.starttime AS time, 'INPUT' AS source,
               i.itemid, d.label, i.amount AS value, i.amountuom AS unit
        FROM {src('inputevents', icu / 'inputevents.csv')} i
        JOIN read_csv_auto('{p(icu)}/d_items.csv') d ON i.itemid = d.itemid
        WHERE i.subject_id IN ({sid_str})
          AND i.statusdescription != 'Rewritten'
    """).df()

    out = con.execute(f"""
        SELECT o.subject_id, o.stay_id,
               o.charttime AS time, 'OUTPUT' AS source,
               o.itemid, d.label, o.value AS value, o.valueuom AS unit
        FROM {src('outputevents', icu / 'outputevents.csv')} o
        JOIN read_csv_auto('{p(icu)}/d_items.csv') d ON o.itemid = d.itemid
        WHERE o.subject_id IN ({sid_str})
    """).df()

    lab = con.execute(f"""
        SELECT l.subject_id, l.hadm_id,
               l.charttime AS time, 'LAB' AS source,
               l.itemid, d.label, l.valuenum AS value, l.valueuom AS unit
        FROM {src('labevents', hosp / 'labevents.csv')} l
        JOIN read_csv_auto('{p(hosp)}/d_labitems.csv') d ON l.itemid = d.itemid
        WHERE l.subject_id IN ({sid_str})
          AND l.valuenum IS NOT NULL
    """).df()

    return chart, inp, out, lab


# -- Per-stay assembly ---------------------------------------------------------

def build_stay_timeline(
    stay: pd.Series,
    chart: pd.DataFrame,
    inp:   pd.DataFrame,
    out:   pd.DataFrame,
    lab:   pd.DataFrame,
) -> pd.DataFrame:
    subject_id = int(stay["subject_id"])
    stay_id    = int(stay["stay_id"])
    hadm_id    = int(stay["hadm_id"])
    intime     = pd.to_datetime(stay["intime"])

    c = chart[chart["stay_id"] == stay_id].drop(columns=["subject_id", "stay_id"])
    i = inp[inp["stay_id"]     == stay_id].drop(columns=["subject_id", "stay_id"])
    o = out[out["stay_id"]     == stay_id].drop(columns=["subject_id", "stay_id"])
    l = lab[(lab["subject_id"] == subject_id) &
            (lab["hadm_id"]    == hadm_id)].drop(columns=["subject_id", "hadm_id"])

    combined = pd.concat([c, i, o, l], ignore_index=True)
    if combined.empty:
        return combined

    combined["time"]        = pd.to_datetime(combined["time"])
    combined["delta_hours"] = ((combined["time"] - intime)
                               .dt.total_seconds() / 3600).round(2)
    return (combined
            .sort_values("time")
            .reset_index(drop=True)
            [["delta_hours", "time", "source", "itemid", "label", "value", "unit"]])


# -- Entry point ---------------------------------------------------------------

def main():
    args = parse_args()

    mimic_root = find_mimic_root(args.root)
    icu  = mimic_root / "icu"
    hosp = mimic_root / "hosp"
    print(f"MIMIC root : {mimic_root}")
    print(f"ICU tables : {icu}")
    print(f"HOSP tables: {hosp}")

    repo_root = Path(__file__).resolve().parent.parent
    out_dir   = Path(args.out)   if args.out   else repo_root / "dataloader" / "all_stays"
    idx_path  = Path(args.index) if args.index else repo_root / "dataloader" / "index.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_label = f"{args.n} patients" if args.n > 0 else "ALL patients"
    print(f"Extracting : {n_label}  ->  {out_dir}\n")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    # -- Convert large CSVs to Parquet (one-time ~10 min, then cached) ---------
    print("Caching event tables as Parquet for fast batch access ...", flush=True)
    pq = preload_parquet(con, icu, hosp)
    print()

    # -- Load stays (small, fine in memory) ------------------------------------
    print("Loading ICU stays ...", flush=True)
    stays   = load_stays(con, icu, args.n)
    n_stays = len(stays)
    print(f"  {n_stays} stays across {stays['subject_id'].nunique()} patients\n")

    all_subjects = stays["subject_id"].unique().tolist()
    sid_str_all  = ",".join(map(str, all_subjects))

    # -- Load labels once (small) ----------------------------------------------
    print("Loading labels ...", flush=True)
    labels = load_labels(con, hosp, sid_str_all)
    print(f"  {len(labels)} admissions\n")

    # -- Batch loop: BATCH_SIZE patients at a time ----------------------------
    batches   = [all_subjects[i:i+BATCH_SIZE]
                 for i in range(0, len(all_subjects), BATCH_SIZE)]
    n_batches = len(batches)

    index_rows = []
    skipped    = 0
    extracted  = 0

    for batch_idx, batch_subjects in enumerate(batches):
        batch_set     = set(batch_subjects)
        sid_str_batch = ",".join(map(str, batch_subjects))
        batch_stays   = stays[stays["subject_id"].isin(batch_set)]

        print(f"\nBatch {batch_idx+1}/{n_batches}  "
              f"({len(batch_subjects)} patients, {len(batch_stays)} stays) ...",
              flush=True)
        print("  Loading events ...", flush=True)

        chart, inp, out_ev, lab = load_events_batch(con, pq, icu, hosp, sid_str_batch)
        print(f"  chart={len(chart):,}  inp={len(inp):,}  "
              f"out={len(out_ev):,}  lab={len(lab):,}", flush=True)

        for _, stay in tqdm(batch_stays.iterrows(),
                            total=len(batch_stays),
                            desc=f"  Batch {batch_idx+1}",
                            leave=False):
            subject_id = int(stay["subject_id"])
            stay_id    = int(stay["stay_id"])
            hadm_id    = int(stay["hadm_id"])
            fname      = f"{subject_id}_{stay_id}.csv"
            fpath      = out_dir / fname

            lbl = labels[labels["hadm_id"] == hadm_id]

            # Resumable: skip stays already on disk
            if fpath.exists():
                index_rows.append({
                    "subject_id"           : subject_id,
                    "stay_id"              : stay_id,
                    "hadm_id"              : hadm_id,
                    "intime"               : stay["intime"],
                    "los"                  : stay["los"],
                    "first_careunit"       : stay["first_careunit"],
                    "age"                  : int(lbl["age"].iloc[0])    if len(lbl) else -1,
                    "gender"               : lbl["gender"].iloc[0]      if len(lbl) else "",
                    "admission_type"       : lbl["admission_type"].iloc[0] if len(lbl) else "",
                    "hospital_expire_flag" : int(lbl["hospital_expire_flag"].iloc[0]) if len(lbl) else -1,
                    "discharge_location"   : lbl["discharge_location"].iloc[0] if len(lbl) else "",
                    "n_events"             : sum(1 for _ in open(fpath)) - 1,
                    "file_path"            : fname,
                })
                continue

            try:
                timeline = build_stay_timeline(stay, chart, inp, out_ev, lab)
            except Exception as e:
                tqdm.write(f"  stay {stay_id}: {e}")
                skipped += 1
                continue

            if timeline.empty:
                skipped += 1
                continue

            timeline.to_csv(fpath, index=False)
            extracted += 1

            index_rows.append({
                "subject_id"           : subject_id,
                "stay_id"              : stay_id,
                "hadm_id"              : hadm_id,
                "intime"               : stay["intime"],
                "los"                  : stay["los"],
                "first_careunit"       : stay["first_careunit"],
                "age"                  : int(lbl["age"].iloc[0])    if len(lbl) else -1,
                "gender"               : lbl["gender"].iloc[0]      if len(lbl) else "",
                "admission_type"       : lbl["admission_type"].iloc[0] if len(lbl) else "",
                "hospital_expire_flag" : int(lbl["hospital_expire_flag"].iloc[0]) if len(lbl) else -1,
                "discharge_location"   : lbl["discharge_location"].iloc[0] if len(lbl) else "",
                "n_events"             : len(timeline),
                "file_path"            : fname,
            })

        # Free batch memory before loading the next batch
        del chart, inp, out_ev, lab
        gc.collect()

        # Checkpoint index after each batch (crash-safe)
        pd.DataFrame(index_rows).to_csv(idx_path, index=False)
        print(f"  Checkpoint: {extracted} extracted, {skipped} skipped so far")

    # -- Final index -----------------------------------------------------------
    index = pd.DataFrame(index_rows)
    index.to_csv(idx_path, index=False)

    # Free Parquet cache to reclaim disk space before the zip step
    freed = 0
    for pq_file in Path("/kaggle/working").glob("*.parquet"):
        freed += pq_file.stat().st_size
        pq_file.unlink()
    if freed:
        print(f"\nFreed {freed/1e9:.1f} GB Parquet cache from disk.")

    print(f"\nDone.")
    print(f"  Stays extracted : {extracted}")
    print(f"  Stays skipped   : {skipped}")
    if len(index) > 0:
        print(f"  Mortality rate  : {index['hospital_expire_flag'].sum()} / {len(index)}"
              f" ({100*index['hospital_expire_flag'].mean():.1f}%)")
        print(f"  Total events    : {index['n_events'].sum():,}")
    print(f"  Index saved     : {idx_path}")


if __name__ == "__main__":
    main()
