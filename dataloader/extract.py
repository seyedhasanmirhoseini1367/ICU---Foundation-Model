"""
dataloader/extract.py

Extracts per-stay timeline CSVs from raw MIMIC-IV tables and writes index.csv.

Each stay → one CSV:  dataloader/all_stays/{subject_id}_{stay_id}.csv
Columns: delta_hours, time, source, itemid, label, value, unit

Output:
    dataloader/
        index.csv               — one row per stay with labels
        all_stays/
            {subject_id}_{stay_id}.csv

Usage:
    # All patients (Kaggle / full run):
    python dataloader/extract.py --root /kaggle/input/icu-datasets

    # Quick smoke-test with 50 patients:
    python dataloader/extract.py --root /path/to/mimic --n 50

    # Override output location:
    python dataloader/extract.py --root /path/to/mimic --out /kaggle/working/dataloader/all_stays
"""

import argparse
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


# ── CLI ───────────────────────────────────────────────────────────────────────

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


# ── Path helpers ──────────────────────────────────────────────────────────────

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


# ── DuckDB queries ────────────────────────────────────────────────────────────

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


def load_events(con, icu: Path, hosp: Path, sid_str: str) -> tuple:
    """Load all 4 event tables for the selected subjects in one pass each."""
    print("  chartevents …", flush=True)
    chart = con.execute(f"""
        SELECT c.subject_id, c.stay_id,
               c.charttime AS time, 'CHART' AS source,
               c.itemid, d.label, c.valuenum AS value, c.valueuom AS unit
        FROM read_csv_auto('{p(icu)}/chartevents.csv', ignore_errors=true) c
        JOIN read_csv_auto('{p(icu)}/d_items.csv') d ON c.itemid = d.itemid
        WHERE c.subject_id IN ({sid_str})
          AND c.valuenum IS NOT NULL AND c.warning = 0
    """).df()
    print(f"    {len(chart):,} rows")

    print("  inputevents …", flush=True)
    inp = con.execute(f"""
        SELECT i.subject_id, i.stay_id,
               i.starttime AS time, 'INPUT' AS source,
               i.itemid, d.label, i.amount AS value, i.amountuom AS unit
        FROM read_csv_auto('{p(icu)}/inputevents.csv', ignore_errors=true) i
        JOIN read_csv_auto('{p(icu)}/d_items.csv') d ON i.itemid = d.itemid
        WHERE i.subject_id IN ({sid_str})
          AND i.statusdescription != 'Rewritten'
    """).df()
    print(f"    {len(inp):,} rows")

    print("  outputevents …", flush=True)
    out = con.execute(f"""
        SELECT o.subject_id, o.stay_id,
               o.charttime AS time, 'OUTPUT' AS source,
               o.itemid, d.label, o.value AS value, o.valueuom AS unit
        FROM read_csv_auto('{p(icu)}/outputevents.csv', ignore_errors=true) o
        JOIN read_csv_auto('{p(icu)}/d_items.csv') d ON o.itemid = d.itemid
        WHERE o.subject_id IN ({sid_str})
    """).df()
    print(f"    {len(out):,} rows")

    print("  labevents …", flush=True)
    lab = con.execute(f"""
        SELECT l.subject_id, l.hadm_id,
               l.charttime AS time, 'LAB' AS source,
               l.itemid, d.label, l.valuenum AS value, l.valueuom AS unit
        FROM read_csv_auto('{p(hosp)}/labevents.csv', ignore_errors=true) l
        JOIN read_csv_auto('{p(hosp)}/d_labitems.csv') d ON l.itemid = d.itemid
        WHERE l.subject_id IN ({sid_str})
          AND l.valuenum IS NOT NULL
    """).df()
    print(f"    {len(lab):,} rows")

    return chart, inp, out, lab


# ── Per-stay assembly ─────────────────────────────────────────────────────────

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


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve MIMIC root (handles nested structures like iv/content/mimic-iv-3.1/)
    mimic_root = find_mimic_root(args.root)
    icu  = mimic_root / "icu"
    hosp = mimic_root / "hosp"
    print(f"MIMIC root : {mimic_root}")
    print(f"ICU tables : {icu}")
    print(f"HOSP tables: {hosp}")

    # Output paths — default to repo-relative locations
    repo_root = Path(__file__).resolve().parent.parent
    out_dir   = Path(args.out)   if args.out   else repo_root / "dataloader" / "all_stays"
    idx_path  = Path(args.index) if args.index else repo_root / "dataloader" / "index.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_label = f"{args.n} patients" if args.n > 0 else "ALL patients"
    print(f"Extracting : {n_label}  →  {out_dir}\n")

    con = duckdb.connect()

    # ── Load stays ────────────────────────────────────────────────────────────
    print("Loading ICU stays …", flush=True)
    stays   = load_stays(con, icu, args.n)
    n_stays = len(stays)
    print(f"  {n_stays} stays across {stays['subject_id'].nunique()} patients\n")

    sid_str = ",".join(map(str, stays["subject_id"].unique().tolist()))

    # ── Load labels once ──────────────────────────────────────────────────────
    print("Loading labels …", flush=True)
    labels = load_labels(con, hosp, sid_str)
    print(f"  {len(labels)} admissions\n")

    # ── Load all 4 event tables once (memory-efficient bulk load) ─────────────
    print("Loading event tables (this takes a few minutes) …")
    chart, inp, out_ev, lab = load_events(con, icu, hosp, sid_str)
    print()

    # ── Build one CSV per stay ────────────────────────────────────────────────
    index_rows = []
    skipped    = 0

    for _, stay in tqdm(stays.iterrows(), total=n_stays, desc="Extracting stays"):
        subject_id = int(stay["subject_id"])
        stay_id    = int(stay["stay_id"])
        hadm_id    = int(stay["hadm_id"])
        fname      = f"{subject_id}_{stay_id}.csv"
        fpath      = out_dir / fname

        # Skip stays already on disk (allows resuming interrupted runs)
        if fpath.exists():
            lbl = labels[labels["hadm_id"] == hadm_id]
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
            tqdm.write(f"  ✗ stay {stay_id}: {e}")
            skipped += 1
            continue

        if timeline.empty:
            skipped += 1
            continue

        timeline.to_csv(fpath, index=False)

        lbl             = labels[labels["hadm_id"] == hadm_id]
        expire_flag     = int(lbl["hospital_expire_flag"].iloc[0]) if len(lbl) else -1
        age             = int(lbl["age"].iloc[0])             if len(lbl) else -1
        gender          = lbl["gender"].iloc[0]               if len(lbl) else ""
        admission_type  = lbl["admission_type"].iloc[0]       if len(lbl) else ""
        discharge_loc   = lbl["discharge_location"].iloc[0]   if len(lbl) else ""

        index_rows.append({
            "subject_id"           : subject_id,
            "stay_id"              : stay_id,
            "hadm_id"              : hadm_id,
            "intime"               : stay["intime"],
            "los"                  : stay["los"],
            "first_careunit"       : stay["first_careunit"],
            "age"                  : age,
            "gender"               : gender,
            "admission_type"       : admission_type,
            "hospital_expire_flag" : expire_flag,
            "discharge_location"   : discharge_loc,
            "n_events"             : len(timeline),
            "file_path"            : fname,
        })

    # ── Write index ───────────────────────────────────────────────────────────
    index = pd.DataFrame(index_rows)
    index.to_csv(idx_path, index=False)

    print(f"\nDone.")
    print(f"  Stays extracted : {len(index_rows)}")
    print(f"  Stays skipped   : {skipped}")
    print(f"  Mortality rate  : {index['hospital_expire_flag'].sum()} / {len(index)}"
          f" ({100*index['hospital_expire_flag'].mean():.1f}%)")
    print(f"  Total events    : {index['n_events'].sum():,}")
    print(f"  Index saved     : {idx_path}")


if __name__ == "__main__":
    main()
