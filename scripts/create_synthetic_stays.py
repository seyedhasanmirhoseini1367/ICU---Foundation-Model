"""
scripts/create_synthetic_stays.py

Generates synthetic ICU stay CSVs and an index.csv for end-to-end pipeline
testing WITHOUT requiring real MIMIC-IV data.

Output layout:
    <out_dir>/
        index.csv          — one row per stay (file_path, subject_id,
                             hospital_expire_flag, los, age, gender)
        stay_0000.csv      — synthetic stay events
        ...
        stay_N.csv

Each stay CSV has columns:
    delta_hours   — hours since ICU admission (monotonically increasing)
    source        — "CHART" / "INPUT" / "OUTPUT" / "LAB"
    itemid        — integer MIMIC-IV item identifier
    value         — numeric measurement (float)

Run:
    python scripts/create_synthetic_stays.py \
        --out dataloader/synthetic_stays --n 20 --seed 42
"""

import argparse
import random
import json
import csv
import numpy as np
from pathlib import Path


# A small set of plausible MIMIC-IV itemids (vitals + labs)
ITEM_POOL = [
    (220045, "CHART", 70.0,  15.0),   # Heart rate
    (220277, "CHART", 96.0,   2.0),   # SpO2
    (220210, "CHART", 16.0,   3.0),   # Respiratory rate
    (220179, "CHART", 120.0, 20.0),   # Systolic BP
    (220180, "CHART", 75.0,  15.0),   # Diastolic BP
    (220052, "CHART", 90.0,  15.0),   # MAP
    (223762, "CHART", 37.0,   0.5),   # Temperature
    (220615, "LAB",   1.0,   0.4),    # Creatinine
    (227456, "LAB",   4.5,   1.5),    # Potassium
    (220224, "LAB",   95.0,   5.0),   # PaO2
    (220235, "LAB",   40.0,   5.0),   # PaCO2
    (220274, "LAB",   7.38,  0.05),   # pH
    (225668, "LAB",   1.2,   0.8),    # Lactate
    (221906, "INPUT", 0.1,   0.05),   # Norepinephrine
]

SOURCES = ["CHART", "INPUT", "OUTPUT", "LAB"]
SOURCE_INTERVALS = {
    "CHART" : 1.0,   # typical hourly charting
    "INPUT" : 4.0,   # q4h
    "OUTPUT": 2.0,   # q2h
    "LAB"   : 8.0,   # q8h
}


def generate_stay(
    stay_id : int,
    rng     : random.Random,
    np_rng  : np.random.Generator,
    min_dur : float = 12.0,   # min ICU stay duration (hours)
    max_dur : float = 120.0,  # max ICU stay duration
) -> list[dict]:
    """Generate one synthetic stay as a list of event dicts."""
    duration = rng.uniform(min_dur, max_dur)
    events   = []

    # For each item, sample observations at roughly its expected interval
    for itemid, source, mean_val, std_val in ITEM_POOL:
        interval  = SOURCE_INTERVALS.get(source, 1.0) * (0.7 + rng.random() * 0.6)
        t         = rng.uniform(0, interval)   # first observation offset
        while t < duration:
            val = float(np_rng.normal(mean_val, std_val))
            events.append({
                "delta_hours": round(t, 4),
                "source"     : source,
                "itemid"     : itemid,
                "value"      : round(val, 4),
            })
            t += interval * (0.8 + rng.random() * 0.4)

    # Sort by time
    events.sort(key=lambda e: e["delta_hours"])
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",  default="dataloader/synthetic_stays",
                        help="Output directory for CSV files")
    parser.add_argument("--n",    type=int, default=20, help="Number of stays")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng    = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    index_rows = []
    fieldnames = ["delta_hours", "source", "itemid", "value"]

    for i in range(args.n):
        stay_id  = i
        events   = generate_stay(stay_id, rng, np_rng)
        filename = f"stay_{i:04d}.csv"
        filepath = out_dir / filename

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(events)

        los               = round(events[-1]["delta_hours"] / 24.0, 4) if events else 1.0
        expire_flag       = int(rng.random() < 0.12)   # ~12% mortality
        age               = rng.randint(18, 90)
        gender            = rng.choice(["M", "F"])

        index_rows.append({
            "file_path"            : filename,
            "subject_id"           : 1000 + i,
            "hospital_expire_flag" : expire_flag,
            "los"                  : los,
            "age"                  : age,
            "gender"               : gender,
        })

    index_path = out_dir / "index.csv"
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file_path","subject_id","hospital_expire_flag","los","age","gender"]
        )
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"Created {args.n} synthetic stays in {out_dir}")
    print(f"Index   : {index_path}")
    total_events = sum(
        sum(1 for _ in open(out_dir / r["file_path"])) - 1
        for r in index_rows
    )
    print(f"Total events across all stays: {total_events:,}")


if __name__ == "__main__":
    main()
