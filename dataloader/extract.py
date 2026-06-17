"""
Extract timeline CSVs for N patients and build index.csv
Output structure:
    dataloader/
        index.csv
        all_stays/
            {subject_id}_{stay_id}.csv
"""

import duckdb, pandas as pd, os

ROOT     = r'D:\datasets\HOSP&ICU-datasets(100000 med-data from 2001-2019）\iv\content\mimic-iv-3.1'
OUT_DIR  = r'D:\datasets\HOSP&ICU-datasets(100000 med-data from 2001-2019）\dataloader\all_stays'
IDX_PATH = r'D:\datasets\HOSP&ICU-datasets(100000 med-data from 2001-2019）\dataloader\index.csv'
N_PATIENTS = 10

os.makedirs(OUT_DIR, exist_ok=True)

con = duckdb.connect()
def p(path): return path.replace('\\', '/')

icu  = p(ROOT + r'\icu')
hosp = p(ROOT + r'\hosp')

# ── 0. pick N patients and their stays ─────────────────
print(f"Selecting {N_PATIENTS} patients...", flush=True)
stays = con.execute(f"""
    SELECT s.subject_id, s.stay_id, s.hadm_id,
           s.intime, s.outtime, ROUND(s.los, 2) AS los,
           s.first_careunit
    FROM read_csv_auto('{icu}/icustays.csv') s
    WHERE s.subject_id IN (
        SELECT DISTINCT subject_id
        FROM read_csv_auto('{icu}/icustays.csv')
        LIMIT {N_PATIENTS}
    )
    ORDER BY s.subject_id, s.intime
""").df()
print(f"Found {len(stays)} stays across {stays['subject_id'].nunique()} patients\n")

subject_ids = stays['subject_id'].tolist()
sid_str     = ','.join(map(str, subject_ids))

# ── 1. pull labels from admissions + patients ───────────
print("Fetching labels...", flush=True)
labels = con.execute(f"""
    SELECT a.hadm_id,
           a.hospital_expire_flag,
           a.admission_type,
           a.discharge_location,
           p.anchor_age AS age,
           p.gender
    FROM read_csv_auto('{hosp}/admissions.csv') a
    JOIN read_csv_auto('{hosp}/patients.csv')   p ON a.subject_id = p.subject_id
    WHERE a.subject_id IN ({sid_str})
""").df()

# ── 2. query all 4 event tables once ───────────────────
print("Querying chartevents (large file)...", flush=True)
chart_all = con.execute(f"""
    SELECT c.subject_id, c.stay_id,
           c.charttime AS time, 'CHART' AS source,
           c.itemid, d.label, c.valuenum AS value, c.valueuom AS unit
    FROM read_csv_auto('{icu}/chartevents.csv', ignore_errors=true) c
    JOIN read_csv_auto('{icu}/d_items.csv') d ON c.itemid = d.itemid
    WHERE c.subject_id IN ({sid_str})
      AND c.valuenum IS NOT NULL AND c.warning = 0
""").df()
print(f"  {len(chart_all):,} rows")

print("Querying inputevents...", flush=True)
input_all = con.execute(f"""
    SELECT i.subject_id, i.stay_id,
           i.starttime AS time, 'INPUT' AS source,
           i.itemid, d.label, i.amount AS value, i.amountuom AS unit
    FROM read_csv_auto('{icu}/inputevents.csv', ignore_errors=true) i
    JOIN read_csv_auto('{icu}/d_items.csv') d ON i.itemid = d.itemid
    WHERE i.subject_id IN ({sid_str})
      AND i.statusdescription != 'Rewritten'
""").df()
print(f"  {len(input_all):,} rows")

print("Querying outputevents...", flush=True)
output_all = con.execute(f"""
    SELECT o.subject_id, o.stay_id,
           o.charttime AS time, 'OUTPUT' AS source,
           o.itemid, d.label, o.value AS value, o.valueuom AS unit
    FROM read_csv_auto('{icu}/outputevents.csv', ignore_errors=true) o
    JOIN read_csv_auto('{icu}/d_items.csv') d ON o.itemid = d.itemid
    WHERE o.subject_id IN ({sid_str})
""").df()
print(f"  {len(output_all):,} rows")

print("Querying labevents (large file)...", flush=True)
lab_all = con.execute(f"""
    SELECT l.subject_id, l.hadm_id,
           l.charttime AS time, 'LAB' AS source,
           l.itemid, d.label, l.valuenum AS value, l.valueuom AS unit
    FROM read_csv_auto('{hosp}/labevents.csv', ignore_errors=true) l
    JOIN read_csv_auto('{hosp}/d_labitems.csv') d ON l.itemid = d.itemid
    WHERE l.subject_id IN ({sid_str})
      AND l.valuenum IS NOT NULL
""").df()
print(f"  {len(lab_all):,} rows\n")

# ── 3. build one file per stay + index rows ─────────────
index_rows = []

for _, stay in stays.iterrows():
    subject_id = int(stay['subject_id'])
    stay_id    = int(stay['stay_id'])
    hadm_id    = int(stay['hadm_id'])
    intime     = pd.to_datetime(stay['intime'])
    los        = stay['los']

    # filter each table
    chart  = chart_all [chart_all ['stay_id'] == stay_id].drop(columns=['subject_id','stay_id'])
    inp    = input_all [input_all ['stay_id'] == stay_id].drop(columns=['subject_id','stay_id'])
    out    = output_all[output_all['stay_id'] == stay_id].drop(columns=['subject_id','stay_id'])
    lab    = lab_all   [(lab_all['subject_id'] == subject_id) &
                        (lab_all['hadm_id']    == hadm_id)   ].drop(columns=['subject_id','hadm_id'])

    combined = pd.concat([chart, inp, out, lab], ignore_index=True)
    combined['time']        = pd.to_datetime(combined['time'])
    combined['delta_hours'] = ((combined['time'] - intime)
                               .dt.total_seconds() / 3600).round(2)
    combined = (combined
                .sort_values('time')
                .reset_index(drop=True)
                [['delta_hours','time','source','itemid','label','value','unit']])

    fname = f'{subject_id}_{stay_id}.csv'
    combined.to_csv(os.path.join(OUT_DIR, fname), index=False)

    # get labels for this admission
    lbl = labels[labels['hadm_id'] == hadm_id]
    expire_flag      = int(lbl['hospital_expire_flag'].iloc[0]) if len(lbl) else -1
    age              = int(lbl['age'].iloc[0])              if len(lbl) else -1
    gender           = lbl['gender'].iloc[0]                if len(lbl) else ''
    admission_type   = lbl['admission_type'].iloc[0]        if len(lbl) else ''
    discharge_loc    = lbl['discharge_location'].iloc[0]    if len(lbl) else ''

    index_rows.append({
        'subject_id'         : subject_id,
        'stay_id'            : stay_id,
        'hadm_id'            : hadm_id,
        'intime'             : stay['intime'],
        'los'                : los,
        'first_careunit'     : stay['first_careunit'],
        'age'                : age,
        'gender'             : gender,
        'admission_type'     : admission_type,
        'hospital_expire_flag': expire_flag,
        'discharge_location' : discharge_loc,
        'n_events'           : len(combined),
        'file_path'          : fname,
    })

    print(f"  subject {subject_id} | stay {stay_id} | "
          f"CHART={len(chart)} INPUT={len(inp)} OUTPUT={len(out)} LAB={len(lab)} "
          f"TOTAL={len(combined)} | died={expire_flag}")

# ── 4. save index ───────────────────────────────────────
index = pd.DataFrame(index_rows)
index.to_csv(IDX_PATH, index=False)
print(f"\nIndex saved → {IDX_PATH}")
print(f"Total stays : {len(index)}")
print(f"Total events: {index['n_events'].sum():,}")
print(f"Mortality   : {index['hospital_expire_flag'].sum()} / {len(index)}")
print(f"\n{index[['subject_id','stay_id','los','age','gender','hospital_expire_flag','n_events']].to_string(index=False)}")
