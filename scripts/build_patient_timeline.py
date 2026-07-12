import duckdb, pandas as pd, os

ROOT    = r'D:\datasets\HOSP&ICU-datasets(100000 med-data from 2001-2019）\iv\content\mimic-iv-3.1'
SAMPLE  = r'D:\datasets\HOSP&ICU-datasets(100000 med-data from 2001-2019）\sample'
SUBJECT = 10016742

con = duckdb.connect()
def p(path): return path.replace('\\', '/')

icu  = p(ROOT + r'\icu')
hosp = p(ROOT + r'\hosp')

# ── 0. get all ICU stays for this patient ───────────────
stays = con.execute(f"""
    SELECT stay_id, hadm_id, intime, outtime, ROUND(los,2) AS los_days
    FROM read_csv_auto('{icu}/icustays.csv')
    WHERE subject_id = {SUBJECT}
    ORDER BY intime
""").df()
print("ICU stays:\n", stays.to_string(index=False), "\n")

# ── query large files once, filter to subject ───────────
print("Querying chartevents...", flush=True)
chart_all = con.execute(f"""
    SELECT c.stay_id, c.charttime AS time, 'CHART' AS source,
           c.itemid, d.label, c.valuenum AS value, c.valueuom AS unit
    FROM read_csv_auto('{icu}/chartevents.csv', ignore_errors=true) c
    JOIN read_csv_auto('{icu}/d_items.csv') d ON c.itemid = d.itemid
    WHERE c.subject_id = {SUBJECT}
      AND c.valuenum IS NOT NULL AND c.warning = 0
""").df()

print("Querying inputevents...", flush=True)
input_all = con.execute(f"""
    SELECT i.stay_id, i.starttime AS time, 'INPUT' AS source,
           i.itemid, d.label, i.amount AS value, i.amountuom AS unit
    FROM read_csv_auto('{icu}/inputevents.csv', ignore_errors=true) i
    JOIN read_csv_auto('{icu}/d_items.csv') d ON i.itemid = d.itemid
    WHERE i.subject_id = {SUBJECT}
      AND i.statusdescription != 'Rewritten'
""").df()

print("Querying outputevents...", flush=True)
output_all = con.execute(f"""
    SELECT o.stay_id, o.charttime AS time, 'OUTPUT' AS source,
           o.itemid, d.label, o.value AS value, o.valueuom AS unit
    FROM read_csv_auto('{icu}/outputevents.csv', ignore_errors=true) o
    JOIN read_csv_auto('{icu}/d_items.csv') d ON o.itemid = d.itemid
    WHERE o.subject_id = {SUBJECT}
""").df()

print("Querying labevents...", flush=True)
lab_all = con.execute(f"""
    SELECT l.hadm_id, l.charttime AS time, 'LAB' AS source,
           l.itemid, d.label, l.valuenum AS value, l.valueuom AS unit
    FROM read_csv_auto('{hosp}/labevents.csv', ignore_errors=true) l
    JOIN read_csv_auto('{hosp}/d_labitems.csv') d ON l.itemid = d.itemid
    WHERE l.subject_id = {SUBJECT}
      AND l.valuenum IS NOT NULL
""").df()

# ── build one file per stay ─────────────────────────────
os.makedirs(os.path.join(SAMPLE, f'patient_{SUBJECT}'), exist_ok=True)

for _, stay in stays.iterrows():
    stay_id  = int(stay['stay_id'])
    hadm_id  = int(stay['hadm_id'])
    intime   = pd.to_datetime(stay['intime'])
    outtime  = pd.to_datetime(stay['outtime'])
    los      = stay['los_days']

    # filter each table to this stay
    chart  = chart_all [chart_all ['stay_id'] == stay_id].copy()
    inp    = input_all [input_all ['stay_id'] == stay_id].copy()
    out    = output_all[output_all['stay_id'] == stay_id].copy()
    lab    = lab_all   [lab_all   ['hadm_id'] == hadm_id].copy()

    # drop the join key before merging
    for df in [chart, inp, out]:  df.drop(columns='stay_id', inplace=True)
    lab.drop(columns='hadm_id', inplace=True)

    combined = pd.concat([chart, inp, out, lab], ignore_index=True)
    combined['time'] = pd.to_datetime(combined['time'])
    combined = combined.sort_values('time').reset_index(drop=True)
    combined['delta_hours'] = ((combined['time'] - intime)
                               .dt.total_seconds() / 3600).round(2)
    combined = combined[['delta_hours','time','source','itemid','label','value','unit']]

    fname = f'patient_{SUBJECT}_stay_{stay_id}.csv'
    fpath = os.path.join(SAMPLE, f'patient_{SUBJECT}', fname)
    combined.to_csv(fpath, index=False)

    print(f"\nStay {stay_id}  (hadm {hadm_id})  {stay['intime']}  LOS={los}d")
    print(f"  CHART={len(chart)}  INPUT={len(inp)}  OUTPUT={len(out)}  LAB={len(lab)}  TOTAL={len(combined)}")
    print(f"  Saved → sample/patient_{SUBJECT}/{fname}")
    print(f"  First 5 events:")
    print(combined.head(5).to_string(index=False))
