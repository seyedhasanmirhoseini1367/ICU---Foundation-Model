import json

path = r'D:\datasets\HOSP&ICU-datasets(100000 med-data from 2001-2019）\explore.ipynb'
# Use the actual path with the fullwidth parenthesis character
import os
root = r'D:\datasets\HOSP&ICU-datasets(100000 med-data from 2001-2019）'
# find the actual path
for d in os.listdir(r'D:\datasets'):
    if 'HOSP' in d:
        path = os.path.join(r'D:\datasets', d, 'explore.ipynb')
        break

print('Patching:', path)

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = (
    'top_inputs = (\n'
    '    inputevents["itemid"].value_counts().head(20)\n'
    '    .reset_index()\n'
    ')\n'
    'top_inputs.columns = ["itemid", "n_records"]\n'
    'top_inputs = top_inputs.merge(\n'
    '    d_items[["itemid", "label", "unitname", "category"]],\n'
    '    on="itemid", how="left"\n'
    ')\n'
    '\n'
    'fig, ax = plt.subplots(figsize=(12, 7))\n'
    'ax.barh(\n'
    '    top_inputs["label"].fillna(top_inputs["itemid"].astype(str)),\n'
    '    top_inputs["n_records"], color="#C44E52"\n'
    ')\n'
    'ax.set_xlabel("Count in sample")\n'
    'ax.set_title("Most common input events — MIMIC-IV sample")\n'
    'ax.invert_yaxis()\n'
    'plt.tight_layout()\n'
    'plt.show()\n'
    '\n'
    '# ordercategoryname is a column in inputevents, not in d_items\n'
    'ordercol = next((c for c in inputevents.columns if "ordercategory" in c.lower()), None)\n'
    'if ordercol:\n'
    '    print("Order categories:")\n'
    '    print(inputevents[ordercol].value_counts().head(10))\n'
    'else:\n'
    '    print("Item categories:")\n'
    '    print(top_inputs["category"].value_counts().head(10))\n'
)

fixed = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'ordercategoryname' in src:
            cell['source'] = new_source
            fixed += 1

print('Cells fixed:', fixed)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Done.')
