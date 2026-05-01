import os
import numpy as np
import pandas as pd

RAW_CSV  = '/resnick/groups/CS156b/from_central/data/student_labels/train2023.csv'
SAVE_DIR = '/resnick/groups/CS156b/from_central/2026/JDP/dean_folder'

df = pd.read_csv(RAW_CSV)
print(f"Raw samples: {len(df)}", flush=True)

# drop leftover index columns
df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')])

# drop rows whose images don't exist on this cluster
df = df[~df['Path'].str.startswith('CheXpert')]
print(f"After path filter: {len(df)}", flush=True)

# extract patient id for patient-level split
df['patient_id'] = df['Path'].str.extract(r'(pid\d+)')

# patient-level 90/10 train/val split (seeded for reproducibility)
patients = df['patient_id'].unique()
np.random.seed(42)
np.random.shuffle(patients)
split = int(0.9 * len(patients))
train_patients = set(patients[:split])
val_patients   = set(patients[split:])

train_df = df[df['patient_id'].isin(train_patients)].drop(columns=['patient_id'])
val_df   = df[df['patient_id'].isin(val_patients)].drop(columns=['patient_id'])

# split by view
frontal_train = train_df[train_df['Frontal/Lateral'] == 'Frontal']
lateral_train = train_df[train_df['Frontal/Lateral'] == 'Lateral']
frontal_val   = val_df[val_df['Frontal/Lateral'] == 'Frontal']
lateral_val   = val_df[val_df['Frontal/Lateral'] == 'Lateral']

os.makedirs(SAVE_DIR, exist_ok=True)
frontal_train.to_csv(os.path.join(SAVE_DIR, 'frontal_train_DY.csv'), index=False)
lateral_train.to_csv(os.path.join(SAVE_DIR, 'lateral_train_DY.csv'), index=False)
frontal_val.to_csv(os.path.join(SAVE_DIR,   'frontal_val_DY.csv'),   index=False)
lateral_val.to_csv(os.path.join(SAVE_DIR,   'lateral_val_DY.csv'),   index=False)

print(f"Frontal train: {len(frontal_train)} | val: {len(frontal_val)}", flush=True)
print(f"Lateral train: {len(lateral_train)} | val: {len(lateral_val)}", flush=True)
print("Saved all CSVs to dean_folder.", flush=True)
