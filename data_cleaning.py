import pandas as pd
import numpy as np
 
DATA_DIR = "/resnick/groups/CS156b/from_central/data"
TRAIN_CSV = f"{DATA_DIR}/student_labels/train2023.csv"
 
LABEL_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Pneumonia", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]
 
df = pd.read_csv(TRAIN_CSV)
print(f"Raw samples: {len(df)}")
 
# drop leftover index columns
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
 
# drop the one row with a mismatched path format
df = df[~df["Path"].str.startswith("CheXpert")]

# frontal views only
df = df[df["Frontal/Lateral"] == "Frontal"]

print(f"After cleanup: {len(df)}")
 
df["patient_id"] = df["Path"].str.extract(r"(pid\d+)")
df["study_id"] = df["Path"].str.extract(r"(study\d+)")
df["view_type"] = df["Path"].str.extract(r"view\d+_([a-zA-Z]+)")
 
# print summary
print(f"\nPatients: {df['patient_id'].nunique()}")
print(f"Studies:  {df[['patient_id', 'study_id']].drop_duplicates().shape[0]}")
 
print("\n── View distribution ──")
print(df["view_type"].value_counts(dropna=False))
 
print("\n── Sex distribution ──")
print(df["Sex"].value_counts(dropna=False))
 
print("\n── Age summary ──")
print(df["Age"].describe())
 
print("\n── Label distribution ──")
summary = pd.DataFrame({
    "positive": df[LABEL_COLS].eq(1).sum(),
    "negative": df[LABEL_COLS].eq(0).sum(),
    "uncertain": df[LABEL_COLS].eq(-1).sum(),
    "missing":  df[LABEL_COLS].isna().sum(),
})
print(summary.sort_values("positive", ascending=False))

patients = df["patient_id"].unique()
np.random.seed(42)
np.random.shuffle(patients)
split = int(0.9 * len(patients))

train_patients = set(patients[:split])
val_patients = set(patients[split:])

train_df = df[df["patient_id"].isin(train_patients)]
val_df = df[df["patient_id"].isin(val_patients)]

train_df.to_csv("train_clean.csv", index=False)
val_df.to_csv("val_clean.csv", index=False)
print(f"\nTrain: {len(train_df)} images ({len(train_patients)} patients)")
print(f"Val:   {len(val_df)} images ({len(val_patients)} patients)")
