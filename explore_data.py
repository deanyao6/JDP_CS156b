# import os
# import pandas as pd

# train_csv = "/resnick/groups/CS156b/from_central/data/student_labels/train2023.csv"

# df = pd.read_csv(train_csv)

# print(df.columns.tolist())
# print("num rows:", len(df))
# print(df.head())


import pandas as pd

train_csv = "/resnick/groups/CS156b/from_central/data/student_labels/train2023.csv"
df = pd.read_csv(train_csv)

label_cols = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Pneumonia", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]

df["patient_id"] = df["Path"].str.extract(r"(pid\d+)")
df["study_id"] = df["Path"].str.extract(r"(study\d+)")
df["view_type"] = df["Path"].str.extract(r"view\d+_([a-zA-Z]+)")

print("num images:", len(df))
print("num patients:", df["patient_id"].nunique())
print("num studies:", df[["patient_id", "study_id"]].drop_duplicates().shape[0])
print("\nSex counts:\n", df["Sex"].value_counts(dropna=False))
print("\nView counts:\n", df["view_type"].value_counts(dropna=False))
print("\nAge summary:\n", df["Age"].describe())

summary = pd.DataFrame({
    "positive": df[label_cols].eq(1).sum(),
    "negative": df[label_cols].eq(0).sum(),
    "missing": df[label_cols].isna().sum()
})
print("\nLabel summary:\n", summary.sort_values("positive", ascending=False))