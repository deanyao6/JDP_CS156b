import pandas as pd


TRAIN_PATH = "/resnick/groups/CS156b/from_central/data/student_labels/train2023.csv"

# Load CSV
df = pd.read_csv(TRAIN_PATH)

# Remove all unnecessary index columns
# Drop first two columns directly
df = df.iloc[:, 2:]

# Drop fully empty rows
df = df.dropna(how="all")

# drop the one row with a mismatched path format
df = df[~df["Path"].str.startswith("CheXpert")]

df["patient_id"] = df["Path"].str.extract(r"(pid\d+)")
df["study_id"] = df["Path"].str.extract(r"(study\d+)")
df["view_type"] = df["Path"].str.extract(r"view\d+_([a-zA-Z]+)")

# Separate datasets
frontal_df = df[df["Frontal/Lateral"] == "Frontal"].copy()
lateral_df = df[df["Frontal/Lateral"] == "Lateral"].copy()

frontal_df = frontal_df.sample(frac=1, random_state=42).reset_index(drop=True)
lateral_df = lateral_df.sample(frac=1, random_state=42).reset_index(drop=True)


# Save clean CSVs WITHOUT extra index columns
frontal_df.to_csv("frontal_dataset.csv", index=False)
lateral_df.to_csv("lateral_dataset.csv", index=False)

print("Cleaned and saved:")
print("Frontal rows:", len(frontal_df))
print("Lateral rows:", len(lateral_df))

print(frontal_df.head(10))
print(frontal_df.head(10))