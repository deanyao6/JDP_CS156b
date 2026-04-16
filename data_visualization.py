import os
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
csv_path = "/resnick/groups/CS156b/from_central/data/student_labels/train2023.csv"

# ----------------------------
# LOAD DATA (NO MODIFICATIONS)
# ----------------------------
df = pd.read_csv(csv_path)

# Drop unnamed columns only (does NOT affect data meaning)
df = df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], errors="ignore")

# Labels
labels = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Pneumonia",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# ----------------------------
# BASIC INFO
# ----------------------------
print("Shape:", df.shape)
print("\nMissing values per column:")
print(df.isnull().sum())

# ----------------------------
# CREATE OUTPUT FOLDER
# ----------------------------
output_dir = "data_plots_raw"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# 1. VALUE DISTRIBUTION PER LABEL (-1, 0, 1, NaN)
# ----------------------------
for label in labels:
    counts = df[label].value_counts(dropna=False)

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title(f"{label} Value Distribution (-1 / 0 / 1 / NaN)")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()

    safe_name = label.lower().replace(" ", "_")
    plt.savefig(os.path.join(output_dir, f"{safe_name}_value_dist.png"))
    plt.show()

# ----------------------------
# 2. TOTAL COUNTS OF EACH VALUE TYPE
# ----------------------------
summary = {}
for label in labels:
    summary[label] = {
        "1 (positive)": (df[label] == 1).sum(),
        "0 (negative)": (df[label] == 0).sum(),
        "-1 (uncertain)": (df[label] == -1).sum(),
        "NaN (missing)": df[label].isna().sum(),
    }

summary_df = pd.DataFrame(summary).T
print("\nDetailed Label Breakdown:")
print(summary_df)

# ----------------------------
# 3. AGE DISTRIBUTION
# ----------------------------
plt.figure(figsize=(8, 5))
df["Age"].hist(bins=50)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "age_distribution.png"))
plt.show()

# ----------------------------
# 4. SEX DISTRIBUTION
# ----------------------------
plt.figure(figsize=(6, 4))
df["Sex"].value_counts(dropna=False).plot(kind="bar")
plt.title("Sex Distribution (including NaN)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "sex_distribution.png"))
plt.show()

# ----------------------------
# 5. AP vs PA DISTRIBUTION
# ----------------------------
if "AP/PA" in df.columns:
    plt.figure(figsize=(6, 4))
    df["AP/PA"].value_counts(dropna=False).plot(kind="bar")
    plt.title("AP vs PA Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ap_pa_distribution.png"))
    plt.show()

# ----------------------------
# 6. FRONTAL vs LATERAL
# ----------------------------
if "Frontal/Lateral" in df.columns:
    plt.figure(figsize=(6, 4))
    df["Frontal/Lateral"].value_counts(dropna=False).plot(kind="bar")
    plt.title("Frontal vs Lateral Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "frontal_lateral_distribution.png"))
    plt.show()

# ----------------------------
# 7. HOW MANY POSITIVE LABELS (ONLY COUNT 1s)
# ----------------------------
positive_counts = (df[labels] == 1).sum(axis=1)

plt.figure(figsize=(8, 5))
positive_counts.hist(bins=10)
plt.title("Number of Positive Findings per Image")
plt.xlabel("# of positive labels (value = 1)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "positive_findings_distribution.png"))
plt.show()

# ----------------------------
# 8. UNCERTAIN LABEL ANALYSIS (-1)
# ----------------------------
uncertain_counts = (df[labels] == -1).sum(axis=1)

plt.figure(figsize=(8, 5))
uncertain_counts.hist(bins=10)
plt.title("Number of Uncertain Labels (-1) per Image")
plt.xlabel("# of uncertain labels")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "uncertain_distribution.png"))
plt.show()

# ----------------------------
# 9. MISSING LABEL ANALYSIS (NaN)
# ----------------------------
missing_counts = df[labels].isna().sum(axis=1)

plt.figure(figsize=(8, 5))
missing_counts.hist(bins=10)
plt.title("Number of Missing Labels (NaN) per Image")
plt.xlabel("# of NaN labels")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "missing_distribution.png"))
plt.show()

# ----------------------------
# 10. POSITIVE RATE BY AGE (ONLY USING TRUE POSITIVES)
# ----------------------------
age_bins = [0, 20, 40, 60, 80, 100]
df["age_bin"] = pd.cut(df["Age"], bins=age_bins)

for label in labels:
    grouped = df.groupby("age_bin")[label].apply(lambda x: (x == 1).mean())

    plt.figure(figsize=(8, 5))
    grouped.plot(kind="bar")
    plt.title(f"{label} Positive Rate by Age (only 1s)")
    plt.ylabel("Positive Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()

    safe_name = label.lower().replace(" ", "_")
    plt.savefig(os.path.join(output_dir, f"{safe_name}_age_rate.png"))
    plt.show()

# ----------------------------
# 11. POSITIVE RATE BY SEX
# ----------------------------
for label in labels:
    grouped = df.groupby("Sex")[label].apply(lambda x: (x == 1).mean())

    plt.figure(figsize=(6, 4))
    grouped.plot(kind="bar")
    plt.title(f"{label} Positive Rate by Sex")
    plt.tight_layout()

    safe_name = label.lower().replace(" ", "_")
    plt.savefig(os.path.join(output_dir, f"{safe_name}_sex_rate.png"))
    plt.show()

print(f"\nAll plots saved in: {output_dir}")