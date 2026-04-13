import os
import pandas as pd

train_csv = "/resnick/groups/CS156b/from_central/data/student_labels/train2023.csv"

# hello

df = pd.read_csv(train_csv)

print(df.columns.tolist())
print("num rows:", len(df))
print(df.head())
