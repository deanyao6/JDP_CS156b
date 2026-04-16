import pandas as pd

# 1. Load the CSV file
df = pd.read_csv('frontal_train.csv')

# 2. Print the shape (rows, columns)
print("Shape:", df.shape)

# 3. Print the column names
print("Columns:", df.columns.tolist())
