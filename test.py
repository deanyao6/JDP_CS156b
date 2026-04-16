import pandas as pd, os
df = pd.read_csv('frontal_train.csv')
sample = df.iloc[0]['Path'].replace('CheXpert-v1.0/', '')
full = os.path.join('/resnick/groups/CS156b/from_central/data', sample)
print(os.path.exists(full))  # should print True