import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter
import matplotlib.pyplot as plt


train_csv = "/resnick/groups/CS156b/from_central/data/student_labels/train2023.csv"
image_base = "/resnick/groups/CS156b/from_central/data/"  # adjust if needed

df = pd.read_csv(train_csv)

# Sample across the dataset, not just the first 20
sample_df = df['Path'].dropna().sample(30, random_state=42).tolist()

widths, heights, channels, modes = [], [], [], []

for rel_path in sample_df:
    full_path = os.path.join(image_base, rel_path)
    if not os.path.exists(full_path):
        print(f"Missing: {full_path}")
        continue
    with Image.open(full_path) as img:
        w, h = img.size
        mode = img.mode
        c = len(img.getbands())
        widths.append(w)
        heights.append(h)
        channels.append(c)
        modes.append(mode)
        print(f"{os.path.basename(rel_path)}: {w}x{h}, mode={mode}, channels={c}")

print("\n--- Aggregate Stats (sample of 30) ---")
print(f"Width  — min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.0f}, median: {int(np.median(widths))}")
print(f"Height — min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.0f}, median: {int(np.median(heights))}")
print(f"Channel counts: {Counter(channels)}")
print(f"Image modes:    {Counter(modes)}")

unique_sizes = set(zip(widths, heights))
print(f"\nUnique (W, H) combos in sample: {len(unique_sizes)}")
for s in sorted(unique_sizes):
    print(f"  {s}")

print(df.head(30))