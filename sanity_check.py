import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from preprocess_DY import LABELS

BASE_DIR = '/resnick/groups/CS156b/from_central/data'
SAVE_DIR = '/resnick/groups/CS156b/from_central/2026/JDP/dean_folder'
TEST_CSV = '/resnick/groups/CS156b/from_central/data/student_labels/test_ids.csv'

SPLITS = {
    'frontal_train': os.path.join(SAVE_DIR, 'frontal_train_DY.csv'),
    'frontal_val':   os.path.join(SAVE_DIR, 'frontal_val_DY.csv'),
    'lateral_train': os.path.join(SAVE_DIR, 'lateral_train_DY.csv'),
    'lateral_val':   os.path.join(SAVE_DIR, 'lateral_val_DY.csv'),
}

def show_images(df, title, out_path, n=5):
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for i, ax in enumerate(axes):
        row = df.iloc[i]
        img = Image.open(os.path.join(BASE_DIR, row['Path'])).convert('RGB')
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(row['Path'].split('/')[-1], fontsize=7)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}", flush=True)

# ── Train / Val splits ────────────────────────────────────────────────────────
for name, csv_path in SPLITS.items():
    df = pd.read_csv(csv_path)
    print(f"\n{'='*50}", flush=True)
    print(f"{name}: {len(df)} images", flush=True)
    print(df[['Path', 'Frontal/Lateral'] + LABELS].head().to_string(), flush=True)
    show_images(df, name, os.path.join(SAVE_DIR, f'sanity_{name}.png'))

# ── Test set ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}", flush=True)
test_df = pd.read_csv(TEST_CSV)
frontal_test = test_df[test_df['Path'].str.contains('frontal', case=False)]
lateral_test = test_df[test_df['Path'].str.contains('lateral', case=False)]

print(f"Test total:   {len(test_df)}", flush=True)
print(f"Test frontal: {len(frontal_test)}", flush=True)
print(f"Test lateral: {len(lateral_test)}", flush=True)

print("\nTest head:", flush=True)
print(test_df.head().to_string(), flush=True)

show_images(frontal_test.reset_index(drop=True), 'test_frontal',
            os.path.join(SAVE_DIR, 'sanity_test_frontal.png'))
show_images(lateral_test.reset_index(drop=True), 'test_lateral',
            os.path.join(SAVE_DIR, 'sanity_test_lateral.png'))

print("\nDone. All plots saved to dean_folder.", flush=True)
