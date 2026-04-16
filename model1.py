import os
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Paths ─────────────────────────────────────────────────────────────

csv_path = "/resnick/groups/CS156b/from_central/data/student_labels/train2023.csv"
base_dir = "/resnick/groups/CS156b/from_central/data"   # IMPORTANT

# ── Transform ─────────────────────────────────────────────────────────

def pad_to_square(img):
    w, h = img.size
    max_side = max(w, h)
    padding = (
        (max_side - w) // 2,
        (max_side - h) // 2,
        (max_side - w + 1) // 2,
        (max_side - h + 1) // 2,
    )
    return ImageOps.expand(img, padding, fill=0)

TRANSFORM = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Load + Clean Data ─────────────────────────────────────────────────

df = pd.read_csv(csv_path)

# keep only lateral images
df = df[df['Frontal/Lateral'] == 'Lateral']

# drop rows where Pneumonia is NaN
df = df.dropna(subset=['Pneumonia'])

# map labels: -1 → 0, keep 1 as is
df['Pneumonia'] = df['Pneumonia'].replace(-1, 0).astype('float32')

df = df.reset_index(drop=True)

# ── Dataset ───────────────────────────────────────────────────────────

class PneumoniaDataset(Dataset):
    def __init__(self, df, base_dir, transform=None):
        self.df = df
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.base_dir, row['Path'])
        img = Image.open(img_path).convert('L')

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(row['Pneumonia'], dtype=torch.float32)

        return img, label

# ── Create Dataset + Loader ───────────────────────────────────────────

dataset = PneumoniaDataset(df, base_dir, transform=TRANSFORM)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# ── Sanity Checks ─────────────────────────────────────────────────────

print("Dataset size:", len(dataset))
print("Label distribution:\n", df['Pneumonia'].value_counts())

img, label = dataset[0]
print("Image shape:", img.shape)   # should be [3, 224, 224]
print("Label:", label)