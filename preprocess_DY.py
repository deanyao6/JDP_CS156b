import os
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

train_path = '/resnick/groups/CS156b/from_central/data'

# transform images 

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

# make the two datasets

LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Pneumonia', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices',
]

class CheXpertDataset(Dataset):
    def __init__(self, csv_path, base_dir, view='frontal', transform=None):
        """
        Args:
            csv_path  : path to train.csv / valid.csv
            base_dir  : root that the CSV paths are relative to
            view      : 'frontal' | 'lateral' | 'all'
            transform : torchvision transform
        """
        df = pd.read_csv(csv_path)

        if view == 'frontal':
            df = df[df['Frontal/Lateral'] == 'Frontal']
        elif view == 'lateral':
            df = df[df['Frontal/Lateral'] == 'Lateral']
        # 'all' keeps everything

        df = df.reset_index(drop=True)

        self.df        = df
        self.base_dir  = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # CSV paths look like "CheXpert-v1.0/train/patient00001/..."
        img_path = os.path.join(self.base_dir, row['Path'])
        img = Image.open(img_path).convert('L')   # load as grayscale PIL

        if self.transform:
            img = self.transform(img)

        # Fill NaN labels: -1 (uncertain) → 0, NaN (unmentioned) → 0
        labels = (
            self.df[LABELS]
            .iloc[idx]
            .fillna(0)
            .replace(-1, 0)
            .values.astype('float32')
        )

        return img, labels

# make the datasets

csv_path = os.path.join(train_path, 'student_labels', 'train2023.csv')

# save filtered CSVs (run once)
# save_dir = '/resnick/groups/CS156b/from_central/2026/JDP/dean_folder'

# df = pd.read_csv(csv_path)
# df[df['Frontal/Lateral'] == 'Frontal'].to_csv(os.path.join(save_dir, 'frontal_train.csv'), index=False)
# df[df['Frontal/Lateral'] == 'Lateral'].to_csv(os.path.join(save_dir, 'lateral_train.csv'), index=False)
# print("Saved frontal_train.csv and lateral_train.csv")

frontal_dataset = CheXpertDataset(csv_path, train_path, view='frontal', transform=TRANSFORM)
lateral_dataset = CheXpertDataset(csv_path, train_path, view='lateral', transform=TRANSFORM)

frontal_loader = DataLoader(frontal_dataset, batch_size=32, shuffle=True,  num_workers=1)
lateral_loader = DataLoader(lateral_dataset, batch_size=32, shuffle=True,  num_workers=1)

print(f"Frontal samples : {len(frontal_dataset)}")
print(f"Lateral samples : {len(lateral_dataset)}")