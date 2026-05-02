import os
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

train_path = '/resnick/groups/CS156b/from_central/data'
SAVE_DIR   = '/resnick/groups/CS156b/from_central/2026/JDP/dean_folder'

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
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Pneumonia', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices',
]

class CheXpertDataset(Dataset):
    def __init__(self, csv_path, base_dir, view='frontal', transform=None):
        df = pd.read_csv(csv_path)

        if view in ('frontal', 'lateral'):
            if 'Frontal/Lateral' in df.columns:
                df = df[df['Frontal/Lateral'] == view.capitalize()]
            else:
                df = df[df['Path'].str.contains(view, case=False)]

        self.df        = df.reset_index(drop=True)
        self.base_dir  = base_dir
        self.transform = transform
        self.has_labels = all(c in df.columns for c in LABELS)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.base_dir, row['Path'])
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        if self.has_labels:
            labels = (
                self.df[LABELS]
                .iloc[idx]
                .replace(-1, float('nan'))
                .values.astype('float32')
            )
            return img, labels

        return img, row['Path']


if __name__ == '__main__':
    csv_path = os.path.join(train_path, 'student_labels', 'train2023.csv')
    df = pd.read_csv(csv_path)
    df[df['Frontal/Lateral'] == 'Frontal'].to_csv(
        os.path.join(SAVE_DIR, 'frontal_train.csv'), index=False)
    df[df['Frontal/Lateral'] == 'Lateral'].to_csv(
        os.path.join(SAVE_DIR, 'lateral_train.csv'), index=False)
    print("Saved frontal_train.csv and lateral_train.csv")

    frontal_dataset = CheXpertDataset(
        os.path.join(SAVE_DIR, 'frontal_train.csv'), train_path, view='all', transform=TRANSFORM)
    lateral_dataset = CheXpertDataset(
        os.path.join(SAVE_DIR, 'lateral_train.csv'), train_path, view='all', transform=TRANSFORM)
    print(f"Frontal samples : {len(frontal_dataset)}")
    print(f"Lateral samples : {len(lateral_dataset)}")
