import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms

PATHOLOGIES = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Pneumonia',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices',
]

def pad_to_square(img):
    """Pad the shorter side with black pixels to make the image square."""
    w, h = img.size
    max_side = max(w, h)
    # (left, top, right, bottom)
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225]),
])


class CheXpertDataset(Dataset):
    """
    A single dataset for one view type and one pathology.
    
    Args:
        df:         DataFrame (already filtered to frontal or lateral)
        pathology:  One of the 9 strings in PATHOLOGIES
        image_base: Root directory prepended to df['Path']
        transform:  Torchvision transform pipeline
    """
    def init(self, df, pathology, image_base, transform=TRANSFORM):
        assert pathology in PATHOLOGIES, f"Unknown pathology: {pathology}"
        self.image_base = image_base
        self.pathology = pathology
        self.transform = transform

        # Drop rows where this pathology label is NaN
        self.df = df[df[pathology].notna()].reset_index(drop=True)
        print(f"  [{pathology}] {len(self.df)} labeled rows")

    def len(self):
        return len(self.df)

    def getitem(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_base, row['Path'])
        
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)

        # Label: 1.0 = positive, 0.0 = negative, -1.0 = uncertain (u-ones/u-zeros policy can replace later)
        label = torch.tensor(row[self.pathology], dtype=torch.float32)
        return img, label


def build_datasets(csv_path, image_base):
    """
    Returns a nested dict:
        datasets['frontal']['Cardiomegaly'] -> CheXpertDataset
        datasets['lateral']['Pleural Effusion'] -> CheXpertDataset
        ...
    """
    df = pd.read_csv(csv_path)

    # Split on Frontal/Lateral column
    frontal_df = df[df['Frontal/Lateral'] == 'Frontal'].reset_index(drop=True)
    lateral_df = df[df['Frontal/Lateral'] == 'Lateral'].reset_index(drop=True)
    print(f"Frontal: {len(frontal_df)} rows | Lateral: {len(lateral_df)} rows\n")

    datasets = {'frontal': {}, 'lateral': {}}

    for view, view_df in [('frontal', frontal_df), ('lateral', lateral_df)]:
        print(f"--- {view.upper()} ---")
        for pathology in PATHOLOGIES:
            datasets[view][pathology] = CheXpertDataset(
                df=view_df,
                pathology=pathology,
                image_base=image_base,
            )

    return datasets


# Usage
if __name__ == "__main__":
    datasets = build_datasets(
        csv_path='/resnick/groups/CS156b/from_central/data/student_labels/train2023.csv',
        image_base='/resnick/groups/CS156b/from_central/data/',
    )

    # Access any sub-dataset
    ds = datasets['frontal']['Cardiomegaly']
    img, label = ds[0]
    print(f"\nSample — image shape: {img.shape}, label: {label.item()}")