import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import argparse

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=str, required=True)
p.add_argument("--output", type=str, default="submission.csv")
args = p.parse_args()

LABEL_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Pneumonia", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]
DATA_DIR = "/resnick/groups/CS156b/from_central/data"
TEST_CSV = "/resnick/groups/CS156b/from_central/data/student_labels/test_ids.csv"


class TestDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = f"{self.data_dir}/{self.df.iloc[idx]['Path']}"
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load best model
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, len(LABEL_COLS))
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# predict
test_dataset = TestDataset(TEST_CSV, DATA_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

all_preds = []
with torch.no_grad():
    for images in test_loader:
        outputs = model(images.to(device))
        all_preds.append(outputs.cpu().numpy())

all_preds = np.concatenate(all_preds)

# build submission
test_df = pd.read_csv(TEST_CSV)
submission = pd.DataFrame(all_preds, columns=LABEL_COLS)
submission.insert(0, "Id", test_df["Id"])
submission.to_csv(args.output, index=False)
print(f"Saved {args.output} with {len(submission)} rows")