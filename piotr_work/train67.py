import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image


DATA_ROOT = "/resnick/groups/CS156b/from_central/data"
CSV_PATH = "frontal_dataset.csv"

LABEL_COLS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Pneumonia",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


class CheXpertDataset(Dataset):
    def __init__(self, dataframe, data_root, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform

        self.df[LABEL_COLS] = self.df[LABEL_COLS].replace({
            -1.0: 0.0,
            0.0: float("nan"),
        })

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.data_root, row["Path"])
        image = Image.open(img_path).convert("RGB")

        labels = torch.tensor(row[LABEL_COLS].values.astype("float32"))

        if self.transform:
            image = self.transform(image)

        return image, labels


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

full_df = pd.read_csv(CSV_PATH)

full_df = full_df.dropna(subset=LABEL_COLS, how="all")

# Optional: shuffle full dataset first
full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Take small subset
train_df = full_df.iloc[:2_000].copy()
val_df = full_df.iloc[2_000:10_000].copy()

train_dataset = CheXpertDataset(
    dataframe=train_df,
    data_root=DATA_ROOT,
    transform=train_transform,
)

val_dataset = CheXpertDataset(
    dataframe=val_df,
    data_root=DATA_ROOT,
    transform=val_transform,
)

# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size

# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Replace final layer for 9 labels
model.fc = nn.Linear(model.fc.in_features, len(LABEL_COLS))

model = model.to(device)

criterion = nn.BCEWithLogitsLoss(reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def masked_loss(outputs, labels):
    mask = ~torch.isnan(labels)

    # If batch has no valid labels, skip it
    if mask.sum() == 0:
        print("masking is empty")
        return None

    safe_labels = torch.where(mask, labels, torch.zeros_like(labels))

    loss_matrix = criterion(outputs, safe_labels)

    loss = loss_matrix[mask].mean()
    return loss


def train_one_epoch(model, loader):
    model.train()
    total_loss = 0
    num_batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = masked_loss(outputs, labels)

        if loss is None:
            continue

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, loader):
    model.eval()

    total_loss = 0
    num_batches = 0

    pathology_loss_sums = torch.zeros(len(LABEL_COLS), device=device)
    pathology_counts = torch.zeros(len(LABEL_COLS), device=device)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = masked_loss(outputs, labels)

            if loss is not None:
                total_loss += loss.item()
                num_batches += 1

            mask = ~torch.isnan(labels)
            safe_labels = torch.where(mask, labels, torch.zeros_like(labels))

            loss_matrix = criterion(outputs, safe_labels)

            valid_loss_matrix = torch.where(
                mask,
                loss_matrix,
                torch.zeros_like(loss_matrix)
            )

            pathology_loss_sums += valid_loss_matrix.sum(dim=0)
            pathology_counts += mask.sum(dim=0)

    avg_total_loss = total_loss / num_batches

    pathology_avg_losses = pathology_loss_sums / torch.clamp(pathology_counts, min=1)
    pathology_avg_losses = pathology_avg_losses.detach().cpu().numpy()

    return avg_total_loss, pathology_avg_losses




num_epochs = 3

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss, val_pathology_losses = validate(model, val_loader)

    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")

    print("Validation loss by pathology:")
    for name, loss_val in zip(LABEL_COLS, val_pathology_losses):
        print(f"  {name}: {loss_val:.4f}")


torch.save(model.state_dict(), "first_run.pth")