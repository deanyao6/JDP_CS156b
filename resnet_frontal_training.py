import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score
from PIL import Image


LABEL_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Pneumonia", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]

DATA_DIR = "/resnick/groups/CS156b/from_central/data"


class CheXpertDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.transform = transform

        self.labels = self.df[LABEL_COLS].values.astype("float32")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = f"{self.data_dir}/{self.df.iloc[idx]['Path']}"
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        labels = torch.tensor(self.labels[idx])
        return img, labels


def get_transforms(split="train", img_size=224):
    if split == "train":
        return transforms.Compose([
            # transforms.Resize(256),
            transforms.Resize((img_size, img_size)),
            # transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            # transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    n_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # mask out NaN labels so they don't affect the loss
        mask = ~torch.isnan(labels)
        if mask.sum() == 0:
            continue

        loss = criterion(outputs[mask], labels[mask])
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * mask.sum().item()
        n_samples += mask.sum().item()

    return running_loss / max(n_samples, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    n_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            mask = ~torch.isnan(labels)
            if mask.sum() > 0:
                loss = criterion(outputs[mask], labels[mask])
                running_loss += loss.item() * mask.sum().item()
                n_samples += mask.sum().item()

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / max(n_samples, 1)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # per-pathology AUC, skipping NaN entries
    aucs = {}
    for i, col in enumerate(LABEL_COLS):
        valid = ~np.isnan(all_labels[:, i])
        if valid.sum() > 0 and len(np.unique(all_labels[valid, i])) > 1:
            binary_labels = (all_labels[valid, i] == 1).astype(float)
            aucs[col] = roc_auc_score(all_labels[valid, i], all_preds[valid, i])
        else:
            aucs[col] = float("nan")

    return avg_loss, aucs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, default="train_clean.csv")
    p.add_argument("--train_csv", type=str, default="train_clean.csv")
    p.add_argument("--valid_csv", type=str, default="val_clean.csv")
    p.add_argument("--data_dir", type=str, default=DATA_DIR)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="checkpoints")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # datasets

    
    # train_dataset = CheXpertDataset(
    #     args.train_csv, args.data_dir,
    #     transform=get_transforms("train"),
    # )

    train_dataset = CheXpertDataset(
        args.train_csv, args.data_dir,
        transform=get_transforms("train"),
    )

    if args.subset:
        train_dataset.df = train_dataset.df.head(args.subset)
        train_dataset.labels = train_dataset.labels[:args.subset]

    val_dataset = CheXpertDataset(
        args.valid_csv, args.data_dir,
        transform=get_transforms("val"),
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # model — ResNet-50 with pretrained weights, new head for 9 pathologies
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, len(LABEL_COLS))
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, aucs = evaluate(model, val_loader, criterion, device)

        mean_auc = np.nanmean(list(aucs.values()))
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.0f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Mean AUC:   {mean_auc:.4f}")
        for col, auc in aucs.items():
            print(f"    {col}: {auc:.4f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mean_auc": mean_auc,
                "aucs": aucs,
                "args": vars(args),
            }, f"{args.output_dir}/best_resnet50.pth")
            print(f"  Saved best model (AUC={mean_auc:.4f})")

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
    }, f"{args.output_dir}/final_resnet50.pth")
    print(f"\nDone. Best mean AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()