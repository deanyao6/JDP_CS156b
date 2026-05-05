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

# Your encoding in the CSV:
#   -1 = negative
#    0 = uncertain
#    1 = positive
#  NaN = missing / unlabeled
#
# For CrossEntropyLoss, classes must be 0, 1, 2, so we map:
#   -1 -> 0  negative
#    0 -> 1  uncertain
#    1 -> 2  positive
CLASS_TO_LABEL = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
LABEL_TO_CLASS = {-1.0: 0, 0.0: 1, 1.0: 2}
NUM_CLASSES = 3
DATA_DIR = "/resnick/groups/CS156b/from_central/data"


class CheXpertDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None, subset=None):
        self.df = pd.read_csv(csv_path)
        if subset is not None:
            self.df = self.df.head(subset).copy()
        self.data_dir = data_dir
        self.transform = transform
        self.labels = self.df[LABEL_COLS].values.astype("float32")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.df.iloc[idx]["Path"])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, labels


def get_transforms(split="train", img_size=224):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.97, 1.03)),
            transforms.ToTensor(),
            # ImageNet normalization is usually best for ImageNet-pretrained ResNet models.
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def build_model(device):
    """ResNet-50 shared backbone with 9 pathology heads, each predicting 3 classes."""
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, len(LABEL_COLS) * NUM_CLASSES)
    model = model.to(device)
    return model


def labels_to_class_targets(labels):
    """
    Convert labels from {-1, 0, 1, NaN} to class ids {0, 1, 2, -100}.
    -100 is the ignore_index used by CrossEntropyLoss.
    """
    targets = torch.full_like(labels, fill_value=-100, dtype=torch.long)
    targets[labels == -1] = 0
    targets[labels == 0] = 1
    targets[labels == 1] = 2
    return targets


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    n_labeled = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        targets = labels_to_class_targets(labels).to(device)  # shape: (B, 9)
        if (targets != -100).sum() == 0:
            continue

        optimizer.zero_grad()
        logits = model(images).view(-1, len(LABEL_COLS), NUM_CLASSES)  # (B, 9, 3)

        # CrossEntropyLoss expects input (N, C) and target (N,).
        loss = criterion(logits.reshape(-1, NUM_CLASSES), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        labeled_count = (targets != -100).sum().item()
        running_loss += loss.item() * labeled_count
        n_labeled += labeled_count

    return running_loss / max(n_labeled, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    n_labeled = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            targets = labels_to_class_targets(labels).to(device)

            logits = model(images).view(-1, len(LABEL_COLS), NUM_CLASSES)
            if (targets != -100).sum() > 0:
                loss = criterion(logits.reshape(-1, NUM_CLASSES), targets.reshape(-1))
                labeled_count = (targets != -100).sum().item()
                running_loss += loss.item() * labeled_count
                n_labeled += labeled_count

            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / max(n_labeled, 1)
    all_probs = np.concatenate(all_probs, axis=0)      # (N, 9, 3)
    all_labels = np.concatenate(all_labels, axis=0)    # (N, 9)

    # For AUC, measure how well P(positive) ranks true positives above non-positives.
    aucs = {}
    pos_probs = all_probs[:, :, 2]
    for i, col in enumerate(LABEL_COLS):
        valid = ~np.isnan(all_labels[:, i])
        if valid.sum() > 0 and len(np.unique(all_labels[valid, i] == 1)) > 1:
            binary_labels = (all_labels[valid, i] == 1).astype(float)
            aucs[col] = roc_auc_score(binary_labels, pos_probs[valid, i])
        else:
            aucs[col] = float("nan")

    return avg_loss, aucs


def make_loader(csv_path, data_dir, split, batch_size, num_workers, subset=None):
    dataset = CheXpertDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        transform=get_transforms(split),
        subset=subset,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def train_for_epochs(model, train_csv, valid_csv, args, device, start_epoch, num_epochs, stage_name):
    train_dataset, train_loader = make_loader(
        train_csv, args.data_dir, "train", args.batch_size, args.num_workers, args.subset
    )
    val_dataset, val_loader = make_loader(
        valid_csv, args.data_dir, "val", args.batch_size, args.num_workers, None
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"\n{stage_name}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    best_auc = -np.inf
    best_state = None

    for local_epoch in range(1, num_epochs + 1):
        epoch = start_epoch + local_epoch - 1
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, aucs = evaluate(model, val_loader, criterion, device)
        mean_auc = np.nanmean(list(aucs.values()))
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch} ({stage_name}, {elapsed:.0f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Mean AUC:   {mean_auc:.4f}")
        for col, auc in aucs.items():
            print(f"    {col}: {auc:.4f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mean_auc": mean_auc,
                "aucs": aucs,
                "args": vars(args),
            }
            torch.save(best_state, os.path.join(args.output_dir, "best_resnet50_3class.pth"))
            print(f"  Saved best model (AUC={mean_auc:.4f})")

    return start_epoch + num_epochs, best_auc, best_state


def pseudo_label_csv(model, input_csv, output_csv, args, device):
    """
    Fill only NaN entries whose max softmax confidence is at least args.pseudo_threshold.
    Original labels are never changed. Low-confidence NaNs remain NaN.
    """
    df = pd.read_csv(input_csv)
    if args.subset is not None:
        # For debugging only. In real training, avoid subset here.
        df = df.head(args.subset).copy()

    dataset = CheXpertDataset(input_csv, args.data_dir, transform=get_transforms("val"), subset=args.subset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    all_probs = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            logits = model(images).view(-1, len(LABEL_COLS), NUM_CLASSES)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)  # (N, 9, 3)
    max_conf = probs.max(axis=-1)              # (N, 9)
    pred_class = probs.argmax(axis=-1)         # (N, 9)
    pred_label = CLASS_TO_LABEL[pred_class]    # convert 0/1/2 back to -1/0/1

    total_filled = 0
    filled_by_label = {}

    for j, col in enumerate(LABEL_COLS):
        original = df[col].values.astype("float32")
        missing = np.isnan(original)
        confident = max_conf[:, j] >= args.pseudo_threshold
        fill_mask = missing & confident

        df.loc[fill_mask, col] = pred_label[fill_mask, j]
        count = int(fill_mask.sum())
        filled_by_label[col] = count
        total_filled += count

    df.to_csv(output_csv, index=False)

    print(f"\nPseudo-labeling complete: {input_csv} -> {output_csv}")
    print(f"Confidence threshold: {args.pseudo_threshold:.2f}")
    print(f"Filled {total_filled} NaN labels")
    for col, count in filled_by_label.items():
        print(f"  {col}: {count}")

    return total_filled


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, default="train_clean.csv")
    p.add_argument("--valid_csv", type=str, default="val_clean.csv")
    p.add_argument("--data_dir", type=str, default=DATA_DIR)
    p.add_argument("--subset", type=int, default=None, help="use only N training samples for quick debugging")

    # Stage 1: train normally until the model is solid.
    p.add_argument("--base_epochs", type=int, default=10)

    # Stage 2: pseudo-label NaNs and retrain.
    p.add_argument("--pseudo_rounds", type=int, default=3)
    p.add_argument("--pseudo_epochs", type=int, default=3)
    p.add_argument("--pseudo_threshold", type=float, default=0.95)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="checkpoints")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(device)
    next_epoch = 1

    # First train on true known labels only.
    current_train_csv = args.train_csv
    next_epoch, best_auc, _ = train_for_epochs(
        model=model,
        train_csv=current_train_csv,
        valid_csv=args.valid_csv,
        args=args,
        device=device,
        start_epoch=next_epoch,
        num_epochs=args.base_epochs,
        stage_name="Base supervised training",
    )

    # Then repeatedly pseudo-label high-confidence NaNs and retrain.
    for round_idx in range(1, args.pseudo_rounds + 1):
        pseudo_csv = os.path.join(args.output_dir, f"train_pseudo_round{round_idx}.csv")
        filled = pseudo_label_csv(
            model=model,
            input_csv=current_train_csv,
            output_csv=pseudo_csv,
            args=args,
            device=device,
        )

        if filled == 0:
            print("No confident NaNs left to fill. Stopping pseudo-labeling rounds.")
            break

        current_train_csv = pseudo_csv
        next_epoch, round_best_auc, _ = train_for_epochs(
            model=model,
            train_csv=current_train_csv,
            valid_csv=args.valid_csv,
            args=args,
            device=device,
            start_epoch=next_epoch,
            num_epochs=args.pseudo_epochs,
            stage_name=f"Pseudo-label training round {round_idx}",
        )
        best_auc = max(best_auc, round_best_auc)

    torch.save({
        "epoch": next_epoch - 1,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
    }, os.path.join(args.output_dir, "first_run_output.pth"))

    print(f"\nDone. Best mean AUC: {best_auc:.4f}")
    print(f"Final training CSV used: {current_train_csv}")


if __name__ == "__main__":
    main()
