import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from preprocess_DY import CheXpertDataset, TRANSFORM, TRAIN_TRANSFORM

BASE_DIR   = '/resnick/groups/CS156b/from_central/data'
SAVE_DIR   = '/resnick/groups/CS156b/from_central/2026/JDP/dean_folder'

os.makedirs(SAVE_DIR, exist_ok=True)

frontal_train = CheXpertDataset(os.path.join(SAVE_DIR, 'frontal_train_DY.csv'), BASE_DIR, view='all', transform=TRAIN_TRANSFORM)
lateral_train = CheXpertDataset(os.path.join(SAVE_DIR, 'lateral_train_DY.csv'), BASE_DIR, view='all', transform=TRAIN_TRANSFORM)
frontal_val   = CheXpertDataset(os.path.join(SAVE_DIR, 'frontal_val_DY.csv'),   BASE_DIR, view='all', transform=TRANSFORM)
lateral_val   = CheXpertDataset(os.path.join(SAVE_DIR, 'lateral_val_DY.csv'),   BASE_DIR, view='all', transform=TRANSFORM)
print(f"Frontal — train: {len(frontal_train)} | val: {len(frontal_val)}", flush=True)
print(f"Lateral — train: {len(lateral_train)} | val: {len(lateral_val)}", flush=True)

NUM_LABELS = 9
NUM_EPOCHS = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

def make_densenet():
    m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    m.classifier = nn.Linear(m.classifier.in_features, NUM_LABELS)
    return m.to(device)

def train(model, train_loader, val_loader, num_epochs, save_path, title):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            mask = ~torch.isnan(labels)
            loss = criterion(logits[mask], labels[mask])
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_loss = total / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                mask = ~torch.isnan(labels)
                total += criterion(logits[mask], labels[mask]).item()
        val_loss = total / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}", flush=True)

    torch.save(model.state_dict(), save_path)
    print(f"Saved {save_path}", flush=True)

    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train')
    plt.plot(range(1, num_epochs + 1), val_losses,   label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plot_path = save_path.replace('.pth', '_loss.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved loss plot {plot_path}", flush=True)

frontal_train_loader = DataLoader(frontal_train, batch_size=64, shuffle=True,  num_workers=8, pin_memory=True)
frontal_val_loader   = DataLoader(frontal_val,   batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
print("Training frontal model...", flush=True)
train(make_densenet(), frontal_train_loader, frontal_val_loader, NUM_EPOCHS,
      os.path.join(SAVE_DIR, 'densenet121_frontal.pth'), 'Frontal Loss')

lateral_train_loader = DataLoader(lateral_train, batch_size=64, shuffle=True,  num_workers=8, pin_memory=True)
lateral_val_loader   = DataLoader(lateral_val,   batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
print("Training lateral model...", flush=True)
train(make_densenet(), lateral_train_loader, lateral_val_loader, NUM_EPOCHS,
      os.path.join(SAVE_DIR, 'densenet121_lateral.pth'), 'Lateral Loss')
