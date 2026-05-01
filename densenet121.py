import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from preprocess_DY import CheXpertDataset, TRANSFORM

BASE_DIR   = '/resnick/groups/CS156b/from_central/data'
SAVE_DIR   = '/resnick/groups/CS156b/from_central/2026/JDP/dean_folder'

os.makedirs(SAVE_DIR, exist_ok=True)

frontal_dataset = CheXpertDataset(os.path.join(SAVE_DIR, 'frontal_train_DY.csv'), BASE_DIR, view='all', transform=TRANSFORM)
lateral_dataset = CheXpertDataset(os.path.join(SAVE_DIR, 'lateral_train_DY.csv'), BASE_DIR, view='all', transform=TRANSFORM)
print(f"Frontal samples: {len(frontal_dataset)}", flush=True)
print(f"Lateral samples: {len(lateral_dataset)}", flush=True)

NUM_LABELS = 9
NUM_EPOCHS = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

def make_densenet():
    m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    m.classifier = nn.Linear(m.classifier.in_features, NUM_LABELS)
    return m.to(device)

def train(model, loader, num_epochs, save_path):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss / len(loader):.4f}", flush=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved {save_path}", flush=True)

frontal_loader = DataLoader(
    frontal_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
print("Training frontal model...", flush=True)
train(make_densenet(), frontal_loader, NUM_EPOCHS,
      os.path.join(SAVE_DIR, 'densenet121_frontal.pth'))

lateral_loader = DataLoader(
    lateral_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
print("Training lateral model...", flush=True)
train(make_densenet(), lateral_loader, NUM_EPOCHS,
      os.path.join(SAVE_DIR, 'densenet121_lateral.pth'))
