import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score
from preprocess_DY import CheXpertDataset, TRANSFORM, LABELS

BASE_DIR  = '/resnick/groups/CS156b/from_central/data'
SAVE_DIR  = '/resnick/groups/CS156b/from_central/2026/JDP/dean_folder'

NUM_LABELS = 9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

frontal_val = CheXpertDataset(os.path.join(SAVE_DIR, 'frontal_val_DY.csv'), BASE_DIR, view='all', transform=TRANSFORM)
lateral_val = CheXpertDataset(os.path.join(SAVE_DIR, 'lateral_val_DY.csv'), BASE_DIR, view='all', transform=TRANSFORM)
print(f"Frontal val samples: {len(frontal_val)}", flush=True)
print(f"Lateral val samples: {len(lateral_val)}", flush=True)

def evaluate(model, loader, name):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-all_logits))

    aucs = []
    for i, label in enumerate(LABELS):
        valid = ~np.isnan(all_labels[:, i])
        y_true = all_labels[valid, i]
        y_pred = probs[valid, i]
        if len(np.unique(y_true)) < 2:
            print(f"  {label}: skipped (only one class present)", flush=True)
            continue
        auc = roc_auc_score(y_true, y_pred)
        aucs.append(auc)
        print(f"  {label}: AUC = {auc:.4f}", flush=True)
    print(f"{name} mean AUC: {np.mean(aucs):.4f}\n", flush=True)

frontal_loader = DataLoader(frontal_val, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
lateral_loader = DataLoader(lateral_val, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

frontal_model = models.densenet121(weights=None)
frontal_model.classifier = nn.Linear(frontal_model.classifier.in_features, NUM_LABELS)
frontal_model.load_state_dict(torch.load(
    os.path.join(SAVE_DIR, 'densenet121_frontal.pth'), map_location=device, weights_only=True))
frontal_model = frontal_model.to(device)
print("Frontal validation:", flush=True)
evaluate(frontal_model, frontal_loader, "Frontal")

lateral_model = models.densenet121(weights=None)
lateral_model.classifier = nn.Linear(lateral_model.classifier.in_features, NUM_LABELS)
lateral_model.load_state_dict(torch.load(
    os.path.join(SAVE_DIR, 'densenet121_lateral.pth'), map_location=device, weights_only=True))
lateral_model = lateral_model.to(device)
print("Lateral validation:", flush=True)
evaluate(lateral_model, lateral_loader, "Lateral")
