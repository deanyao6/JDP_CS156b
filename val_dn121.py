from preprocess_DY import CheXpertDataset, pad_to_square
import os
from torchvision import transforms, models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
import numpy as np

# constants
train_path = '/resnick/groups/CS156b/from_central/data'
save_dir = '/resnick/groups/CS156b/from_central/2026/JDP/dean_folder'
TRANSFORM = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

NUM_LABELS = 9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Pneumonia', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices',
]

# ── Datasets (5001st image onwards) ──────────────────────────────────────────

frontal_full = CheXpertDataset(os.path.join(save_dir, 'frontal_train.csv'), train_path, view='all', transform=TRANSFORM)
lateral_full = CheXpertDataset(os.path.join(save_dir, 'lateral_train.csv'), train_path, view='all', transform=TRANSFORM)

frontal_val = Subset(frontal_full, range(len(frontal_full) - 5000, len(frontal_full)))
lateral_val = Subset(lateral_full, range(len(lateral_full) - 5000, len(lateral_full)))

frontal_loader = DataLoader(frontal_val, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
lateral_loader = DataLoader(lateral_val, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# ── Eval function ─────────────────────────────────────────────────────────────

def evaluate(model, loader, name):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-all_logits))  # sigmoid

    # AUC per label (skip labels with only one class present)
    aucs = []
    for i, label in enumerate(LABELS):
        if len(np.unique(all_labels[:, i])) < 2:
            print(f"  {label}: skipped (only one class in val set)")
            continue
        auc = roc_auc_score(all_labels[:, i], probs[:, i])
        aucs.append(auc)
        print(f"  {label}: AUC = {auc:.4f}")

    print(f"{name} mean AUC: {np.mean(aucs):.4f}\n")

# ── Load and evaluate frontal ─────────────────────────────────────────────────

frontal_model = models.densenet121(weights=None)
frontal_model.classifier = nn.Linear(frontal_model.classifier.in_features, NUM_LABELS)
frontal_model.load_state_dict(torch.load(os.path.join(save_dir, 'densenet121_frontal_small.pth'), map_location=device, weights_only = True))
frontal_model = frontal_model.to(device)

print("Frontal validation:")
evaluate(frontal_model, frontal_loader, "Frontal")

# ── Load and evaluate lateral ─────────────────────────────────────────────────

lateral_model = models.densenet121(weights=None)
lateral_model.classifier = nn.Linear(lateral_model.classifier.in_features, NUM_LABELS)
lateral_model.load_state_dict(torch.load(os.path.join(save_dir, 'densenet121_lateral_small.pth'), map_location=device, weights_only = True))
lateral_model = lateral_model.to(device)

print("Lateral validation:")
evaluate(lateral_model, lateral_loader, "Lateral")