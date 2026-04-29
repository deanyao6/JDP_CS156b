import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import models
from preprocess_DY import CheXpertDataset, TRANSFORM, LABELS

BASE_DIR        = '/resnick/groups/CS156b/from_central/data'
SAVE_DIR        = '/resnick/groups/CS156b/from_central/2026/JDP/dean_folder'
TEST_CSV        = '/resnick/groups/CS156b/from_central/data/student_labels/test_ids.csv'
FRONTAL_WEIGHTS = os.path.join(SAVE_DIR, 'densenet121_frontal.pth')
LATERAL_WEIGHTS = os.path.join(SAVE_DIR, 'densenet121_lateral.pth')
OUT_CSV         = os.path.join(SAVE_DIR, 'submission_DY.csv')

NUM_LABELS = 9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

def load_model(weights_path):
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, NUM_LABELS)
    m.load_state_dict(torch.load(weights_path, map_location=device))
    m.to(device)
    m.eval()
    return m

def run_inference(model, dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    all_preds = []
    all_paths = []
    with torch.no_grad():
        for imgs, paths in loader:
            logits = model(imgs.to(device))
            preds  = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_paths.extend(paths)
    return np.vstack(all_preds), all_paths

frontal_model = load_model(FRONTAL_WEIGHTS)
lateral_model = load_model(LATERAL_WEIGHTS)

frontal_ds = CheXpertDataset(TEST_CSV, BASE_DIR, view='frontal', transform=TRANSFORM)
lateral_ds = CheXpertDataset(TEST_CSV, BASE_DIR, view='lateral', transform=TRANSFORM)
print(f"Frontal test samples: {len(frontal_ds)}", flush=True)
print(f"Lateral test samples: {len(lateral_ds)}", flush=True)

frontal_preds, frontal_paths = run_inference(frontal_model, frontal_ds)
lateral_preds, lateral_paths = run_inference(lateral_model, lateral_ds)

frontal_df = pd.DataFrame(frontal_preds, columns=LABELS)
frontal_df.insert(0, 'Path', frontal_paths)

lateral_df = pd.DataFrame(lateral_preds, columns=LABELS)
lateral_df.insert(0, 'Path', lateral_paths)

preds_df = pd.concat([frontal_df, lateral_df], ignore_index=True)

test_df = pd.read_csv(TEST_CSV)
out_df = test_df[['Id', 'Path']].merge(preds_df, on='Path', how='left')
out_df = out_df.drop(columns=['Path'])

out_df.to_csv(OUT_CSV, index=False)
print(f"Saved {OUT_CSV} with {len(out_df)} rows", flush=True)
