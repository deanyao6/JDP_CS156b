# import os
# import pandas as pd
# import numpy as np
# from PIL import Image, ImageOps
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import torch.nn as nn

# # pathologies list
# PATHOLOGIES = [
#     'No Finding',
#     'Enlarged Cardiomediastinum',
#     'Cardiomegaly',
#     'Lung Opacity',
#     'Pneumonia',
#     'Pleural Effusion',
#     'Pleural Other',
#     'Fracture',
#     'Support Devices',
# ]

# def pad_to_square(img):
#     w, h = img.size
#     max_side = max(w, h)
#     padding = (
#         (max_side - w) // 2,
#         (max_side - h) // 2,
#         (max_side - w + 1) // 2,
#         (max_side - h + 1) // 2,
#     )
#     return ImageOps.expand(img, padding, fill=0)

# TRANSFORM = transforms.Compose([
#     transforms.Lambda(pad_to_square),
#     transforms.Resize((224, 224)),
#     transforms.Grayscale(num_output_channels=3),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])

# class CheXpertDataset(Dataset):
#     def __init__(self, df, image_base, transform=TRANSFORM):
#         self.image_base = image_base
#         self.transform = transform
#         # fill NaN labels with 0
#         self.df = df.copy()
#         self.df[PATHOLOGIES] = self.df[PATHOLOGIES].fillna(0)
#         self.df = self.df.reset_index(drop=True)
#         print(f"  Dataset size: {len(self.df)} rows")

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         rel_path = row['Path'].replace('CheXpert-v1.0/', '')
#         img_path = os.path.join(self.image_base, rel_path)
#         img = Image.open(img_path).convert('L')
#         if self.transform:
#             img = self.transform(img)
#         labels = torch.tensor(row[PATHOLOGIES].values.astype(np.float32))
#         return img, labels


# def build_datasets(csv_path, image_base):
#     df = pd.read_csv(csv_path)
#     frontal_df = df[df['Frontal/Lateral'] == 'Frontal'].reset_index(drop=True)
#     lateral_df = df[df['Frontal/Lateral'] == 'Lateral'].reset_index(drop=True)
#     print(f"Frontal: {len(frontal_df)} rows | Lateral: {len(lateral_df)} rows")
#     print("--- FRONTAL ---")
#     frontal_ds = CheXpertDataset(frontal_df, image_base)
#     print("--- LATERAL ---")
#     lateral_ds = CheXpertDataset(lateral_df, image_base)
#     return {'frontal': frontal_ds, 'lateral': lateral_ds}


# class BasicCNN(nn.Module):
#     def __init__(self, num_classes=9):
#         super().__init__()
#         self.block1 = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.block2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.classifier = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(64, num_classes)  # 9 outputs, one per pathology
#         )

#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         return self.classifier(x)


# def make_loss(dataset):
#     # compute pos_weight for each of the 9 pathologies
#     labels = dataset.df[PATHOLOGIES].values
#     n_pos = (labels == 1.0).sum(axis=0)
#     n_neg = (labels == 0.0).sum(axis=0)
#     # avoid division by zero
#     n_pos = np.where(n_pos == 0, 1, n_pos)
#     pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)
#     print(f"  pos_weights: {pos_weight.numpy().round(2)}")
#     return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# def train_one_epoch(model, loader, criterion, optimizer, device):
#     model.train()
#     total_loss = 0
#     for images, labels in loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
#         logits = model(images)
#         loss = criterion(logits, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     return total_loss / len(loader)


# if __name__ == "__main__":
#     CSV_PATH   = '/resnick/groups/CS156b/from_central/data/student_labels/train2023.csv'
#     IMAGE_BASE = '/resnick/groups/CS156b/from_central/data/'
#     EPOCHS     = 5
#     BATCH_SIZE = 64

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     datasets = build_datasets(CSV_PATH, IMAGE_BASE)

#     os.makedirs("models", exist_ok=True)

#     for view in ['frontal', 'lateral']:
#         print(f"\n=== Training: {view} ===")
#         ds = datasets[view]
#         loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

#         model = BasicCNN(num_classes=len(PATHOLOGIES)).to(device)
#         criterion = make_loss(ds).to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#         for epoch in range(EPOCHS):
#             loss = train_one_epoch(model, loader, criterion, optimizer, device)
#             print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")

#         save_path = f"/groups/CS156b/from_central/2026/JDP/dean_folder/models/{view}_basiccnn.pt"
#         torch.save(model.state_dict(), save_path)
#         print(f"  Saved to {save_path}")


print("hello world.!")