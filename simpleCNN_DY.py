import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocess_DY import ChestXrayDataset, TRANSFORM, save_dir, train_path

train_path = '/resnick/groups/CS156b/from_central/data'
save_dir   = '/resnick/groups/CS156b/from_central/2026/JDP/dean_folder'

frontal_dataset = ChestXrayDataset(
    os.path.join(save_dir, 'frontal_train.csv'), train_path, view='all', transform=TRANSFORM)

NUM_LABELS = 9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256), nn.ReLU(),
            nn.Linear(256, NUM_LABELS),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model     = SimpleCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

frontal_loader = DataLoader(
    frontal_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

NUM_EPOCHS = 5
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in frontal_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {total_loss / len(frontal_loader):.4f}")

torch.save(model.state_dict(), os.path.join(save_dir, 'simple_cnn_frontal.pth'))
print("Model saved.")
