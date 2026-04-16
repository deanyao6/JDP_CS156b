from preprocess_DY import CheXpertDataset, pad_to_square
import os
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim

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

# datasets
frontal_dataset = CheXpertDataset(os.path.join(save_dir, 'frontal_train.csv'), train_path, view='all', transform=TRANSFORM)
lateral_dataset = CheXpertDataset(os.path.join(save_dir, 'lateral_train.csv'), train_path, view='all', transform=TRANSFORM)

# simple CNN
NUM_LABELS = 9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 224 → 112
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 112 → 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 56  → 28
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256), nn.ReLU(),
            nn.Linear(256, NUM_LABELS)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# train
model = SimpleCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# small_frontal = Subset(frontal_dataset, range(1000))
frontal_loader = DataLoader(frontal_dataset, batch_size=64, shuffle=True, num_workers=1, pin_memory=True)

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

torch.save(model.state_dict(), os.path.join(save_dir, 'simple_cnn.pth'))
print("Model saved.")