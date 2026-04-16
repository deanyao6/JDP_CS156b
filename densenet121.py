from preprocess_DY import CheXpertDataset, pad_to_square
import os
from torchvision import transforms, models
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

# densenet121
NUM_LABELS = 9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# frontal model
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
model.classifier = nn.Linear(model.classifier.in_features, NUM_LABELS)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

small_frontal = Subset(frontal_dataset, range(5000))
frontal_loader = DataLoader(small_frontal, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

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

# torch.save(model.state_dict(), os.path.join(save_dir, 'densenet121_frontal.pth'))
torch.save(model.state_dict(), os.path.join(save_dir, 'densenet121_frontal_small.pth'))
print("Frontal Model saved.")

# lateral model
# reinitialize for lateral
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
model.classifier = nn.Linear(model.classifier.in_features, NUM_LABELS)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

small_lateral = Subset(lateral_dataset, range(5000))
lateral_loader = DataLoader(lateral_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

NUM_EPOCHS = 5
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in lateral_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {total_loss / len(lateral_loader):.4f}")

# torch.save(model.state_dict(), os.path.join(save_dir, 'densenet121_lateral.pth'))
torch.save(model.state_dict(), os.path.join(save_dir, 'densenet121_lateral_small.pth'))
print("Lateral Model saved.")