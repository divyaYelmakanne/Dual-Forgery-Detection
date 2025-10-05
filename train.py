import torch, os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
class ForgeryDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform):
        self.real = [(os.path.join(real_dir, f), 0) for f in os.listdir(real_dir) if f.endswith(('.png', '.jpg'))]
        self.fake = [(os.path.join(fake_dir, f), 1) for f in os.listdir(fake_dir) if f.endswith(('.png', '.jpg'))]
        self.data = self.real + self.fake
        self.transform = transform

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label

# Paths
real_dir = './dataset/real'
fake_dir = './dataset/fake'

# Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Dataset & Split
full_dataset = ForgeryDataset(real_dir, fake_dir, transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*64*64, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.fc(x)

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(10):
    model.train()
    total, correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}: Training Accuracy = {acc:.2f}%")

# ✅ Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# 📊 Metrics
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))

# 📉 Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(' Confusion Matrix')
plt.tight_layout()
plt.show()

# Save model
torch.save(model.state_dict(), "forgery_classifier.pth")
print(" Model saved to forgery_classifier.pth")
