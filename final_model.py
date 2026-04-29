import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
from torchinfo import summary

# ---------------------------
# 固定隨機種子（可重現）
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ---------------------------
# 裝置與路徑
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = r"D:\car\dataset"
save_dir = r"D:\car"
plot_dir = os.path.join(save_dir, "plots")
model_dir = os.path.join(save_dir, "models")

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# ---------------------------
# Early Stopping
# ---------------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.stop = False

    def step(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

# ---------------------------
# ✅ 最終資料增強（Step4）
# ---------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(64, scale=(0.8,1.0)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomAffine(degrees=15, translate=(0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# ---------------------------
# Dataset
# ---------------------------
train_ds = datasets.ImageFolder(os.path.join(data_dir,"train"), transform=train_transform)
val_ds   = datasets.ImageFolder(os.path.join(data_dir,"val"),   transform=test_transform)
test_ds  = datasets.ImageFolder(os.path.join(data_dir,"test"),  transform=test_transform)

num_classes = len(train_ds.classes)

# ---------------------------
# DataLoader（最終設定）
# ---------------------------
batch_size = 32

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

# ---------------------------
# Tiny CNN Model
# ---------------------------
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )

    def forward(self,x):
        return self.classifier(self.features(x))

model = CNN(num_classes).to(device)

# ---------------------------
# 最終超參數設定
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 100
early_stopper = EarlyStopping(patience=10)

# ---------------------------
# 紀錄用
# ---------------------------
history = {
    "train_acc": [],
    "val_acc": [],
    "train_loss": [],
    "val_loss": []
}
summary(
    model,
    input_size=(1, 3, 64, 64),
    col_names=["input_size", "output_size", "num_params"],
    device=device
)
# ---------------------------
# Training Loop
# ---------------------------
for epoch in range(num_epochs):
    # Train
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*x.size(0)
        correct += (out.argmax(1)==y).sum().item()
        total += y.size(0)

    train_acc = correct/total
    train_loss /= total

    # Validation
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out,y)
            val_loss += loss.item()*x.size(0)
            correct += (out.argmax(1)==y).sum().item()
            total += y.size(0)

    val_acc = correct/total
    val_loss /= total

    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    early_stopper.step(val_loss)
    if early_stopper.stop:
        print(f"📌 Early stopping at epoch {epoch+1}")
        break

# ---------------------------
# Test
# ---------------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x,y in test_loader:
        x,y = x.to(device), y.to(device)
        out = model(x)
        correct += (out.argmax(1)==y).sum().item()
        total += y.size(0)

test_acc = correct/total
print(f"✅ Final Test Accuracy: {test_acc:.4f}")

# ---------------------------
# Save model
# ---------------------------
model_path = os.path.join(model_dir,"final.pth")
torch.save(model.state_dict(), model_path)

# ---------------------------
# Visualization
# ---------------------------
epochs_range = range(1,len(history["train_acc"])+1)

plt.figure(figsize=(12,5))



# Accuracy
plt.subplot(1,2,1)
plt.plot(epochs_range, history["train_acc"], label="Train Acc")
plt.plot(epochs_range, history["val_acc"], label="Val Acc")
plt.axhline(test_acc, linestyle="--", color="red", label=f"Test Acc {test_acc:.4f}")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1,2,2)
plt.plot(epochs_range, history["train_loss"], label="Train Loss")
plt.plot(epochs_range, history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir,"final_training_result.png"))
plt.show()
