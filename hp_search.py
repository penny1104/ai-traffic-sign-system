import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os, csv, random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------
# 固定隨機種子
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ---------------------------
# 設定裝置與路徑
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = r"D:\car\dataset"
save_dir = r"D:\car"
plot_dir = os.path.join(save_dir, "plots")
model_dir = os.path.join(save_dir, "models")

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# ---------------------------
# Data Augmentation
# ---------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
    transforms.RandomAffine(degrees=15, translate=(0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ---------------------------
# Dataset
# ---------------------------
train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=test_transform)
test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=test_transform)

num_classes = len(train_ds.classes)

# ---------------------------
# CNN Model
# ---------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(self.features(x))

# ---------------------------
# Early Stopping
# ---------------------------
class EarlyStopping:
    def __init__(self, patience=10):
        self.best = None
        self.counter = 0
        self.patience = patience
        self.stop = False

    def step(self, val_loss):
        if self.best is None or val_loss < self.best:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

# ---------------------------
# Hyperparameter Search
# ---------------------------
learning_rates = [1e-3, 5e-4]
batch_sizes = [32, 64]
optimizers = ["Adam", "SGD"]

csv_path = os.path.join(plot_dir, "hyperparam_search_results.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(
        ["lr", "batch_size", "optimizer", "best_train_acc", "best_val_acc", "test_acc"]
    )

# ---------------------------
# Hyperparameter Search Loop
# ---------------------------
for lr in learning_rates:
    for bs in batch_sizes:
        for opt_name in optimizers:

            print(f"\n🔍 LR={lr}, BS={bs}, OPT={opt_name}")

            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=bs)
            test_loader  = DataLoader(test_ds, batch_size=bs)

            model = CNN().to(device)
            criterion = nn.CrossEntropyLoss()

            optimizer = (
                optim.Adam(model.parameters(), lr=lr)
                if opt_name == "Adam"
                else optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            )

            early = EarlyStopping(patience=10)

            best_val_acc = 0.0
            best_train_acc = 0.0
            best_model_state = None

            for epoch in range(100):

                # -------- Train --------
                model.train()
                correct, total = 0, 0

                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)

                    optimizer.zero_grad()
                    out = model(x)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()

                    correct += (out.argmax(1) == y).sum().item()
                    total += y.size(0)

                train_acc = correct / total

                # -------- Validation --------
                model.eval()
                correct, total, val_loss = 0, 0, 0

                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        out = model(x)
                        loss = criterion(out, y)

                        val_loss += loss.item()
                        correct += (out.argmax(1) == y).sum().item()
                        total += y.size(0)

                val_acc = correct / total

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_train_acc = train_acc
                    best_model_state = model.state_dict()

                early.step(val_loss)
                if early.stop:
                    break

            # -------- Save best model --------
            model_name = f"cnn_lr{lr}_bs{bs}_{opt_name}.pth"
            torch.save(best_model_state, os.path.join(model_dir, model_name))

            # -------- Test --------
            model.load_state_dict(best_model_state)
            model.eval()
            correct, total = 0, 0

            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total += y.size(0)

            test_acc = correct / total

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [lr, bs, opt_name, best_train_acc, best_val_acc, test_acc]
                )

            print(
                f"✅ Train={best_train_acc:.4f}, "
                f"Val={best_val_acc:.4f}, "
                f"Test={test_acc:.4f}, "
                f"Saved={model_name}"
            )

# ---------------------------
# 視覺化
# ---------------------------
df = pd.read_csv(csv_path)

plt.figure(figsize=(12,5))
plt.plot(df["best_train_acc"], label="Train")
plt.plot(df["best_val_acc"], label="Val")
plt.plot(df["test_acc"], label="Test")
plt.legend()
plt.title("Hyperparameter Search Accuracy")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "hyperparam_search_accuracy.png"))
plt.show()
