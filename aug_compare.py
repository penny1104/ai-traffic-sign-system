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

# ---------------------------
# 固定隨機種子
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
# 設定裝置與路徑
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = r"D:\car\dataset"
save_dir = r"D:\car"
plot_dir = os.path.join(save_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)

# ---------------------------
# EarlyStopping 類別
# ---------------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, val_loss):
        score = -val_loss  # 驗證損失越小越好
        if self.best_score is None:
            self.best_score = score
            return
        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# ---------------------------
# 定義逐步資料增強
# ---------------------------
transform_steps = [
    transforms.Compose([  # Step0: Baseline
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ]),
    transforms.Compose([  # Step1: RandomResizedCrop
        transforms.RandomResizedCrop(64, scale=(0.8,1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ]),
    transforms.Compose([  # Step2: + RandomRotation
        transforms.RandomResizedCrop(64, scale=(0.8,1.0)),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ]),
    transforms.Compose([  # Step3: + ColorJitter
        transforms.RandomResizedCrop(64, scale=(0.8,1.0)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ]),
    transforms.Compose([  # Step4: + RandomAffine
        transforms.RandomResizedCrop(64, scale=(0.8,1.0)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1,0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ]),
    transforms.Compose([  # Step5: + RandomGrayscale
        transforms.RandomResizedCrop(64, scale=(0.8,1.0)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1,0.1)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
]

# ---------------------------
# Tiny CNN 模型
# ---------------------------
class CNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CNN, self).__init__()
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

# ---------------------------
# 訓練設定
# ---------------------------
num_epochs = 100
batch_size = 32
total_csv_path = os.path.join(plot_dir, "training_history_all_steps.csv")

with open(total_csv_path, "w", newline="", encoding="utf-8") as f_total:
    writer_total = csv.writer(f_total)
    writer_total.writerow(["step","epoch","train_loss","train_acc","val_loss","val_acc","test_acc"])

all_step_histories = []
step_test_accs = []

for step_idx, transform_train in enumerate(transform_steps):
    print(f"\n=== Step {step_idx} ===")
    
    train_dataset = datasets.ImageFolder(os.path.join(data_dir,"train"), transform=transform_train)
    val_dataset   = datasets.ImageFolder(os.path.join(data_dir,"val"), transform=transform_steps[0])
    test_dataset  = datasets.ImageFolder(os.path.join(data_dir,"test"), transform=transform_steps[0])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    num_classes = len(train_dataset.classes)
    model = CNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    early_stopper = EarlyStopping(patience=10)
    
    for epoch in range(num_epochs):
        # 訓練
        model.train()
        running_loss, correct, total = 0.0,0,0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*images.size(0)
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
        train_loss = running_loss/total
        train_acc = correct/total
        
        # 驗證
        model.eval()
        v_loss, v_correct, v_total = 0.0,0,0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                v_loss += loss.item()*images.size(0)
                _, predicted = torch.max(outputs,1)
                v_total += labels.size(0)
                v_correct += (predicted==labels).sum().item()
        val_loss = v_loss/v_total
        val_acc = v_correct/v_total
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # CSV
        with open(total_csv_path, "a", newline="", encoding="utf-8") as f_total:
            writer_total = csv.writer(f_total)
            writer_total.writerow([step_idx, epoch+1, train_loss, train_acc, val_loss, val_acc, ""])
        
        # Early stopping
        early_stopper.step(val_loss)
        if early_stopper.early_stop:
            print(f"📌 Step {step_idx} Early stopping at epoch {epoch+1}")
            break
    
    # Test Accuracy
    model.eval()
    test_correct, test_total = 0,0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs,1)
            test_total += labels.size(0)
            test_correct += (predicted==labels).sum().item()
    test_acc = test_correct/test_total
    step_test_accs.append(test_acc)
    print(f"📌 Step {step_idx} Test Accuracy: {test_acc:.4f}")
    
    # 更新 CSV Test Accuracy
    rows=[]
    with open(total_csv_path,"r",newline="",encoding="utf-8") as f_read:
        reader=csv.reader(f_read)
        header=next(reader)
        for row in reader:
            if int(row[0])==step_idx:
                row[6]=test_acc
            rows.append(row)
    with open(total_csv_path,"w",newline="",encoding="utf-8") as f_write:
        writer = csv.writer(f_write)
        writer.writerow(header)
        writer.writerows(rows)
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(save_dir,"models",f"cnn_step{step_idx}.pth"))
    
    all_step_histories.append(history)
    
    # ---------------------------
    # 單步視覺化
    # ---------------------------
    plt.figure(figsize=(12,5))
    epochs_range = range(1,len(history["train_acc"])+1)
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(epochs_range, history["train_acc"], label="Train Acc")
    plt.plot(epochs_range, history["val_acc"], label="Val Acc")
    plt.axhline(test_acc, linestyle='--', color='r', alpha=0.5, label=f"Test Acc {test_acc:.4f}")
    plt.title(f"Step {step_idx} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    # Loss
    plt.subplot(1,2,2)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    plt.title(f"Step {step_idx} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"step{step_idx}_accuracy_loss.png"))
    plt.close()

# ---------------------------
# 總圖
# ---------------------------
plt.figure(figsize=(14,6))
for step_idx, history in enumerate(all_step_histories):
    epochs_range = range(1,len(history["train_acc"])+1)
    plt.subplot(1,2,1)
    plt.plot(epochs_range, history["train_acc"], label=f"Step {step_idx} Train")
    plt.plot(epochs_range, history["val_acc"], label=f"Step {step_idx} Val")
    plt.axhline(step_test_accs[step_idx], linestyle='--', alpha=0.5, label=f"Step {step_idx} Test {step_test_accs[step_idx]:.4f}")
    
    plt.subplot(1,2,2)
    plt.plot(epochs_range, history["train_loss"], label=f"Step {step_idx} Train Loss")
    plt.plot(epochs_range, history["val_loss"], label=f"Step {step_idx} Val Loss")

plt.subplot(1,2,1)
plt.title("Train/Val Accuracy Across Steps")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.grid(True)
plt.subplot(1,2,2)
plt.title("Train/Val Loss Across Steps")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir,"accuracy_loss_all_steps.png"))
plt.show()
