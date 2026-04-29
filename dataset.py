import os
import shutil
import random

# ---------------------------
# 設定路徑
# ---------------------------
base_dir = r"D:\car"                       # 原始資料夾（含 green, red, ...）
dataset_dir = os.path.join(base_dir, "dataset")  # 新資料集總資料夾

train_dir = os.path.join(dataset_dir, "train")
val_dir   = os.path.join(dataset_dir, "val")
test_dir  = os.path.join(dataset_dir, "test")

# 拆分比例
train_ratio = 0.7
val_ratio   = 0.15
test_ratio  = 0.15

# ---------------------------
# 建立 train/val/test 資料夾
# ---------------------------
for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)

# ---------------------------
# 取得類別（跳過 dataset 資料夾）
# ---------------------------
classes = [d for d in os.listdir(base_dir) 
           if os.path.isdir(os.path.join(base_dir, d)) 
           and d != "dataset"]  # 跳過 dataset 資料夾

# ---------------------------
# 拆分資料
# ---------------------------
for cls in classes:
    cls_path = os.path.join(base_dir, cls)
    images = [f for f in os.listdir(cls_path) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)
    
    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)
    
    splits = {
        train_dir: images[:n_train],
        val_dir:   images[n_train:n_train+n_val],
        test_dir:  images[n_train+n_val:]
    }
    
    for split_dir, split_images in splits.items():
        cls_split_dir = os.path.join(split_dir, cls)
        os.makedirs(cls_split_dir, exist_ok=True)
        for img in split_images:
            src = os.path.join(cls_path, img)
            dst = os.path.join(cls_split_dir, img)
            shutil.copy2(src, dst)

print("✅ 資料集拆分完成！")
