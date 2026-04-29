# 🚗 我的小車不是三寶：視覺辨識下的交通守規挑戰
## 📌 專案簡介

本專案為基於 Raspberry Pi（樹莓派）自走車系統，結合 卷積神經網路（CNN） 與 超音波感測技術，實現交通號誌辨識與即時避障控制，模擬自動駕駛中的「感知 → 判斷 → 行動」流程。

## 🎯 專案目標

本專案旨在建立一套具備以下能力的嵌入式 AI 系統：

交通號誌即時辨識（Traffic Sign Recognition）
基於距離感測的自動避障
自走車行為決策系統（Decision System）
模擬自動駕駛核心感知架構
## 🧠 系統架構（Perception → Decision → Control）
### 📌 感知層（Perception）
攝影機影像輸入
超音波距離感測
CNN 交通號誌辨識
### 📌 決策層（Decision）

根據以下條件進行行為判斷：

紅燈 / Stop → 停止或後退右轉
綠燈 / None → 直行
左 / 右轉 → 對應轉向
慢 → 減速
距離 < 40cm → 觸發避障
### 📌 控制層（Control）
馬達控制（前進 / 後退 / 轉向）
即時反應控制迴圈
## 🛠️ 使用技術
Python
PyTorch（CNN 模型）
OpenCV（影像處理）
Raspberry Pi（嵌入式控制）
超音波感測器（距離偵測）
GPIO 馬達控制
## 🧠 CNN 模型設計
### 📌 模型輸入

3×64×64

### 📌 模型架構
Conv + ReLU + MaxPool（16 / 32 / 64 channels）
Flatten（4096 features）
Fully Connected（128 neurons）
Output Layer（6 classes）
## 📊 資料前處理與增強
### 📌 基本處理

64×64

Resize
Normalization
Tensor 轉換
### 📌 Data Augmentation
隨機旋轉（±15°）
隨機裁切
Color Jitter
仿射變換與平移
## ⚙️ 訓練設定（Hyperparameters）
Learning Rate：0.001
Batch Size：32
Optimizer：Adam
### 📌 模型表現
Training Accuracy：99.26%
Testing Accuracy：100%
## 🚥 行為控制邏輯（Car Logic）
### 📌 超音波避障
距離 < 40cm → 停止 → 後退 → 左轉
### 📌 影像辨識觸發
距離 < 75cm → 啟動辨識
### 📌 行為規則
紅燈 / Stop → 停止或後退右轉
綠燈 / None → 直行
左 / 右轉 → 對應轉向
慢 → 減速
## 📈 專案成果
### ✅ 成功點
完成 Raspberry Pi 嵌入式 AI 系統
實現即時影像辨識 + 避障控制
建立自動駕駛感知模型
### ⚠️ 挑戰
遠距離辨識準確率下降
光線影響模型表現
馬達受電力與摩擦影響
### 🚀 未來優化方向
使用 Transfer Learning 提升準確率
加入夜間 / 雨天資料集
即時攝影機辨識
模型部署為 Web App
強化控制系統穩定性
### 💡 專案定位

本專案屬於：

Computer Vision + Embedded AI + Robotics 系統整合專案
