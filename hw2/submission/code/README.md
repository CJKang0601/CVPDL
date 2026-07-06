# YOLOv8m 多類別物件偵測 - README

本專案使用 YOLOv8m 架構進行交通場景的多類別物件偵測，針對長尾分布資料集進行優化。

## 目錄
- [環境需求](#環境需求)
- [安裝步驟](#安裝步驟)
- [資料集準備](#資料集準備)
- [訓練模型](#訓練模型)
- [執行預測](#執行預測)
- [專案結構](#專案結構)
- [超參數調整](#超參數調整)
- [常見問題](#常見問題)

---

## 環境需求

### 硬體需求
- **GPU:** NVIDIA GPU 具備 CUDA 支援（建議 8GB+ VRAM）
- **記憶體:** 16GB+ RAM
- **儲存空間:** 10GB+ 可用空間

### 軟體需求
- **作業系統:** Windows 10/11, Linux, macOS
- **Python:** 3.8 或以上版本
- **CUDA:** 11.8 或以上（GPU 訓練）
- **cuDNN:** 對應 CUDA 版本

---

## 安裝步驟

### 1. 克隆或下載專案

```bash
# 解壓縮提交的 zip 檔案
unzip hw2_RE6144051.zip
cd hw2_RE6144051
```

### 2. 建立虛擬環境（建議）

```bash
# 使用 conda
conda create -n yolov8m python=3.9
conda activate yolov8m

# 或使用 venv
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. 安裝相依套件

#### 方法一：使用 requirements.txt（推薦）

```bash
pip install -r requirements.txt
```

#### 方法二：手動安裝（如果遇到版本衝突）

```bash
# 安裝 PyTorch（根據您的 CUDA 版本）
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安裝 ultralytics 和其他套件
pip install ultralytics pandas scikit-learn tqdm pillow
```

### 4. 驗證安裝

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('Ultralytics YOLO installed successfully')"
```

預期輸出：
```
PyTorch: 2.x.x
CUDA Available: True
Ultralytics YOLO installed successfully
```

---

## 資料集準備

### 資料集結構

確保您的資料集按照以下結構組織：

```
CVPDL_hw2/
├── train/
│   ├── img0001.png
│   ├── img0001.txt  # 標註檔案
│   ├── img0002.png
│   ├── img0002.txt
│   └── ...
└── test/
    ├── img0001.png
    ├── img0002.png
    └── ...
```

### 標註格式

訓練集的 `.txt` 標註檔案格式為逗號分隔：
```
class_id,x,y,width,height
0,100.5,200.3,50.2,80.1
2,300.0,400.0,20.5,30.8
```

- `class_id`: 0=car, 1=hov, 2=person, 3=motorcycle
- `x, y`: 邊界框左上角座標（像素）
- `width, height`: 邊界框寬度和高度（像素）

### 修改程式碼中的路徑

在 `train.py` 中修改以下路徑：

```python
BASE_PATH = Path(r"D:\CJK\114-1\CV\hw2_RE6144051\CVPDL_hw2\CVPDL_hw2")  # 修改為您的路徑
TRAIN_PATH = BASE_PATH / "train"
TEST_PATH = BASE_PATH / "test"
OUTPUT_PATH = BASE_PATH / "output_yolov8m_multiclass"
```

---

## 訓練模型

### 基本訓練

```bash
python src/train.py
```

### 訓練過程說明

1. **資料集準備階段：**
   - 讀取訓練影像和標註
   - 分割訓練集/驗證集（75%/25%）
   - 轉換為 YOLO 格式
   - 分析類別分布

2. **訓練階段：**
   - 從零開始訓練 YOLOv8m 模型
   - 總共 85 個 epoch
   - 自動儲存最佳模型（best.pt）
   - 早停機制（patience=15）

3. **驗證階段：**
   - 計算 mAP50、mAP50-95 等指標
   - 生成各類別性能報告

4. **預測階段：**
   - 對測試集進行批次預測
   - 應用 Per-Class Confidence Threshold
   - 生成提交檔案（CSV 格式）

### 訓練時間估計

- **GPU (RTX 3080):** 約 3-4 小時
- **GPU (RTX 4090):** 約 2-3 小時
- **CPU (不建議):** 約 30+ 小時

### 監控訓練進度

訓練過程中會顯示：
- 當前 epoch 和損失值
- 驗證集 mAP 指標
- 進度條和預估剩餘時間

輸出的模型和日誌將儲存在：
```
output_yolov8m_multiclass/
└── yolov8m_multiclass_scratch/
    ├── weights/
    │   ├── best.pt      # 最佳模型
    │   └── last.pt      # 最後一個 epoch
    ├── results.png      # 訓練曲線
    └── confusion_matrix.png
```

---

## 執行預測

### 使用訓練好的模型預測

如果您只想使用已訓練的模型進行預測：

```python
from ultralytics import YOLO
from pathlib import Path

# 載入模型
model = YOLO('output_yolov8m_multiclass/yolov8m_multiclass_scratch/weights/best.pt')

# 預測單張影像
results = model.predict(
    source='test/img0001.png',
    conf=0.001,  # 最低信心度閾值
    iou=0.5,
    imgsz=1024,
    augment=True  # 啟用 TTA
)

# 顯示結果
results[0].show()
```

### 批次預測測試集

程式碼中已包含測試集預測，會自動：
1. 讀取測試集所有影像
2. 批次預測（batch size = 8）
3. 應用 Per-Class Confidence Threshold
4. 生成 `submission_yolov8m.csv`

提交檔案格式：
```csv
Image_ID,PredictionString
1,0.95 100.0 200.0 50.0 80.0 0 0.88 300.0 400.0 60.0 90.0 0
2,0.75 150.0 250.0 40.0 70.0 2
```

---

## 專案結構

```
hw2_RE6144051/
├── report_RE6144051.pdf          # 實驗報告（3-5 頁）
├── code_RE6144051.zip
│   ├── src/
│   │   └── model.py               # 主要訓練和預測腳本
│   ├── readme.md                  # 本文件
│   └── requirements.txt           # 套件清單
├── CVPDL_hw2/                     # 資料集（不包含在提交中）
│   ├── train/
│   └── test/
└── output_yolov8m_multiclass/     # 輸出結果
    ├── dataset/                   # YOLO 格式資料集
    ├── yolov8m_multiclass_scratch/ # 訓練日誌和模型
    └── submission_yolov8m.csv     # 預測結果
```

---

## 超參數調整

### 關鍵超參數

如需調整模型性能，可以修改以下參數：

#### 1. Per-Class Confidence Threshold

```python
PER_CLASS_THRESHOLDS = {
    0: 0.02,   # car - 調高以減少誤檢
    1: 0.01,   # hov
    2: 0.001,  # person - 調低以增加召回率
    3: 0.01,   # motorcycle
}
```

**調整建議：**
- 如果某類別誤檢過多 → 調高閾值
- 如果某類別漏檢嚴重 → 調低閾值

#### 2. 訓練 Epoch 數

```python
epochs = 85  # 增加以改善性能，但注意過擬合
patience = 15  # 早停耐心值
```

#### 3. Batch Size

```python
batch = 4  # 根據 GPU 記憶體調整
# 8GB VRAM → batch=2-4
# 12GB VRAM → batch=4-8
# 24GB VRAM → batch=8-16
```

#### 4. 資料增強強度

```python
mosaic = 1.0    # 降低以減少訓練難度
mixup = 0.2     # 增加以改善類別混淆
copy_paste = 0.3  # 增加以強化少數類別
```

#### 5. 學習率

```python
lr0 = 0.0001   # 初始學習率（降低以穩定訓練）
lrf = 0.01     # 最終學習率比例
```

### 性能調優策略

1. **改善 Recall（召回率）：**
   - 降低 Per-Class Threshold
   - 增加 copy_paste 和 mixup
   - 延長訓練 epoch

2. **改善 Precision（精確度）：**
   - 提高 Per-Class Threshold
   - 增加 label_smoothing
   - 降低資料增強強度

3. **改善小物體偵測：**
   - 增加 mosaic 比例
   - 使用更大的 imgsz（如 1280）
   - 針對小物體過採樣

---

## 常見問題

### Q1: CUDA Out of Memory

**症狀：** 訓練時出現 `RuntimeError: CUDA out of memory`

**解決方法：**
```python
# 1. 降低 batch size
batch = 2  # 或更小

# 2. 降低影像尺寸
imgsz = 896  # 從 1024 降低

# 3. 關閉多尺度訓練
multi_scale = False
```

### Q2: 訓練速度過慢

**解決方法：**
```python
# 1. 增加 workers 數量
workers = 8  # 根據 CPU 核心數調整

# 2. 啟用快取
cache = 'ram'  # 或 'disk'

# 3. 關閉不必要的增強
mosaic = 0.5
```

### Q3: 模型過擬合

**症狀：** 訓練 loss 持續下降，但驗證 mAP 不再提升

**解決方法：**
```python
# 1. 降低 epochs
epochs = 50

# 2. 增加 dropout
dropout = 0.1

# 3. 增加資料增強
mixup = 0.3
copy_paste = 0.5
```

### Q4: 少數類別效果差

**解決方法：**
```python
# 1. 調整 Per-Class Threshold
PER_CLASS_THRESHOLDS[2] = 0.0005  # 進一步降低

# 2. 增加針對性增強
copy_paste = 0.5  # 特別針對少數類別

# 3. 考慮使用 Focal Loss（需修改程式碼）
```

### Q5: 如何使用 CPU 訓練

雖然不建議，但如需使用 CPU：

```python
# 在 train.py 中修改
device = 'cpu'

# 調整參數
batch = 1
workers = 2
cache = False
```

### Q6: 如何繼續訓練

```python
# 載入已訓練的模型
model = YOLO('output_yolov8m_multiclass/yolov8m_multiclass_scratch/weights/last.pt')

# 繼續訓練
model.train(
    data=yaml_path,
    epochs=20,  # 額外的 epoch 數
    resume=True
)
```




**最後更新：** 2025 年 11 月  
**作者：** RE6144051  
**版本：** 1.0