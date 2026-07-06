# YOLOv8 豬隻物件偵測專案

本專案旨在訓練一個 YOLOv8s 模型來偵測影像中的豬隻。專案採用遷移學習策略，將預訓練的 YOLOv8s 模型骨幹網路作為固定的特徵提取器 (Feature Extractor)，並僅訓練其偵測頭 (Detection Head) 以適應新的任務。

## 專案結構

```
.
|-- src/
|   |-- model.py             # 主執行腳本 (包含資料處理、訓練、驗證與預測)
|-- readme.md               # 本說明檔案
|-- requirements.txt        # Python 環境依賴包
```

## 環境安裝

1.  **確認 Python 版本**
    建議使用 Python 3.8 或更高版本。

2.  **建立虛擬環境 (建議)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **安裝依賴包**
    所有必要的 Python 套件都已列在 `requirements.txt` 中。請執行以下指令進行安裝：
    ```bash
    pip install -r requirements.txt
    ```
    **注意:** 使用 GPU 進行訓練，請確保您已安裝與 PyTorch 相容的 NVIDIA 驅動程式與 CUDA Toolkit。`ultralytics` 套件會自動安裝 PyTorch，通常會包含 CUDA 版本。

## 如何執行

### 1. 資料集準備

在執行腳本前，請確保您的資料夾結構如下。將 `HW1` 資料夾放置在您本機的任意位置 (例如桌面)。

```
C:/Users/YourUser/Desktop/HW1/
|-- train/
|   |-- img/
|   |   |-- 00000001.jpg
|   |   |-- ...
|   |-- gt.txt
|-- test/
|   |-- img/
|   |   |-- 00004121.jpg
|   |   |-- ...
```

### 2. 修改腳本中的路徑

打開 `src/model.py` 檔案，找到 `BASE_PATH` 這個變數，並將其路徑修改為您本機上 `HW1` 資料夾所在的絕對路徑。

```python
# line 24
BASE_PATH = Path(r"C:\Users\YourUser\Desktop\HW1") # <--- 請修改此路徑
```

### 3. 執行訓練與預測

本腳本是一個端到端 (End-to-End) 的流程。執行單一指令即可完成 **資料準備**、**模型訓練**、**驗證** 以及 **生成最終預測檔案** 的所有步驟。

在終端機中，確保您位於 `code_<student-id>` 資料夾的根目錄下，然後執行：

```bash
python src/main.py
```

### 4. 查看結果

* **訓練過程與模型**：所有的訓練日誌、權重檔案 (`best.pt`) 和視覺化結果都會被儲存在 `BASE_PATH` 下的 `output_yolov8_freeze` 資料夾中。
* **最終提交檔案**：腳本執行