"""
- 效率優化：圖片尺寸硬編碼為 640x360
- 訓練策略：yolov8s當作 backbone freeze 骨幹 只訓練 detection head, 100 epochs, AdamW
- 競賽策略：啟用 TTA, 調整 NMS 參數 (conf=0.05, iou=0.65)
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import torch
import subprocess


# ============================================================
# 1. 環境、常數與路徑設定
# ============================================================

# 圖片固定尺寸
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
CLASS_ID = 0 # 只有一類 (pig)

# 路徑配置 
BASE_PATH = Path(r"C:\Users\cjkan\Desktop\CJK\114-1\CV\HW1")
TRAIN_PATH = BASE_PATH / "train"
TEST_PATH = BASE_PATH / "test"
OUTPUT_PATH = BASE_PATH / "output_yolov8_freeze"


# install ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    print("安裝 ultralytics...")
    subprocess.run(['pip', 'install', 'ultralytics'], check=True)
    from ultralytics import YOLO



# ============================================================
# 2. 函數定義
# ============================================================

def convert_to_yolo(bbox):
    """轉換為 YOLO 格式 (正規化的 x_center, y_center, width, height)"""
    x_center = (bbox['x'] + bbox['w'] / 2) / IMAGE_WIDTH
    y_center = (bbox['y'] + bbox['h'] / 2) / IMAGE_HEIGHT
    width = bbox['w'] / IMAGE_WIDTH
    height = bbox['h'] / IMAGE_HEIGHT
    
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_dataset(image_list, split_name, yolo_base, img_dir, annotations):
    """處理資料集：複製圖片並建立標註檔案"""
    for img_name in tqdm(image_list, desc=f'建立 {split_name} 資料'):
        # 複製圖片
        shutil.copy2(
            img_dir / img_name,
            yolo_base / split_name / 'images' / img_name
        )
        
        # 建立標註檔案
        label_file = yolo_base / split_name / 'labels' / img_name.replace('.jpg', '.txt')
        
        if img_name in annotations:
            with open(label_file, 'w') as f:
                for bbox in annotations[img_name]:
                    yolo_line = convert_to_yolo(bbox)
                    f.write(yolo_line + '\n')
        else:
            label_file.touch()  

# ============================================================
# 3. 主執行區塊 (if __name__ == '__main__':)
# ============================================================

if __name__ == '__main__':
    
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    OUTPUT_PATH.mkdir(exist_ok=True)
    
    # --- 3.1 讀取與處理標註資料 ---
    
    gt_file = TRAIN_PATH / 'gt.txt'
    img_dir = TRAIN_PATH / 'img'
    
    # 建立圖片列表
    all_images = sorted([f.name for f in img_dir.glob('*.jpg')])
    filename_to_idx = {int(f.split('.')[0]): f for f in all_images}
    annotations = {}
    
    print(f"\n=== 1. 資料讀取與處理 ===")
    print(f"預期尺寸: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    
    with open(gt_file, 'r') as f:
        for line in tqdm(f, desc="處理 gt.txt"):
            parts = line.strip().split(',')
            if len(parts) < 5: continue
            
            try:
                frame, x, y, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                
                if frame not in filename_to_idx: continue
                img_name = filename_to_idx[frame]
                
                # 邊界裁剪與檢查 (確保框在 640x360 範圍內)
                x_max_raw, y_max_raw = x + w, y + h
                x, y = max(0.0, x), max(0.0, y)
                w = min(IMAGE_WIDTH - x, x_max_raw - x)
                h = min(IMAGE_HEIGHT - y, y_max_raw - y)
                
                if w <= 1.0 or h <= 1.0: continue
                
                if img_name not in annotations: annotations[img_name] = []
                annotations[img_name].append({'x': x, 'y': y, 'w': w, 'h': h})
                
            except (ValueError, IndexError):
                continue
    
    print(f"圖片總數: {len(all_images)}, 有標註: {len(annotations)}, 總標註數: {sum(len(v) for v in annotations.values())}")

    # --- 3.2 建立 YOLO 格式資料集 ---

    print(f"\n=== 2. 建立 YOLO 資料集 ===")
    train_imgs, valid_imgs = train_test_split(all_images, test_size=0.3, random_state=20251005)
    print(f"訓練集: {len(train_imgs)}, 驗證集: {len(valid_imgs)}")

    yolo_base = OUTPUT_PATH / 'dataset'
    if yolo_base.exists():
        shutil.rmtree(yolo_base)

    for split in ['train', 'valid']:
        (yolo_base / split / 'images').mkdir(parents=True)
        (yolo_base / split / 'labels').mkdir(parents=True)

    process_dataset(train_imgs, 'train', yolo_base, img_dir, annotations)
    process_dataset(valid_imgs, 'valid', yolo_base, img_dir, annotations)

    # 建立配置檔案
    yaml_path = OUTPUT_PATH / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"""path: {yolo_base.absolute().as_posix()}
train: train/images
val: valid/images
names:
  {CLASS_ID}: pig
nc: 1
""")
    print("資料集準備完成")
    
    # --- 3.3 訓練模型 ---

    print("\n" + "="*60)
    print("=== 3. 開始訓練 YOLOv8s (Max 100 Epochs) ===")
    print("="*60)

    model = YOLO('yolov8s.pt')
    FREEZE_LAYERS = 10

    results = model.train(
        data=yaml_path.as_posix(),
        epochs=100,             
        imgsz=640,
        batch=32,               
        patience=20,            
        project=OUTPUT_PATH.as_posix(),
        name='yolov8s_run',
        exist_ok=True,
        optimizer='AdamW',
        lr0=0.0001,
        fliplr=0.5,             
        mosaic=1.0,             
        verbose=True,
        seed=42,
        workers=4,              
        device=0 if torch.cuda.is_available() else 'cpu',
        freeze=FREEZE_LAYERS
    )
    print("訓練完成")

    # --- 3.4 模型驗證 ---

    best_model_path = OUTPUT_PATH / 'yolov8s_run' / 'weights' / 'best.pt'
    best_model = YOLO(best_model_path)

    metrics = best_model.val(data=yaml_path.as_posix())
    print(f"\n=== 4. 最終驗證結果 ===")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")

    # --- 3.5 預測測試集 ---

    print("\n=== 5. 預測測試集 (啟用 TTA) ===")

    test_img_dir = TEST_PATH / 'img'
    test_images = sorted([f.name for f in test_img_dir.glob('*.jpg')])

    submission_data = []

    
    BATCH_SIZE = 32 
    CONF_THRESHOLD = 0.05 # 提升召回率
    IOU_THRESHOLD = 0.65 # NMS IOU

    for i in tqdm(range(0, len(test_images), BATCH_SIZE), desc='預測'):
        batch_imgs = test_images[i:i+BATCH_SIZE]
        batch_paths = [test_img_dir / f for f in batch_imgs]
        
        results_list = best_model.predict(
            source=batch_paths,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=640,
            verbose=False,
            augment=True, # 啟用 Test Time Augmentation
        )
        
        for img_name, result in zip(batch_imgs, results_list):
            try:
                img_id = int(img_name.split('.')[0])
            except ValueError:
                 continue

            boxes = result.boxes
            
            if len(boxes) == 0:
                submission_data.append({'Image_ID': img_id, 'PredictionString': ''})
                continue
            
            pred_parts = []
            for box in boxes:
                # YOLOv8 輸出 xyxy 已經是原始像素座標
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()
                
                # 邊界裁剪 (確保在 640x360 範圍內)
                x1 = max(0, min(IMAGE_WIDTH, x1))
                y1 = max(0, min(IMAGE_HEIGHT, y1))
                x2 = max(0, min(IMAGE_WIDTH, x2))
                y2 = max(0, min(IMAGE_HEIGHT, y2))
                
                width = x2 - x1
                height = y2 - y1
                
                # 輸出格式: Conf X Y W H ClassID (ClassID=0)
                pred_parts.append(f"{conf:.6f} {x1:.2f} {y1:.2f} {width:.2f} {height:.2f} {CLASS_ID}")
            
            submission_data.append({
                'Image_ID': img_id,
                'PredictionString': " ".join(pred_parts)
            })

    # --- 3.6 儲存結果與最終驗證 ---

    df = pd.DataFrame(submission_data).sort_values('Image_ID')
    submission_path = OUTPUT_PATH / 'submission.csv'
    df.to_csv(submission_path, index=False)

    print(f"\n=== 6. 提交檔案輸出 ===")
    print(f"提交檔案: {submission_path.as_posix()}")
    print(f"總圖片: {len(df)}")
    print(f"有預測: {len(df[df['PredictionString'] != ''])}")

    # 最終座標驗證
    print("\n--- 座標驗證 (前3張) ---")
    for idx, row in df[df['PredictionString'] != ''].head(3).iterrows():
        print(f"\nImage {row['Image_ID']:08d}.jpg:")
        
        try:
            parts = row['PredictionString'].split()
            box_data = []
            for i in range(0, len(parts), 6):
                if i + 5 < len(parts):
                    conf, x, y, w, h, class_id = map(float, parts[i:i+6])
                    box_data.append((conf, x, y, w, h))
            
            for i, (conf, x, y, w, h) in enumerate(box_data[:3], 1):
                print(f"  Box {i}: conf={conf:.3f}, x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")
                if x < 0 or y < 0 or x + w > IMAGE_WIDTH + 1 or y + h > IMAGE_HEIGHT + 1:
                    print(f"    ⚠️ 警告: 座標超出範圍 ({IMAGE_WIDTH}x{IMAGE_HEIGHT})")
                    
        except Exception as e:
            print(f"  解析錯誤: {e}")

    print(f"\n完成！模型: {best_model_path.as_posix()}")