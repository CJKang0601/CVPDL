"""
檢查資料集是否正確載入
"""

import os
import argparse
from dataset import get_mnist_dataloader
import matplotlib.pyplot as plt
import numpy as np


def check_dataset(data_path):
    """檢查資料集"""
    
    print(f"檢查資料集路徑: {data_path}")
    print(f"路徑存在: {os.path.exists(data_path)}")
    
    if not os.path.exists(data_path):
        print(f"❌ 錯誤: 路徑 {data_path} 不存在!")
        return
    
    # 列出前 10 個檔案
    files = os.listdir(data_path)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\n找到 {len(image_files)} 張圖片")
    print(f"前 10 個檔案:")
    for i, f in enumerate(image_files[:10]):
        print(f"  {i+1}. {f}")
    
    # 載入 DataLoader
    print("\n建立 DataLoader...")
    try:
        dataloader = get_mnist_dataloader(data_path, batch_size=16, num_workers=0)
        print(f"✅ DataLoader 建立成功!")
        print(f"批次數量: {len(dataloader)}")
        
        # 載入一個批次
        print("\n載入第一個批次...")
        images, labels = next(iter(dataloader))
        print(f"批次形狀: {images.shape}")
        print(f"數值範圍: [{images.min():.3f}, {images.max():.3f}]")
        print(f"數值類型: {images.dtype}")
        
        # 視覺化前 16 張圖片
        print("\n生成視覺化圖片...")
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        fig.suptitle('Dataset Sample Images', fontsize=16)
        
        for i, ax in enumerate(axes.flat):
            # 反正規化到 [0, 1]
            img = (images[i] + 1) / 2
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('dataset_check.png', dpi=100, bbox_inches='tight')
        print(f"✅ 視覺化圖片已儲存至 dataset_check.png")
        
        print("\n🎉 資料集檢查完成!一切正常!")
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='檢查資料集')
    parser.add_argument('--data_path', type=str, 
                       default=r'D:\CJK\114-1\CV\hw3_RE6144051\hw3_mnist',
                       help='資料集路徑')
    
    args = parser.parse_args()
    check_dataset(args.data_path)
