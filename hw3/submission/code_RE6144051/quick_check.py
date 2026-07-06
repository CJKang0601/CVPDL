"""
快速測試資料集載入 (不需要 matplotlib)
"""

import os
import sys

def quick_check(data_path):
    """快速檢查資料集"""
    
    print("=" * 60)
    print("MNIST 資料集快速檢查")
    print("=" * 60)
    
    # 1. 檢查路徑
    print(f"\n1. 檢查路徑: {data_path}")
    if not os.path.exists(data_path):
        print(f"   ❌ 路徑不存在!")
        print(f"   請確認路徑是否正確")
        return False
    else:
        print(f"   ✅ 路徑存在")
    
    # 2. 統計圖片數量
    print(f"\n2. 統計圖片檔案")
    files = os.listdir(data_path)
    image_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    image_files = [f for f in files if f.endswith(image_extensions)]
    
    print(f"   總檔案數: {len(files)}")
    print(f"   圖片檔案數: {len(image_files)}")
    
    if len(image_files) == 0:
        print(f"   ❌ 沒有找到任何圖片!")
        return False
    elif len(image_files) < 1000:
        print(f"   ⚠️  警告: 圖片數量較少,可能影響訓練效果")
    else:
        print(f"   ✅ 圖片數量充足")
    
    # 3. 顯示範例檔名
    print(f"\n3. 範例檔名 (前 10 個):")
    for i, filename in enumerate(sorted(image_files)[:10], 1):
        print(f"   {i:2d}. {filename}")
    
    # 4. 嘗試載入一張圖片
    print(f"\n4. 測試圖片載入")
    try:
        from PIL import Image
        test_file = os.path.join(data_path, image_files[0])
        img = Image.open(test_file)
        print(f"   測試檔案: {image_files[0]}")
        print(f"   圖片大小: {img.size}")
        print(f"   圖片模式: {img.mode}")
        print(f"   ✅ 圖片可以正常載入")
        
        # 轉換為 RGB
        if img.mode != 'RGB':
            img_rgb = img.convert('RGB')
            print(f"   ℹ️  已轉換為 RGB 模式")
        
    except Exception as e:
        print(f"   ❌ 載入圖片時發生錯誤: {e}")
        return False
    
    # 5. 測試 DataLoader
    print(f"\n5. 測試 DataLoader")
    try:
        from dataset import get_mnist_dataloader
        dataloader = get_mnist_dataloader(data_path, batch_size=16, num_workers=0)
        
        print(f"   批次大小: 16")
        print(f"   總批次數: {len(dataloader)}")
        
        # 載入第一批
        images, labels = next(iter(dataloader))
        print(f"   第一批形狀: {images.shape}")
        print(f"   數值範圍: [{images.min():.3f}, {images.max():.3f}]")
        print(f"   ✅ DataLoader 正常運作")
        
    except Exception as e:
        print(f"   ❌ DataLoader 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 總結
    print("\n" + "=" * 60)
    print("🎉 資料集檢查完成!一切正常!")
    print("=" * 60)
    print(f"\n你可以開始訓練了:")
    print(f"  python train.py")
    print()
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='快速檢查資料集')
    parser.add_argument('--data_path', type=str,
                       default=r'D:\CJK\114-1\CV\hw3_RE6144051\hw3_mnist',
                       help='資料集路徑')
    
    args = parser.parse_args()
    
    success = quick_check(args.data_path)
    sys.exit(0 if success else 1)
