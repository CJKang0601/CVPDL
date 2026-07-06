"""
測試模型輸入輸出尺寸是否匹配
"""

import torch
import sys
sys.path.append('.')

from ddpm import UNet

def test_model():
    """測試模型"""
    print("=" * 60)
    print("測試 U-Net 模型尺寸")
    print("=" * 60)
    
    # 建立模型
    model = UNet(
        in_channels=3,
        model_channels=64,
        out_channels=3,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        attention_resolutions=(1,)
    )
    
    print(f"\n模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 測試不同的批次大小和圖片尺寸
    test_cases = [
        (1, 28),   # 單張圖片
        (16, 28),  # 小批次
        (128, 28), # 訓練批次
    ]
    
    print("\n測試不同的輸入尺寸:")
    print("-" * 60)
    
    all_passed = True
    
    for batch_size, img_size in test_cases:
        try:
            # 建立輸入
            x = torch.randn(batch_size, 3, img_size, img_size)
            t = torch.randint(0, 1000, (batch_size,))
            
            # 前向傳播
            with torch.no_grad():
                output = model(x, t)
            
            # 檢查輸出尺寸
            if output.shape == x.shape:
                print(f"✅ Batch={batch_size:3d}, Size={img_size:2d}x{img_size:2d} → 輸出: {tuple(output.shape)} (正確)")
            else:
                print(f"❌ Batch={batch_size:3d}, Size={img_size:2d}x{img_size:2d} → 輸出: {tuple(output.shape)} (錯誤,應為 {tuple(x.shape)})")
                all_passed = False
                
        except Exception as e:
            print(f"❌ Batch={batch_size:3d}, Size={img_size:2d}x{img_size:2d} → 錯誤: {e}")
            all_passed = False
    
    print("-" * 60)
    
    if all_passed:
        print("\n" + "=" * 60)
        print("🎉 所有測試通過!模型可以正常使用!")
        print("=" * 60)
        print("\n你可以開始訓練了:")
        print("  python train.py")
        return True
    else:
        print("\n" + "=" * 60)
        print("❌ 測試失敗!請檢查模型架構")
        print("=" * 60)
        return False


if __name__ == '__main__':
    success = test_model()
    sys.exit(0 if success else 1)