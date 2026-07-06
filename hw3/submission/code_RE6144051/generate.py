"""
使用訓練好的模型生成圖片
"""

import os
import argparse
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np

from ddpm import UNet, DDPM


def generate_images(ddpm, num_images, output_dir, batch_size=100, device='cuda'):
    """
    生成指定數量的圖片
    
    Args:
        ddpm: DDPM 模型
        num_images: 要生成的圖片數量
        output_dir: 輸出資料夾
        batch_size: 每批生成的數量
        device: 運算裝置
    """
    os.makedirs(output_dir, exist_ok=True)
    ddpm.model.eval()
    
    num_batches = (num_images + batch_size - 1) // batch_size
    image_idx = 1
    
    print(f"開始生成 {num_images} 張圖片...")
    
    for batch_idx in tqdm(range(num_batches), desc="生成圖片"):
        # 計算此批次要生成的數量
        current_batch_size = min(batch_size, num_images - batch_idx * batch_size)
        
        # 生成圖片
        with torch.no_grad():
            images = ddpm.sample(batch_size=current_batch_size, channels=3, image_size=28)
        
        # 反正規化到 [0, 255]
        images = (images + 1) / 2  # [-1, 1] -> [0, 1]
        images = torch.clamp(images, 0, 1)
        images = (images * 255).cpu().numpy().astype(np.uint8)
        
        # 儲存圖片
        for i in range(current_batch_size):
            img = images[i].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            img_pil = Image.fromarray(img)
            
            # 按照作業要求的命名格式: 00001.png ~ 10000.png
            filename = f"{image_idx:05d}.png"
            img_pil.save(os.path.join(output_dir, filename))
            
            image_idx += 1
    
    print(f"成功生成 {num_images} 張圖片,儲存於 {output_dir}")


def main(args):
    # 設定裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")
    
    # 載入檢查點
    print(f"載入模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # 重建模型
    model = UNet(
        in_channels=3,
        model_channels=checkpoint['args'].model_channels,
        out_channels=3,
        channel_mult=(1, 2, 4),  # 改為 3 層
        num_res_blocks=2,
        attention_resolutions=(1,)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 初始化 DDPM
    ddpm = DDPM(
        model=model,
        timesteps=checkpoint['args'].timesteps,
        beta_start=checkpoint['args'].beta_start,
        beta_end=checkpoint['args'].beta_end,
        device=device
    )
    
    # 生成圖片
    generate_images(
        ddpm=ddpm,
        num_images=args.num_images,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=device
    )
    
    print("\n完成!")
    print(f"請將 {args.output_dir} 資料夾內的圖片壓縮成 img_<student-id>.zip")
    print(f"並確保解壓後圖片直接位於根目錄(無子資料夾)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成圖片')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型檢查點路徑')
    parser.add_argument('--num_images', type=int, default=10000,
                       help='要生成的圖片數量')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='輸出資料夾路徑')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='每批生成的數量')
    
    args = parser.parse_args()
    main(args)