"""
視覺化擴散過程 
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from ddpm import UNet, DDPM


def visualize_diffusion_process(ddpm, output_path='diffusion_process.png', device='cuda'):
    """
    視覺化擴散過程
    
    生成 8 個樣本,每個樣本記錄 8 個時間點 (包含最終結果)
    排列成 8x8 的網格
    """
    print("生成擴散過程視覺化...")
    
    # 生成帶有中間過程的樣本
    trajectory = ddpm.sample_with_trajectory(
        batch_size=8,
        channels=3,
        image_size=28,
        num_snapshots=8
    )
    
    # 建立 8x8 的圖表
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    fig.suptitle('Diffusion Process: 8 Generated Samples', fontsize=16, y=0.995)
    
    # 填充圖表
    for sample_idx in range(8):
        for time_idx in range(8):
            ax = axes[time_idx, sample_idx]
            
            # 取得該時間點的圖片
            img = trajectory[time_idx][sample_idx]
            
            # 反正規化到 [0, 1]
            img = (img + 1) / 2
            img = torch.clamp(img, 0, 1)
            
            # 轉換為可顯示格式
            img = img.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
            
            # 顯示圖片
            ax.imshow(img)
            ax.axis('off')
            
            # 在第一列加上標題
            if sample_idx == 0:
                if time_idx == 0:
                    ax.set_ylabel('t=T (noise)', fontsize=10, rotation=0, ha='right')
                elif time_idx == 7:
                    ax.set_ylabel('t=0 (clean)', fontsize=10, rotation=0, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"擴散過程視覺化已儲存至: {output_path}")
    
    plt.close()


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
        channel_mult=(1, 2, 4),  
        num_res_blocks=2,
        attention_resolutions=(1,)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    ddpm = DDPM(
        model=model,
        timesteps=checkpoint['args'].timesteps,
        beta_start=checkpoint['args'].beta_start,
        beta_end=checkpoint['args'].beta_end,
        device=device
    )
    
    # 視覺化擴散過程
    visualize_diffusion_process(ddpm, output_path=args.output_path, device=device)
    
    print("\n完成!")
    print(f"請將此圖片放入報告中")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='視覺化擴散過程')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型檢查點路徑')
    parser.add_argument('--output_path', type=str, default='diffusion_process.png',
                       help='輸出圖片路徑')
    
    args = parser.parse_args()
    main(args)