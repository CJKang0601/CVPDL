"""
訓練 DDPM 模型
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from ddpm import UNet, DDPM
from dataset import get_mnist_dataloader


def train_epoch(ddpm, dataloader, optimizer, device, epoch, writer):
    """訓練一個 epoch"""
    ddpm.model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device)
        batch_size = images.shape[0]
        
        # 隨機選擇時間步
        t = torch.randint(0, ddpm.timesteps, (batch_size,), device=device).long()
        
        # 計算損失
        loss = ddpm.p_losses(images, t)
        
        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # 更新進度條
        pbar.set_postfix({'loss': loss.item()})
        
        # 記錄到 tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('Loss/train', loss.item(), global_step)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def sample_images(ddpm, num_images=16, save_path=None):
    """生成樣本圖片"""
    ddpm.model.eval()
    images = ddpm.sample(batch_size=num_images)
    
    # 反正規化到 [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    if save_path:
        from torchvision.utils import save_image
        save_image(images, save_path, nrow=4)
    
    return images


def main(args):
    # 設定裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")
    
    # 建立輸出資料夾
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'samples'), exist_ok=True)
    
    # 載入資料
    print("載入資料...")
    dataloader = get_mnist_dataloader(args.data_path, batch_size=args.batch_size)
    
    # 建立模型
    print("建立模型...")
    model = UNet(
        in_channels=3,
        model_channels=args.model_channels,
        out_channels=3,
        channel_mult=(1, 2, 4),  # 改為 3 層,適合 28×28
        num_res_blocks=2,
        attention_resolutions=(1,)  # 在第二層加入注意力
    ).to(device)
    
    # 初始化 DDPM
    ddpm = DDPM(
        model=model,
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device
    )
    
    # 優化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 學習率調度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 載入 checkpoint (如果有)
    start_epoch = 1
    best_loss = float('inf')
    if args.resume:
        print(f"載入 checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
        print(f"從 epoch {start_epoch} 繼續訓練")
    
    # Tensorboard
    writer = SummaryWriter(args.log_dir)
    
    # 訓練
    print("開始訓練...")
    
    for epoch in range(start_epoch, args.epochs + 1):
        # 訓練
        avg_loss = train_epoch(ddpm, dataloader, optimizer, device, epoch, writer)
        scheduler.step()
        
        print(f"Epoch {epoch}/{args.epochs} - Avg Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        # 每隔一定 epoch 生成樣本
        if epoch % args.sample_interval == 0:
            sample_path = os.path.join(args.save_dir, 'samples', f'epoch_{epoch}.png')
            sample_images(ddpm, num_images=16, save_path=sample_path)
            print(f"樣本已儲存至 {sample_path}")
        
        # 儲存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': args
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"最佳模型已儲存 (Loss: {best_loss:.4f})")
        
        # 定期儲存檢查點
        if epoch % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': args
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    writer.close()
    print("訓練完成!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='訓練 DDPM 模型')
    
    # 資料參數
    parser.add_argument('--data_path', type=str, default=r'D:\CJK\114-1\CV\hw3_RE6144051\hw3_mnist',
                       help='資料集路徑')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    
    # 模型參數
    parser.add_argument('--model_channels', type=int, default=64, help='基礎通道數')
    parser.add_argument('--timesteps', type=int, default=1000, help='擴散步數')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Beta 起始值')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Beta 結束值')
    
    # 訓練參數
    parser.add_argument('--epochs', type=int, default=100, help='訓練輪數')
    parser.add_argument('--lr', type=float, default=2e-4, help='學習率')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='權重衰減')
    parser.add_argument('--resume', type=str, default=None, help='從 checkpoint 繼續訓練')
    
    # 儲存參數
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型儲存路徑')
    parser.add_argument('--log_dir', type=str, default='logs', help='日誌路徑')
    parser.add_argument('--save_interval', type=int, default=10, help='儲存檢查點間隔')
    parser.add_argument('--sample_interval', type=int, default=5, help='生成樣本間隔')
    
    args = parser.parse_args()
    main(args)