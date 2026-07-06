"""
DDPM (Denoising Diffusion Probabilistic Models) 實作
參考: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    """時間步編碼"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """殘差區塊"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.activation = nn.SiLU()
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb):
        residual = self.residual_conv(x)
        
        # 第一層卷積
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        # 加入時間嵌入
        time_emb = self.activation(self.time_mlp(time_emb))
        x = x + time_emb[:, :, None, None]
        
        # 第二層卷積
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        
        return x + residual


class AttentionBlock(nn.Module):
    """自注意力機制"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # 重塑為 (B, C, H*W)
        q = q.view(B, C, -1)
        k = k.view(B, C, -1)
        v = v.view(B, C, -1)
        
        # 計算注意力分數
        attn = torch.bmm(q.transpose(1, 2), k) / np.sqrt(C)
        attn = F.softmax(attn, dim=-1)
        
        # 套用注意力
        out = torch.bmm(v, attn.transpose(1, 2))
        out = out.view(B, C, H, W)
        out = self.proj(out)
        
        return out + residual


class UNet(nn.Module):
    """U-Net 架構用於雜訊預測"""
    def __init__(self, in_channels=3, model_channels=64, out_channels=3, 
                 channel_mult=(1, 2, 4), num_res_blocks=2, attention_resolutions=(1,)):
        super().__init__()
        
        time_emb_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 編碼器
        self.encoder = nn.ModuleList()
        self.encoder_channels = []
        ch = model_channels
        input_ch = in_channels
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                self.encoder.append(ResidualBlock(input_ch, ch * mult, time_emb_dim))
                input_ch = ch * mult
                self.encoder_channels.append(input_ch)
                
                # 在特定解析度加入注意力
                if level in attention_resolutions:
                    self.encoder.append(AttentionBlock(input_ch))
                    self.encoder_channels.append(input_ch)
            
            # 下採樣 (除了最後一層)
            if level != len(channel_mult) - 1:
                self.encoder.append(nn.Conv2d(input_ch, input_ch, 3, stride=2, padding=1))
                self.encoder_channels.append(input_ch)
        
        # 瓶頸層
        self.middle = nn.ModuleList([
            ResidualBlock(input_ch, input_ch, time_emb_dim),
            AttentionBlock(input_ch),
            ResidualBlock(input_ch, input_ch, time_emb_dim),
        ])
        
        # 解碼器
        self.decoder = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            for _ in range(num_res_blocks + 1):
                self.decoder.append(ResidualBlock(input_ch, ch * mult, time_emb_dim))
                input_ch = ch * mult
                
                if level in attention_resolutions:
                    self.decoder.append(AttentionBlock(input_ch))
            
            # 上採樣 (除了第一層)
            if level != 0:
                self.decoder.append(nn.Upsample(scale_factor=2, mode='nearest'))
                self.decoder.append(nn.Conv2d(input_ch, input_ch, 3, padding=1))
        
        # 輸出層
        self.out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps):
        # 時間嵌入
        t_emb = self.time_mlp(timesteps)
        
        # 編碼
        h = x
        for module in self.encoder:
            if isinstance(module, ResidualBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
        
        # 瓶頸
        for module in self.middle:
            if isinstance(module, ResidualBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
        
        # 解碼
        for module in self.decoder:
            if isinstance(module, ResidualBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
        
        # 輸出
        return self.out(h)


class DDPM:
    """擴散模型訓練與採樣"""
    def __init__(self, model, timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # 定義 beta 排程 (線性)
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 計算用於採樣的係數
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 後驗分布方差
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """前向擴散過程: 加入雜訊"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, noise=None):
        """訓練損失: 預測雜訊"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t)
        
        # 簡單 MSE 損失
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, x, t):
        """反向去噪單步採樣"""
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t][:, None, None, None]
        
        # 預測雜訊並去除
        predicted_noise = self.model(x, t)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][:, None, None, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, batch_size=16, channels=3, image_size=28):
        """完整採樣過程"""
        self.model.eval()
        
        # 從純雜訊開始
        x = torch.randn(batch_size, channels, image_size, image_size).to(self.device)
        
        # 逐步去噪
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, dtype=torch.long, device=self.device)
            x = self.p_sample(x, t)
        
        return x
    
    @torch.no_grad()
    def sample_with_trajectory(self, batch_size=8, channels=3, image_size=28, num_snapshots=8):
        """採樣並記錄中間過程"""
        self.model.eval()
        
        x = torch.randn(batch_size, channels, image_size, image_size).to(self.device)
        trajectory = [x.cpu()]
        
        # 計算每隔幾步記錄一次
        interval = self.timesteps // (num_snapshots - 1)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, dtype=torch.long, device=self.device)
            x = self.p_sample(x, t)
            
            if (self.timesteps - i - 1) % interval == 0 or i == 0:
                trajectory.append(x.cpu())
        
        return trajectory