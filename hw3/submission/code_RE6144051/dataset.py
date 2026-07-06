"""
自訂 Dataset 用於載入 MNIST 圖片
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MNISTDataset(Dataset):
    """
    MNIST 資料集載入器
    假設所有圖片直接放在同一個資料夾中,不分子資料夾
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 包含所有圖片的資料夾路徑
            transform (callable, optional): 對圖片進行的轉換
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 取得所有圖片檔案
        self.image_files = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
        
        for filename in os.listdir(root_dir):
            if filename.endswith(valid_extensions):
                self.image_files.append(filename)
        
        self.image_files.sort()  # 排序確保順序一致
        
        print(f"找到 {len(self.image_files)} 張圖片在 {root_dir}")
        
        if len(self.image_files) == 0:
            raise ValueError(f"在 {root_dir} 中沒有找到任何圖片!")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.image_files[idx])
        
        # 載入圖片
        image = Image.open(img_name)
        
        # 確保是 RGB 格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 套用轉換
        if self.transform:
            image = self.transform(image)
        
        # 返回圖片和假標籤 (因為不需要真實標籤用於無條件生成)
        return image, 0


def get_mnist_dataloader(data_path, batch_size=128, num_workers=4, image_size=28):
    """
    建立 MNIST DataLoader
    
    Args:
        data_path: 圖片資料夾路徑
        batch_size: 批次大小
        num_workers: 資料載入的工作執行緒數
        image_size: 圖片大小
    
    Returns:
        DataLoader
    """
    # 定義轉換
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 正規化到 [-1, 1]
    ])
    
    # 建立資料集
    dataset = MNISTDataset(root_dir=data_path, transform=transform)
    
    # 建立 DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 丟棄最後不完整的批次
    )
    
    return dataloader
