import os
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image

# 定义图像转换（这里只转换为Tensor）
transform = transforms.ToTensor()

# 加载自定义数据集
image_dir = 'your_dataset_path'
dataset = CustomDataset(image_dir, transform=transform)
loader = DataLoader(dataset, batch_size=14, shuffle=False, num_workers=4)

# 初始化变量
mean = 0.
std = 0.
nb_samples = 0.

# 遍历数据集，累积求和
for data in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

# 计算均值和方差
mean /= nb_samples
std /= nb_samples

print(f'均值: {mean}')
print(f'方差: {std}')
