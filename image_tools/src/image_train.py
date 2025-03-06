# 导入所需的库
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datetime import datetime

# PyTorch相关导入
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# matplotlib导入
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class RegressionDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # 读取标签文件
        labels_file = os.path.join(data_dir, f'{os.path.basename(data_dir)}_labels.csv')
        self.labels_df = pd.read_csv(labels_file)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # 获取图片路径和标签
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.data_dir, row['image_name'])
        label = row['label']

        # 读取和转换图片
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

class ImageRegressor(nn.Module):
    """图像回归模型类"""
    def __init__(self, backbone='resnet18', pretrained=True):
        super(ImageRegressor, self).__init__()
        # 加载预训练模型
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)

        # 修改最后的全连接层用于回归
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.backbone(x).squeeze()

class Trainer:
    """训练器类"""
    def __init__(self, model, train_loader, val_loader, device,
                 learning_rate=0.001, save_dir='checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 定义优化器和损失函数
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # 记录训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf')
        }

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc='Training')

        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(self.train_loader)

    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_values = []

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validating'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                true_values.extend(labels.cpu().numpy())

        return (total_loss / len(self.val_loader),
                np.array(predictions),
                np.array(true_values))

    def train(self, num_epochs):
        """完整训练过程"""
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # 训练阶段
            train_loss = self.train_epoch()

            # 验证阶段
            val_loss, predictions, true_values = self.validate()

            # 更新学习率
            self.scheduler.step(val_loss)

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            # 打印结果
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")

            # 保存最佳模型
            if val_loss < self.history['best_val_loss']:
                self.history['best_val_loss'] = val_loss
                self.save_checkpoint(f'best_model.pth')

            # 每个epoch都保存一次
            self.save_checkpoint(f'epoch_{epoch+1}.pth')

            # 绘制并保存当前预测结果图
            self.plot_predictions(predictions, true_values, epoch+1)

        # 保存训练历史
        self.save_history()

    def save_checkpoint(self, filename):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))

    def load_checkpoint(self, filename):
        """加载检查点"""
        checkpoint = torch.load(os.path.join(self.save_dir, filename))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']

    def plot_predictions(self, predictions, true_values, epoch):
        """绘制预测结果图"""
        plt.figure(figsize=(10, 6))
        plt.scatter(true_values, predictions, alpha=0.5)
        plt.plot([true_values.min(), true_values.max()],
                 [true_values.min(), true_values.max()],
                 'r--', lw=2)
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(f'预测值 vs 真实值 - Epoch {epoch}')
        plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch}.png'))
        plt.close()

    def save_history(self):
        """保存训练历史"""
        history_file = os.path.join(self.save_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(self.history, f)

class Predictor:
    """预测器类"""
    def __init__(self, model, device, transform=None):
        self.model = model.to(device)
        self.device = device
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        """预测单张图片"""
        self.model.eval()

        # 加载和预处理图像
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            prediction = self.model(image).item()

        return prediction

def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 设置数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 设置路径
    dataset_path = "/image_tools/first_batch_of_samples/dataset"
    save_dir = "../image_train/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据集
    train_dataset = RegressionDataset(
        os.path.join(dataset_path, 'train'),
        transform=transform
    )
    val_dataset = RegressionDataset(
        os.path.join(dataset_path, 'val'),
        transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    # 创建模型
    model = ImageRegressor(backbone='resnet34', pretrained=True)

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001,
        save_dir=save_dir
    )

    # 训练模型
    trainer.train(num_epochs=50)

    print("训练完成！")

if __name__ == "__main__":
    main()