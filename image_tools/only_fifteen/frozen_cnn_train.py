# 导入所需的库
import os
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
from datetime import datetime
from skimage.feature import local_binary_pattern
from skimage import filters, exposure, morphology
import cv2

# PyTorch相关导入
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as F
from sklearn.model_selection import train_test_split

# matplotlib导入
import matplotlib.pyplot as plt
import seaborn as sns

class FixedRotation(object):
    """固定角度旋转: 只旋转0°、90°、180°或270°"""
    def __init__(self, p=0.75):
        """
        参数:
            p: 应用旋转的概率 (0°角度的概率为1-p)
        """
        self.p = p
        self.angles = [90, 180, 270]  # 可选旋转角度

    def __call__(self, img):
        if torch.rand(1) < self.p:
            # 随机选择一个角度
            angle = self.angles[torch.randint(0, len(self.angles), (1,)).item()]
            return img.rotate(angle)
        return img  # 不旋转(0°)

class AdaptiveEdgeEnhancer(object):
    """自适应边缘增强器"""
    def __init__(self, alpha=1.5, beta=0.5, p=0.7):
        """
        参数:
            alpha: 边缘增强强度
            beta: 原始图像保留比例
            p: 应用此变换的概率
        """
        self.alpha = alpha
        self.beta = beta
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            # 转为numpy数组处理
            img_np = np.array(img)

            # 转为灰度进行边缘检测
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np

            # 使用自适应阈值方法 - 高斯权重, 块大小11, 常数2
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # 用Canny进一步检测边缘
            edges = cv2.Canny(gray, 50, 150)

            # 应用形态学操作来连接相近的边缘
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            edges = cv2.erode(edges, kernel, iterations=1)

            # 结合两种边缘效果
            combined_edges = cv2.bitwise_or(binary, edges)

            # 如果是彩色图像,使用边缘增强
            if len(img_np.shape) == 3:
                # 创建边缘蒙版
                edge_mask = combined_edges / 255.0
                edge_mask_3d = np.stack([edge_mask] * 3, axis=2)

                # 锐化原图
                sharpened = img_np.astype(float)
                blurred = cv2.GaussianBlur(img_np, (0, 0), 3)
                sharpened = cv2.addWeighted(img_np, 1.5, blurred, -0.5, 0)

                # 混合原图和边缘信息
                result = img_np * self.beta + sharpened * (1 - self.beta)
                # 在边缘位置额外增强
                result = result * (1 - edge_mask_3d * self.alpha) + sharpened * (edge_mask_3d * self.alpha)
                result = np.clip(result, 0, 255).astype(np.uint8)

                return Image.fromarray(result)

            else:
                # 灰度图处理
                sharpened = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0, 0), 3), -0.5, 0)
                result = gray * self.beta + sharpened * (1 - self.beta)
                result = result * (1 - edge_mask * self.alpha) + sharpened * (edge_mask * self.alpha)
                result = np.clip(result, 0, 255).astype(np.uint8)

                return Image.fromarray(result)

        return img

class ContrastTextureEnhancer(object):
    """对比度感知纹理增强器"""
    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8), p=0.7):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            # 转为numpy数组
            img_np = np.array(img)

            # 转为LAB颜色空间
            if len(img_np.shape) == 3:  # 彩色图像
                lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)

                # 对L通道应用CLAHE (对比度受限的自适应直方图均衡化)
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
                cl = clahe.apply(l)

                # 合并回LAB再转回RGB
                enhanced_lab = cv2.merge((cl, a, b))
                enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

                return Image.fromarray(enhanced_rgb)
            else:  # 灰度图
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
                enhanced = clahe.apply(img_np)
                return Image.fromarray(enhanced)

        return img

# 数据集分割器 - 新增加的类
class DatasetSplitter:
    """将原始数据集分割为训练集、验证集和测试集"""
    def __init__(self, source_dir, labels_file, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        self.source_dir = source_dir
        self.labels_file = labels_file
        self.target_dir = target_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        # 确保比例和为1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"

    def split_dataset(self):
        """执行数据集分割"""
        print(f"Splitting dataset from {self.source_dir}")

        # 创建目标目录
        os.makedirs(os.path.join(self.target_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.target_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.target_dir, 'test'), exist_ok=True)

        # 读取标签文件
        df = pd.read_csv(self.labels_file)
        print(f"Found {len(df)} entries in labels file")

        # 验证所有图片是否存在
        valid_entries = []
        for idx, row in df.iterrows():
            img_path = os.path.join(self.source_dir, row['image_name'])
            if os.path.exists(img_path):
                valid_entries.append(row)
            else:
                print(f"Warning: Image not found: {img_path}")

        # 创建只包含有效条目的数据帧
        valid_df = pd.DataFrame(valid_entries)
        print(f"Found {len(valid_df)} valid images")

        if len(valid_df) == 0:
            raise ValueError("No valid images found. Please check the source directory and labels file.")

        # 分割为训练集和临时集
        train_df, temp_df = train_test_split(
            valid_df,
            train_size=self.train_ratio,
            random_state=self.random_state
        )

        # 将临时集再分为验证集和测试集
        val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size,
            random_state=self.random_state
        )

        print(f"Split dataset: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test")

        # 保存子集
        self._save_subset(train_df, 'train')
        self._save_subset(val_df, 'val')
        self._save_subset(test_df, 'test')

    def _save_subset(self, df, subset_name):
        """保存数据子集"""
        target_dir = os.path.join(self.target_dir, subset_name)

        # 复制图片并保存标签
        for idx, row in tqdm(df.iterrows(), desc=f"Copying {subset_name} set", total=len(df)):
            # 构建源路径和目标路径
            src_path = os.path.join(self.source_dir, row['image_name'])
            dst_path = os.path.join(target_dir, row['image_name'])

            # 复制图片
            try:
                image = Image.open(src_path)
                image.save(dst_path)
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
                continue

        # 保存标签文件
        df.to_csv(os.path.join(target_dir, f'{subset_name}_labels.csv'), index=False)
        print(f"Saved {len(df)} images to {subset_name} set")

class DatasetAugmenter:
    """数据集增强器 - 修改后版本"""
    def __init__(self, augmentation_factor=5):
        self.augmentation_factor = augmentation_factor
        self.transform = transforms.Compose([
            # 1) 固定角度旋转 (0°, 90°, 180°, 270°)
            FixedRotation(p=0.75),

            # 1) 自适应边缘增强
            AdaptiveEdgeEnhancer(alpha=1.7, beta=0.4, p=0.8),

            # 2) 对比度感知纹理增强
            ContrastTextureEnhancer(clip_limit=3.0, tile_grid_size=(8, 8), p=0.7),

            # 调整到指定大小
            transforms.Resize((224, 224)),
        ])

    def augment_dataset(self, source_dir, target_dir):
        """增强数据集并保存"""
        os.makedirs(target_dir, exist_ok=True)

        # 读取原始标签文件 - 适应现有的命名约定
        subset_name = os.path.basename(source_dir)  # 'train', 'val', 或 'test'
        labels_file = os.path.join(source_dir, f'{subset_name}_labels.csv')

        if not os.path.exists(labels_file):
            raise ValueError(f"Labels file not found: {labels_file}")

        print(f"Reading labels from {labels_file}")
        original_df = pd.read_csv(labels_file)
        print(f"Found {len(original_df)} entries in labels file")

        # 用于存储新的标签
        new_records = []

        # 对每张图片进行增强
        for idx, row in tqdm(original_df.iterrows(), desc=f"Augmenting {subset_name} dataset"):
            img_path = os.path.join(source_dir, row['image_name'])

            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue

            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error opening image {img_path}: {e}")
                continue

            # 保存原始图片
            original_name = f"orig_{row['image_name']}"
            image.save(os.path.join(target_dir, original_name))
            new_records.append({
                'image_name': original_name,
                'label': row['label']
            })

            # 生成增强图片
            for aug_idx in range(self.augmentation_factor):
                aug_image = self.transform(image)
                aug_name = f"aug{aug_idx}_{row['image_name']}"
                aug_image.save(os.path.join(target_dir, aug_name))
                new_records.append({
                    'image_name': aug_name,
                    'label': row['label']
                })

        # 保存新的标签文件
        new_df = pd.DataFrame(new_records)
        new_df.to_csv(os.path.join(target_dir, f'{subset_name}_labels.csv'),
                      index=False)

        print(f"Augmented {subset_name} dataset: {len(original_df)} original images -> {len(new_df)} total images")

class RegressionDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # 读取标签文件
        subset_name = os.path.basename(data_dir)  # 'train', 'val', 或 'test'
        labels_file = os.path.join(data_dir, f'{subset_name}_labels.csv')

        if not os.path.exists(labels_file):
            raise ValueError(f"Labels file not found: {labels_file}")

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

# 修改后的冻结CNN+FC层模型
class FrozenCNNRegressor(nn.Module):
    """使用冻结CNN特征提取器和可训练FC层的纹理回归模型"""
    def __init__(self, backbone='densenet121', pretrained=True, initial_value=15.0):
        super(FrozenCNNRegressor, self).__init__()

        # 加载预训练骨干网络
        if backbone == 'densenet121':
            base_model = models.densenet121(pretrained=pretrained)
            self.features = base_model.features
            feature_dim = base_model.classifier.in_features  # 1024
        elif backbone == 'densenet169':
            base_model = models.densenet169(pretrained=pretrained)
            self.features = base_model.features
            feature_dim = base_model.classifier.in_features  # 1664
        elif backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            # 移除全局平均池化层和全连接层
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            feature_dim = 512
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            feature_dim = 512
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            feature_dim = 2048
        elif backbone == 'mobilenet_v2':
            base_model = models.mobilenet_v2(pretrained=pretrained)
            self.features = base_model.features
            feature_dim = 1280
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")

        # 冻结特征提取器
        for param in self.features.parameters():
            param.requires_grad = False

        # 全局平均池化层
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 带有L2正则化效果的回归头 (类似Ridge回归)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),  # 批量归一化有助于稳定训练
            nn.ReLU(),
            nn.Dropout(0.3),      # 丢弃法帮助防止过拟合
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        # 初始化最后一层的偏置为指定值
        final_layer = self.regressor[-1]
        nn.init.constant_(final_layer.bias, initial_value)

    def forward(self, x):
        # 提取特征 (冻结阶段)
        with torch.no_grad():
            features = self.features(x)

        # 全局平均池化
        pooled = self.global_pool(features)

        # 回归预测 (可训练部分)
        output = self.regressor(pooled).squeeze()

        return output

    def unfreeze_last_layers(self, num_layers=2):
        """解冻CNN特征提取器的最后几层进行微调"""
        # 对于不同的骨干网络，需要具体实现解冻机制
        if isinstance(self.features, nn.Sequential):
            # 这适用于ResNet等顺序模型
            for i, module in enumerate(list(self.features.children())[-num_layers:]):
                for param in module.parameters():
                    param.requires_grad = True
            print(f"已解冻最后{num_layers}层顺序模块")
        elif hasattr(self.features, 'denseblock4'):
            # 这适用于DenseNet
            for param in self.features.denseblock4.parameters():
                param.requires_grad = True
            for param in self.features.norm5.parameters():
                param.requires_grad = True
            print(f"已解冻DenseNet的最后一个密集块和norm层")
        else:
            print("无法识别的骨干网络结构，未解冻任何层")

class MixedRegressionLoss(nn.Module):
    """混合回归损失函数"""
    def __init__(self, mse_weight=0.5, l1_weight=0.3, huber_weight=0.2):
        super(MixedRegressionLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.huber = nn.SmoothL1Loss()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.huber_weight = huber_weight

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        huber_loss = self.huber(pred, target)

        return (self.mse_weight * mse_loss +
                self.l1_weight * l1_loss +
                self.huber_weight * huber_loss)

class Trainer:
    """训练器类"""
    def __init__(self, model, train_loader, val_loader, test_loader=None, device='cuda',
                 learning_rate=0.001, save_dir='checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 定义优化器和混合损失函数
        self.criterion = MixedRegressionLoss(mse_weight=0.5, l1_weight=0.3, huber_weight=0.2)

        # 获取只有require_grad=True的参数进行优化
        # 这样只会更新解冻部分的参数，大大提高训练效率
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

        print(f"可训练参数数量: {len(params_to_update)}")

        # 使用AdamW优化器，带有L2正则化
        self.optimizer = optim.AdamW(params_to_update, lr=learning_rate, weight_decay=1e-4)

        # 学习率调度 - 使用余弦退火
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,        # 重启周期
            T_mult=2,      # 每次重启后周期增加系数
            eta_min=1e-6   # 最小学习率
        )

        # 记录训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }

        # 早停计数器
        self.patience = 25  # 25个epoch没有提升则停止
        self.patience_counter = 0
        self.early_stop = False

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

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(self.train_loader)

    def validate(self, data_loader, desc='Validating'):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_values = []

        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc=desc):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                true_values.extend(labels.cpu().numpy())

        return (total_loss / len(data_loader),
                np.array(predictions),
                np.array(true_values))

    def evaluate_all_datasets(self):
        """对所有数据集进行评估并生成综合报告"""
        print("\n评估所有数据集...")
        # 加载最佳模型
        self.load_checkpoint('best_model.pth')

        # 创建结果存储字典
        results = {}

        # 评估训练集
        train_loss, train_preds, train_true = self.validate(self.train_loader, desc='Evaluating Training Set')
        results['train'] = {
            'loss': train_loss,
            'predictions': train_preds,
            'true_values': train_true,
            'r2': self.calculate_r2(train_preds, train_true),
            'rmse': np.sqrt(np.mean((train_preds - train_true) ** 2)),
            'mae': np.mean(np.abs(train_preds - train_true))
        }

        # 评估验证集
        val_loss, val_preds, val_true = self.validate(self.val_loader, desc='Evaluating Validation Set')
        results['val'] = {
            'loss': val_loss,
            'predictions': val_preds,
            'true_values': val_true,
            'r2': self.calculate_r2(val_preds, val_true),
            'rmse': np.sqrt(np.mean((val_preds - val_true) ** 2)),
            'mae': np.mean(np.abs(val_preds - val_true))
        }

        # 评估测试集
        test_loss, test_preds, test_true = self.validate(self.test_loader, desc='Evaluating Test Set')
        results['test'] = {
            'loss': test_loss,
            'predictions': test_preds,
            'true_values': test_true,
            'r2': self.calculate_r2(test_preds, test_true),
            'rmse': np.sqrt(np.mean((test_preds - test_true) ** 2)),
            'mae': np.mean(np.abs(test_preds - test_true))
        }

        # 保存结果
        self.plot_all_datasets_comparison(results)
        self.create_metrics_table(results)

        return results

    def plot_all_datasets_comparison(self, results):
        """绘制所有数据集的预测与真实值对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        datasets = ['train', 'val', 'test']
        titles = ['训练集', '验证集', '测试集']
        colors = ['blue', 'green', 'red']

        for i, (dataset, title, color) in enumerate(zip(datasets, titles, colors)):
            data = results[dataset]
            axes[i].scatter(data['true_values'], data['predictions'], alpha=0.6, s=30, c=color)

            # 添加完美预测线
            min_val = min(np.min(data['true_values']), np.min(data['predictions']))
            max_val = max(np.max(data['true_values']), np.max(data['predictions']))
            axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)

            # 添加回归线
            z = np.polyfit(data['true_values'], data['predictions'], 1)
            p = np.poly1d(z)
            axes[i].plot(data['true_values'], p(data['true_values']), 'g-', lw=1.5, alpha=0.7)

            # 设置标题和标签
            axes[i].set_title(f'{title}\nR² = {data["r2"]:.4f}, RMSE = {data["rmse"]:.4f}')
            axes[i].set_xlabel('真实值')
            axes[i].set_ylabel('预测值')
            axes[i].grid(True, alpha=0.3)

            # 设置轴范围
            margin = (max_val - min_val) * 0.05  # 5% margin
            axes[i].set_xlim(min_val - margin, max_val + margin)
            axes[i].set_ylim(min_val - margin, max_val + margin)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'all_datasets_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_metrics_table(self, results):
        """创建包含所有数据集性能指标的表格"""
        # 准备表格数据
        metrics_data = {
            '数据集': ['训练集', '验证集', '测试集'],
            '样本数量': [
                len(results['train']['true_values']),
                len(results['val']['true_values']),
                len(results['test']['true_values'])
            ],
            '损失值': [
                results['train']['loss'],
                results['val']['loss'],
                results['test']['loss']
            ],
            'R²': [
                results['train']['r2'],
                results['val']['r2'],
                results['test']['r2']
            ],
            'RMSE': [
                results['train']['rmse'],
                results['val']['rmse'],
                results['test']['rmse']
            ],
            'MAE': [
                results['train']['mae'],
                results['val']['mae'],
                results['test']['mae']
            ]
        }

        # 创建表格可视化
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')

        # 创建表格
        table = ax.table(
            cellText=[
                [f"{metrics_data['数据集'][i]}",
                 f"{metrics_data['样本数量'][i]}",
                 f"{metrics_data['损失值'][i]:.4f}",
                 f"{metrics_data['R²'][i]:.4f}",
                 f"{metrics_data['RMSE'][i]:.4f}",
                 f"{metrics_data['MAE'][i]:.4f}"]
                for i in range(3)
            ],
            colLabels=['数据集', '样本数量', '损失值', 'R²', 'RMSE', 'MAE'],
            loc='center',
            cellLoc='center'
        )

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)

        # 设置表头样式
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # 表头行
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4472C4')
            elif j == 0:  # 第一列
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#D9E1F2')
            elif i % 2 == 1:  # 奇数行
                cell.set_facecolor('#E9EDF4')

        plt.title('模型在各数据集上的性能指标对比', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'performance_metrics_table.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 保存指标为CSV文件
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(os.path.join(self.save_dir, 'performance_metrics.csv'), index=False)

        # 保存为HTML格式的表格
        html_table = metrics_df.to_html(index=False)
        with open(os.path.join(self.save_dir, 'performance_metrics.html'), 'w') as f:
            f.write("<html><head><style>")
            f.write("table {border-collapse: collapse; width: 100%; margin: 20px 0;}")
            f.write("th {background-color: #4472C4; color: white; font-weight: bold; text-align: center; padding: 10px;}")
            f.write("td {padding: 8px; text-align: center; border: 1px solid #ddd;}")
            f.write("tr:nth-child(even) {background-color: #E9EDF4;}")
            f.write("tr:hover {background-color: #ddd;}")
            f.write("</style></head><body>")
            f.write("<h2>模型性能指标表</h2>")
            f.write(html_table)
            f.write("</body></html>")

    def train(self, num_epochs, eval_every=1, unfreeze_at_epoch=None):
        """完整训练过程"""
        for epoch in range(num_epochs):
            if self.early_stop:
                print("Early stopping triggered!")
                break

            # 在指定epoch解冻部分CNN层
            if unfreeze_at_epoch and epoch == unfreeze_at_epoch:
                print(f"Epoch {epoch+1}: 解冻CNN的最后几层进行微调")
                self.model.unfreeze_last_layers(num_layers=2)
                # 调整学习率为原来的1/10，避免破坏预训练特征
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")

            # 训练阶段
            train_loss = self.train_epoch()

            # 验证阶段 (根据eval_every参数决定频率)
            if (epoch + 1) % eval_every == 0:
                val_loss, predictions, true_values = self.validate(self.val_loader)

                # 更新学习率
                self.scheduler.step()

                # 记录历史
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['learning_rates'].append(current_lr)

                # 打印结果
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}")

                # 计算验证集的R^2和RMSE
                r2 = self.calculate_r2(predictions, true_values)
                rmse = np.sqrt(np.mean((predictions - true_values) ** 2))
                print(f"Val R²: {r2:.4f}, RMSE: {rmse:.4f}")

                # 保存最佳模型
                if val_loss < self.history['best_val_loss']:
                    self.history['best_val_loss'] = val_loss
                    self.history['best_epoch'] = epoch + 1
                    self.save_checkpoint(f'best_model.pth')
                    print(f"New best model saved with validation loss: {val_loss:.4f}")
                    # 重置早停计数器
                    self.patience_counter = 0
                else:
                    # 增加早停计数器
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"Early stopping after {self.patience} epochs without improvement")
                        self.early_stop = True

                # 每5个evaluation周期保存一次
                if (epoch + 1) % (eval_every * 5) == 0:
                    self.save_checkpoint(f'epoch_{epoch+1}.pth')

                # 绘制并保存当前预测结果图
                self.plot_predictions(predictions, true_values, epoch+1)

                # 每10个evaluation周期绘制学习曲线
                if (epoch + 1) % (eval_every * 10) == 0:
                    self.plot_learning_curves()

        # 保存训练历史
        self.save_history()
        print(f"Training completed! Best validation loss: {self.history['best_val_loss']:.4f} at epoch {self.history['best_epoch']}")

        # 如果有测试集，使用最佳模型评估测试集
        if self.test_loader:
            self.evaluate_test_set()

    def calculate_r2(self, predictions, true_values):
        """计算R²决定系数"""
        mean_true = np.mean(true_values)
        ss_tot = np.sum((true_values - mean_true) ** 2)
        ss_res = np.sum((true_values - predictions) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))  # 添加小值防止除零
        return r2

    def evaluate_test_set(self):
        """评估测试集性能"""
        print("\nEvaluating on test set...")
        # 先加载最佳模型
        self.load_checkpoint('best_model.pth')

        # 评估测试集
        test_loss, test_preds, test_true = self.validate(self.test_loader, desc='Testing')

        # 计算测试集指标
        test_r2 = self.calculate_r2(test_preds, test_true)
        test_rmse = np.sqrt(np.mean((test_preds - test_true) ** 2))
        test_mae = np.mean(np.abs(test_preds - test_true))

        # 输出测试集结果
        print(f"Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  R²: {test_r2:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE: {test_mae:.4f}")

        # 绘制测试集预测图
        self.plot_predictions(test_preds, test_true, epoch='test', save_path=os.path.join(self.save_dir, 'test_predictions.png'))

        # 保存测试集结果
        test_results = {
            'loss': float(test_loss),
            'r2': float(test_r2),
            'rmse': float(test_rmse),
            'mae': float(test_mae),
            'predictions': test_preds.tolist(),
            'true_values': test_true.tolist()
        }

        with open(os.path.join(self.save_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f)

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
        checkpoint = torch.load(os.path.join(self.save_dir, filename), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']

        return checkpoint

    def plot_predictions(self, predictions, true_values, epoch, save_path=None):
        """绘制预测结果图"""
        plt.figure(figsize=(10, 8))

        # 计算性能指标
        r2 = self.calculate_r2(predictions, true_values)
        rmse = np.sqrt(np.mean((predictions - true_values) ** 2))

        # 绘制散点图
        plt.scatter(true_values, predictions, alpha=0.6, s=40)

        # 添加完美预测线
        min_val = min(np.min(true_values), np.min(predictions))
        max_val = max(np.max(true_values), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        # 添加回归线
        z = np.polyfit(true_values, predictions, 1)
        p = np.poly1d(z)
        plt.plot(true_values, p(true_values), 'g-', lw=1.5, alpha=0.7)

        # 添加图表标题和轴标签
        if isinstance(epoch, str):
            title = f'Predictions vs True Values - {epoch}'
        else:
            title = f'Predictions vs True Values - Epoch {epoch}'
        plt.title(f'{title}\nR² = {r2:.4f}, RMSE = {rmse:.4f}')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.grid(True, alpha=0.3)

        # 设置轴范围
        margin = (max_val - min_val) * 0.05  # 5% margin
        plt.xlim(min_val - margin, max_val + margin)
        plt.ylim(min_val - margin, max_val + margin)

        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_learning_curves(self):
        """绘制学习曲线"""
        epochs = range(1, len(self.history['train_loss']) + 1)

        plt.figure(figsize=(12, 10))

        # 绘制损失曲线
        plt.subplot(2, 1, 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        plt.axvline(x=self.history['best_epoch'], color='g', linestyle='--', alpha=0.7)
        plt.text(self.history['best_epoch'], min(self.history['val_loss']),
                 f'Best: {self.history["best_val_loss"]:.4f}',
                 verticalalignment='bottom', horizontalalignment='right')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 绘制学习率曲线
        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.history['learning_rates'], 'g-')
        plt.title('Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.yscale('log')  # 对数尺度更容易看出学习率变化
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def save_history(self):
        """保存训练历史"""
        history_file = os.path.join(self.save_dir, 'training_history.json')
        # 转换NumPy数组为Python列表以便JSON序列化
        serializable_history = {
            'train_loss': [float(x) for x in self.history['train_loss']],
            'val_loss': [float(x) for x in self.history['val_loss']],
            'learning_rates': [float(x) for x in self.history['learning_rates']],
            'best_val_loss': float(self.history['best_val_loss']),
            'best_epoch': self.history['best_epoch']
        }
        with open(history_file, 'w') as f:
            json.dump(serializable_history, f)

def visualize_preprocessing(image_path, save_dir):
    """可视化不同预处理技术的效果"""
    # 原始图像
    original = Image.open(image_path).convert('RGB')

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 保存原始图像
    original.save(os.path.join(save_dir, '01_original.png'))

    # 应用各种预处理
    # 1. 固定角度旋转
    rotated = FixedRotation(p=1.0)(original)
    rotated.save(os.path.join(save_dir, '02_rotated.png'))

    # 2. 自适应边缘增强
    edge_enhanced = AdaptiveEdgeEnhancer(p=1.0)(original)
    edge_enhanced.save(os.path.join(save_dir, '03_edge_enhanced.png'))

    # 3. 对比度增强
    contrast_enhanced = ContrastTextureEnhancer(p=1.0)(original)
    contrast_enhanced.save(os.path.join(save_dir, '04_contrast_enhanced.png'))

    # 4. 完整增强链
    transform = transforms.Compose([
        AdaptiveEdgeEnhancer(p=1.0),
        ContrastTextureEnhancer(p=1.0)
    ])
    fully_enhanced = transform(original)
    fully_enhanced.save(os.path.join(save_dir, '05_fully_enhanced.png'))

    # 5. 对比可视化
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    # 显示图像
    axes[0].imshow(np.array(original))
    axes[0].set_title('Original')

    axes[1].imshow(np.array(rotated))
    axes[1].set_title('Rotated')

    axes[2].imshow(np.array(edge_enhanced))
    axes[2].set_title('Edge Enhanced')

    axes[3].imshow(np.array(contrast_enhanced))
    axes[3].set_title('Contrast Enhanced')

    axes[4].imshow(np.array(fully_enhanced))
    axes[4].set_title('Fully Enhanced')

    # 移除刻度
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '00_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Preprocessing visualizations saved to {save_dir}")

def inference_example(model_path, image_path, device='cuda'):
    """推理示例，展示如何使用保存的模型进行预测"""
    # 加载模型
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)

    # 创建模型实例
    model = FrozenCNNRegressor(backbone='densenet121', pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载图像
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 执行推理
    with torch.no_grad():
        prediction = model(input_tensor).item()

    return prediction

def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 设置路径 - 适应新的数据集结构
    base_dir = "/home/lsy/gbx_cropping_ws/src/image_tools/only_fifteen"
    raw_dataset_path = os.path.join(base_dir, "dataset")            # 包含所有原始图片和labels.csv的目录
    labels_file = os.path.join(raw_dataset_path, "labels.csv")      # 标签文件路径
    split_dataset_path = os.path.join(base_dir, "split_dataset")    # 分割后数据集的存放路径
    augmented_dataset_path = os.path.join(base_dir, "augmented_dataset")
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # 检查标签文件是否存在
    if not os.path.exists(labels_file):
        raise ValueError(f"Labels file not found: {labels_file}")

    print(f"Found labels file: {labels_file}")

    # 创建实验文件夹，带有时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(save_dir, f"frozen_cnn_experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # 分割数据集
    print("Splitting dataset into train, validation, and test sets...")
    splitter = DatasetSplitter(
        source_dir=raw_dataset_path,
        labels_file=labels_file,
        target_dir=split_dataset_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    splitter.split_dataset()

    # 首先进行数据增强
    augmenter = DatasetAugmenter(augmentation_factor=5)  # 每张图片生成5个增强版本

    # 增强训练集
    print("Augmenting training set...")
    augmenter.augment_dataset(
        os.path.join(split_dataset_path, 'train'),
        os.path.join(augmented_dataset_path, 'train')
    )

    # 增强验证集
    print("Augmenting validation set...")
    augmenter.augment_dataset(
        os.path.join(split_dataset_path, 'val'),
        os.path.join(augmented_dataset_path, 'val')
    )

    # 增强测试集
    print("Augmenting test set...")
    augmenter.augment_dataset(
        os.path.join(split_dataset_path, 'test'),
        os.path.join(augmented_dataset_path, 'test')
    )

    # 可视化一些预处理步骤
    # 找到第一张图片进行可视化
    df = pd.read_csv(labels_file)
    if len(df) > 0:
        first_image = df.iloc[0]['image_name']
        sample_img_path = os.path.join(raw_dataset_path, first_image)
        if os.path.exists(sample_img_path):
            visualize_preprocessing(
                sample_img_path,
                os.path.join(experiment_dir, 'preprocessing_visualization')
            )
        else:
            print(f"Warning: Could not find first image at {sample_img_path}")
    else:
        print("No images found in the labels file")

    # 设置用于训练的数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载增强后的数据集
    train_dataset = RegressionDataset(
        os.path.join(augmented_dataset_path, 'train'),
        transform=transform
    )
    val_dataset = RegressionDataset(
        os.path.join(augmented_dataset_path, 'val'),
        transform=transform
    )
    test_dataset = RegressionDataset(
        os.path.join(augmented_dataset_path, 'test'),
        transform=transform
    )

    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 选择要使用的骨干网络
    backbone = 'densenet121'  # 可以选择: 'resnet34', 'densenet121', 'mobilenet_v2' 等

    # 创建冻结CNN+FC模型
    model = FrozenCNNRegressor(backbone=backbone, pretrained=True, initial_value=15.0)

    # 验证哪些层是冻结的
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"创建了使用{backbone}骨干网络的冻结CNN+FC模型")
    print(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,} ({trainable_params/total_params:.2%})")

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=0.001,
        save_dir=experiment_dir
    )

    # 训练模型，在80 epoch时解冻最后几层进行微调
    trainer.train(num_epochs=200, eval_every=1, unfreeze_at_epoch=80)

    trainer.evaluate_all_datasets()

    # 保存配置信息
    config = {
        'backbone': backbone,
        'augmentation_factor': 5,
        'batch_size': 32,
        'initial_lr': 0.001,
        'num_epochs': 200,
        'unfreeze_at_epoch': 80,
        'experiment_timestamp': timestamp,
        'model_type': 'FrozenCNN_with_FC'
    }

    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    # 创建推理示例代码
    inference_code = """
# 推理示例代码
import torch
from torchvision import transforms
from PIL import Image

def predict_image(model_path, image_path, device='cuda'):
    # 加载模型
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 执行推理
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    return prediction

# 使用示例
# prediction = predict_image('best_model.pth', 'test_image.jpg')
# print(f"预测值: {prediction:.2f}")
"""

    with open(os.path.join(experiment_dir, 'inference_example.py'), 'w') as f:
        f.write(inference_code)

    print(f"训练完成！结果保存在 {experiment_dir}")
    print("提示: 使用保存的'best_model.pth'文件进行推理，示例代码已保存到inference_example.py")

if __name__ == "__main__":
    main()