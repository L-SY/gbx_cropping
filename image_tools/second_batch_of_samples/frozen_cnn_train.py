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
from sklearn.model_selection import train_test_split, StratifiedKFold

# matplotlib导入
import matplotlib.pyplot as plt
import seaborn as sns

class FixedRotation(object):
    """Fixed angle rotation: only rotate 0°, 90°, 180° or 270°"""
    def __init__(self, p=0.75):
        """
        Parameters:
            p: probability to apply rotation (0° angle probability is 1-p)
        """
        self.p = p
        self.angles = [90, 180, 270]  # possible rotation angles

    def __call__(self, img):
        if torch.rand(1) < self.p:
            # randomly select an angle
            angle = self.angles[torch.randint(0, len(self.angles), (1,)).item()]
            return img.rotate(angle)
        return img  # no rotation (0°)

class AdaptiveEdgeEnhancer(object):
    """Adaptive edge enhancer"""
    def __init__(self, alpha=1.5, beta=0.5, p=0.7):
        """
        Parameters:
            alpha: edge enhancement strength
            beta: original image retention ratio
            p: probability to apply transformation
        """
        self.alpha = alpha
        self.beta = beta
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            # Convert to numpy array
            img_np = np.array(img)

            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np

            # Use adaptive threshold method - Gaussian weights, block size 11, constant 2
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Use Canny for further edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Apply morphological operations to connect nearby edges
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            edges = cv2.erode(edges, kernel, iterations=1)

            # Combine both edge effects
            combined_edges = cv2.bitwise_or(binary, edges)

            # For color images, use edge enhancement
            if len(img_np.shape) == 3:
                # Create edge mask
                edge_mask = combined_edges / 255.0
                edge_mask_3d = np.stack([edge_mask] * 3, axis=2)

                # Sharpen original image
                sharpened = img_np.astype(float)
                blurred = cv2.GaussianBlur(img_np, (0, 0), 3)
                sharpened = cv2.addWeighted(img_np, 1.5, blurred, -0.5, 0)

                # Blend original image and edge information
                result = img_np * self.beta + sharpened * (1 - self.beta)
                # Extra enhancement at edge positions
                result = result * (1 - edge_mask_3d * self.alpha) + sharpened * (edge_mask_3d * self.alpha)
                result = np.clip(result, 0, 255).astype(np.uint8)

                return Image.fromarray(result)

            else:
                # Grayscale processing
                sharpened = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0, 0), 3), -0.5, 0)
                result = gray * self.beta + sharpened * (1 - self.beta)
                result = result * (1 - edge_mask * self.alpha) + sharpened * (edge_mask * self.alpha)
                result = np.clip(result, 0, 255).astype(np.uint8)

                return Image.fromarray(result)

        return img

class ContrastTextureEnhancer(object):
    """Contrast-aware texture enhancer"""
    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8), p=0.7):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            # Convert to numpy array
            img_np = np.array(img)

            # Convert to LAB color space
            if len(img_np.shape) == 3:  # color image
                lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)

                # Apply CLAHE to L channel (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
                cl = clahe.apply(l)

                # Merge back to LAB and convert to RGB
                enhanced_lab = cv2.merge((cl, a, b))
                enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

                return Image.fromarray(enhanced_rgb)
            else:  # grayscale
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
                enhanced = clahe.apply(img_np)
                return Image.fromarray(enhanced)

        return img

class DatasetSplitter:
    """Split original dataset into training, validation and test sets with improved distribution balancing"""
    def __init__(self, source_dir, labels_file, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        self.source_dir = source_dir
        self.labels_file = labels_file
        self.target_dir = target_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        # Ensure ratios sum to 1
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

        # 按标签值排序
        valid_df = valid_df.sort_values('label')

        # 使用均匀采样而非随机采样或分层采样
        # 这样确保各个数据集都能覆盖完整的标签分布
        indices = np.arange(len(valid_df))

        # 分配索引到不同集合，确保每个值区间都均匀分布
        train_indices = []
        val_indices = []
        test_indices = []

        # 使用系统采样方法分配样本
        for i in range(len(indices)):
            r = i % 5  # 使用周期为5的系统采样
            if r < int(5 * self.train_ratio):
                train_indices.append(i)
            elif r < int(5 * (self.train_ratio + self.val_ratio)):
                val_indices.append(i)
            else:
                test_indices.append(i)

        # 创建数据子集
        train_df = valid_df.iloc[train_indices].copy()
        val_df = valid_df.iloc[val_indices].copy()
        test_df = valid_df.iloc[test_indices].copy()

        print(f"Split dataset: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test")

        # 分析数据分布
        self._analyze_distribution(train_df, val_df, test_df)

        # 保存子集
        self._save_subset(train_df, 'train')
        self._save_subset(val_df, 'val')
        self._save_subset(test_df, 'test')

    def _analyze_distribution(self, train_df, val_df, test_df):
        """分析各数据集的标签分布"""
        print("\nAnalyzing label distribution:")

        # 计算各数据集的标签范围和统计信息
        train_min, train_max = train_df['label'].min(), train_df['label'].max()
        val_min, val_max = val_df['label'].min(), val_df['label'].max()
        test_min, test_max = test_df['label'].min(), test_df['label'].max()

        print(f"Train label range: {train_min:.2f} to {train_max:.2f}, mean={train_df['label'].mean():.2f}, std={train_df['label'].std():.2f}")
        print(f"Val label range: {val_min:.2f} to {val_max:.2f}, mean={val_df['label'].mean():.2f}, std={val_df['label'].std():.2f}")
        print(f"Test label range: {test_min:.2f} to {test_max:.2f}, mean={test_df['label'].mean():.2f}, std={test_df['label'].std():.2f}")

        # 使用直方图可视化分布
        plt.figure(figsize=(12, 8))

        plt.hist(train_df['label'], alpha=0.5, bins=20, label='Train')
        plt.hist(val_df['label'], alpha=0.5, bins=20, label='Validation')
        plt.hist(test_df['label'], alpha=0.5, bins=20, label='Test')

        plt.title('Label Distribution Across Datasets')
        plt.xlabel('Label Value')
        plt.ylabel('Frequency')
        plt.legend()

        # 保存分布图
        os.makedirs(os.path.join(self.target_dir, 'analysis'), exist_ok=True)
        plt.savefig(os.path.join(self.target_dir, 'analysis', 'label_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _save_subset(self, df, subset_name):
        """Save a subset of data"""
        target_dir = os.path.join(self.target_dir, subset_name)

        # Copy images and save labels
        for idx, row in tqdm(df.iterrows(), desc=f"Copying {subset_name} set", total=len(df)):
            # Build source and destination paths
            src_path = os.path.join(self.source_dir, row['image_name'])
            dst_path = os.path.join(target_dir, row['image_name'])

            # Copy image
            try:
                image = Image.open(src_path)
                image.save(dst_path)
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
                continue

        # Save labels file
        df.to_csv(os.path.join(target_dir, f'{subset_name}_labels.csv'), index=False)
        print(f"Saved {len(df)} images to {subset_name} set")

class DatasetAugmenter:
    """Dataset augmenter - modified version"""
    def __init__(self, augmentation_factor=5, is_training=True):
        self.augmentation_factor = augmentation_factor

        # Determine the transform based on whether this is for training data
        if is_training:
            self.transform = transforms.Compose([
                # 1) Fixed angle rotation (0°, 90°, 180°, 270°)
                FixedRotation(p=0.75),
                # 2) Adaptive edge enhancement
                AdaptiveEdgeEnhancer(alpha=1.7, beta=0.4, p=0.8),
                # 3) Contrast-aware texture enhancement
                ContrastTextureEnhancer(
                    clip_limit=3.0, tile_grid_size=(8, 8), p=0.7),
                # Resize to target size
                transforms.Resize((224, 224)),
            ])
        else:
            # Lighter transforms for validation/test
            self.transform = transforms.Compose([
                # Only essential preprocessing for validation/test
                FixedRotation(p=0.75),

                # AdaptiveEdgeEnhancer(alpha=1.7, beta=0.4, p=0.8),
                # # 3) Contrast-aware texture enhancement
                # ContrastTextureEnhancer(
                #     clip_limit=3.0, tile_grid_size=(8, 8), p=0.7),
                transforms.Resize((224, 224)),
            ])

    def augment_dataset(self, source_dir, target_dir, is_training=True):
        """增强数据集并保存"""
        os.makedirs(target_dir, exist_ok=True)

        # 读取原始标签文件
        subset_name = os.path.basename(source_dir)  # 'train', 'val', 或 'test'
        labels_file = os.path.join(source_dir, f'{subset_name}_labels.csv')

        if not os.path.exists(labels_file):
            raise ValueError(f"Labels file not found: {labels_file}")

        print(f"Reading labels from {labels_file}")
        original_df = pd.read_csv(labels_file)
        print(f"Found {len(original_df)} entries in labels file")

        # 用于存储新的标签
        new_records = []

        # 对每张图片进行处理
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
            resized_image = transforms.Resize((224, 224))(image)  # 确保所有图像都调整为相同大小
            resized_image.save(os.path.join(target_dir, original_name))
            new_records.append({
                'image_name': original_name,
                'label': row['label']
            })

            # 为训练集生成增强图片
            if self.augmentation_factor > 0:
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
    """Custom dataset class"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Read labels file
        subset_name = os.path.basename(data_dir)  # 'train', 'val', or 'test'
        labels_file = os.path.join(data_dir, f'{subset_name}_labels.csv')

        if not os.path.exists(labels_file):
            raise ValueError(f"Labels file not found: {labels_file}")

        self.labels_df = pd.read_csv(labels_file)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Get image path and label
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.data_dir, row['image_name'])
        label = row['label']

        # Read and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

class FrozenCNNRegressor(nn.Module):
    """Texture regression model using frozen CNN feature extractor and trainable FC layers"""
    def __init__(self, backbone='densenet121', pretrained=True, initial_value=15.0, dropout_rate=0.5):
        super(FrozenCNNRegressor, self).__init__()

        # Load pretrained backbone network
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
            # Remove global average pooling and fully connected layers
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
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze feature extractor
        for param in self.features.parameters():
            param.requires_grad = False

        # Global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Regression head with L2 regularization effect (like Ridge regression)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(64, 1)
        )

        # Initialize the bias of the final layer to the specified value
        final_layer = self.regressor[-1]
        nn.init.constant_(final_layer.bias, initial_value)

    def forward(self, x):
        # Extract features (frozen stage)
        with torch.no_grad():
            features = self.features(x)

        # Global average pooling
        pooled = self.global_pool(features)

        # Regression prediction (trainable part)
        output = self.regressor(pooled).squeeze()

        return output

    def unfreeze_last_layers(self, num_layers=2):
        """Unfreeze last few layers of CNN feature extractor for fine-tuning"""
        # Implementation depends on the specific backbone
        if isinstance(self.features, nn.Sequential):
            # This is suitable for ResNet and other sequential models
            for i, module in enumerate(list(self.features.children())[-num_layers:]):
                for param in module.parameters():
                    param.requires_grad = True
            print(f"Unfrozen last {num_layers} sequential modules")
        elif hasattr(self.features, 'denseblock4'):
            # This is suitable for DenseNet
            for param in self.features.denseblock4.parameters():
                param.requires_grad = True
            for param in self.features.norm5.parameters():
                param.requires_grad = True
            print(f"Unfrozen DenseNet's last dense block and norm layer")
        else:
            print("Unknown backbone structure, no layers unfrozen")

class MixedRegressionLoss(nn.Module):
    """Mixed regression loss function"""
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
    """Trainer class"""
    def __init__(self, model, train_loader, val_loader, test_loader=None, device='cuda',
                 learning_rate=0.001, save_dir='checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Define optimizer and mixed loss function
        self.criterion = MixedRegressionLoss(mse_weight=0.5, l1_weight=0.3, huber_weight=0.2)

        # Get only parameters with requires_grad=True for optimization
        # This will only update unfrozen parameters, greatly improving training efficiency
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

        print(f"Trainable parameters: {len(params_to_update)}")

        # Use AdamW optimizer with L2 regularization
        self.optimizer = optim.AdamW(params_to_update, lr=learning_rate, weight_decay=1e-4)

        # Learning rate scheduler - cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,        # restart period
            T_mult=2,      # period multiplier after each restart
            eta_min=1e-6   # minimum learning rate
        )

        # Record training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }

        # Early stopping counter
        self.patience = 80  # Increased from 25 to 30 epochs
        self.patience_counter = 0
        self.early_stop = False

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc='Training')

        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()

            # Gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(self.train_loader)

    def validate(self, data_loader, desc='Validating'):
        """Validate the model"""
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
        """Evaluate all datasets and generate comprehensive report"""
        print("\nEvaluating all datasets...")
        # Load best model
        self.load_checkpoint('best_model.pth')

        # Create results storage dictionary
        results = {}

        # Evaluate training set
        train_loss, train_preds, train_true = self.validate(self.train_loader, desc='Evaluating Training Set')
        results['train'] = {
            'loss': train_loss,
            'predictions': train_preds,
            'true_values': train_true,
            'r2': self.calculate_r2(train_preds, train_true),
            'rmse': np.sqrt(np.mean((train_preds - train_true) ** 2)),
            'mae': np.mean(np.abs(train_preds - train_true))
        }

        # Evaluate validation set
        val_loss, val_preds, val_true = self.validate(self.val_loader, desc='Evaluating Validation Set')
        results['val'] = {
            'loss': val_loss,
            'predictions': val_preds,
            'true_values': val_true,
            'r2': self.calculate_r2(val_preds, val_true),
            'rmse': np.sqrt(np.mean((val_preds - val_true) ** 2)),
            'mae': np.mean(np.abs(val_preds - val_true))
        }

        # Evaluate test set
        test_loss, test_preds, test_true = self.validate(self.test_loader, desc='Evaluating Test Set')
        results['test'] = {
            'loss': test_loss,
            'predictions': test_preds,
            'true_values': test_true,
            'r2': self.calculate_r2(test_preds, test_true),
            'rmse': np.sqrt(np.mean((test_preds - test_true) ** 2)),
            'mae': np.mean(np.abs(test_preds - test_true))
        }

        # Save results
        self.plot_all_datasets_comparison(results)
        self.create_metrics_table(results)

        # Analyze the validation performance
        self.analyze_validation_performance(results)

        return results

    def analyze_validation_performance(self, results):
        """Analyze validation set performance to identify potential issues"""
        print("\nAnalyzing validation set performance...")

        # Compare distributions
        train_mean = np.mean(results['train']['true_values'])
        train_std = np.std(results['train']['true_values'])
        val_mean = np.mean(results['val']['true_values'])
        val_std = np.std(results['val']['true_values'])
        test_mean = np.mean(results['test']['true_values'])
        test_std = np.std(results['test']['true_values'])

        print(f"Training set: mean={train_mean:.2f}, std={train_std:.2f}")
        print(f"Validation set: mean={val_mean:.2f}, std={val_std:.2f}")
        print(f"Test set: mean={test_mean:.2f}, std={test_std:.2f}")

        # Calculate distribution difference
        train_val_diff = abs(train_mean - val_mean) / train_std
        train_test_diff = abs(train_mean - test_mean) / train_std

        print(f"Normalized mean difference (train-val): {train_val_diff:.4f}")
        print(f"Normalized mean difference (train-test): {train_test_diff:.4f}")

        # Identify outliers in validation set
        val_predictions = results['val']['predictions']
        val_true = results['val']['true_values']
        errors = np.abs(val_predictions - val_true)
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        outlier_threshold = mean_error + 2 * std_error

        outliers = np.where(errors > outlier_threshold)[0]
        if len(outliers) > 0:
            print(f"Found {len(outliers)} potential outliers in validation set")
            print(f"Outlier threshold: {outlier_threshold:.4f}")
            print(f"Top 5 largest errors:")
            top_errors = np.argsort(errors)[-5:]
            for idx in reversed(top_errors):
                print(f"  True: {val_true[idx]:.2f}, Predicted: {val_predictions[idx]:.2f}, Error: {errors[idx]:.2f}")

        # Plot distribution comparison
        plt.figure(figsize=(10, 6))
        plt.hist(results['train']['true_values'], alpha=0.5, bins=20, label='Training')
        plt.hist(results['val']['true_values'], alpha=0.5, bins=20, label='Validation')
        plt.hist(results['test']['true_values'], alpha=0.5, bins=20, label='Test')
        plt.title('Label Distribution Comparison')
        plt.xlabel('Label Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'distribution_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Plot error distribution
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=20)
        plt.axvline(x=outlier_threshold, color='r', linestyle='--', label=f'Outlier threshold: {outlier_threshold:.2f}')
        plt.title('Validation Set Error Distribution')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'validation_error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_all_datasets_comparison(self, results):
        """Plot predictions vs true values comparison for all datasets"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        datasets = ['train', 'val', 'test']
        titles = ['Training Set', 'Validation Set', 'Test Set']
        colors = ['blue', 'green', 'red']

        for i, (dataset, title, color) in enumerate(zip(datasets, titles, colors)):
            data = results[dataset]
            axes[i].scatter(data['true_values'], data['predictions'], alpha=0.6, s=30, c=color)

            # Add perfect prediction line
            min_val = min(np.min(data['true_values']), np.min(data['predictions']))
            max_val = max(np.max(data['true_values']), np.max(data['predictions']))
            axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)

            # Add regression line
            z = np.polyfit(data['true_values'], data['predictions'], 1)
            p = np.poly1d(z)
            axes[i].plot(data['true_values'], p(data['true_values']), 'g-', lw=1.5, alpha=0.7)

            # Set title and labels
            axes[i].set_title(f'{title}\nR² = {data["r2"]:.4f}, RMSE = {data["rmse"]:.4f}')
            axes[i].set_xlabel('True Values')
            axes[i].set_ylabel('Predicted Values')
            axes[i].grid(True, alpha=0.3)

            # Set axis range
            margin = (max_val - min_val) * 0.05  # 5% margin
            axes[i].set_xlim(min_val - margin, max_val + margin)
            axes[i].set_ylim(min_val - margin, max_val + margin)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'all_datasets_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_metrics_table(self, results):
        """Create a table with performance metrics for all datasets"""
        # Prepare table data
        metrics_data = {
            'Dataset': ['Training Set', 'Validation Set', 'Test Set'],
            'Sample Size': [
                len(results['train']['true_values']),
                len(results['val']['true_values']),
                len(results['test']['true_values'])
            ],
            'Loss': [
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

        # Create table visualization
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')

        # Create table
        table = ax.table(
            cellText=[
                [f"{metrics_data['Dataset'][i]}",
                 f"{metrics_data['Sample Size'][i]}",
                 f"{metrics_data['Loss'][i]:.4f}",
                 f"{metrics_data['R²'][i]:.4f}",
                 f"{metrics_data['RMSE'][i]:.4f}",
                 f"{metrics_data['MAE'][i]:.4f}"]
                for i in range(3)
            ],
            colLabels=['Dataset', 'Sample Size', 'Loss', 'R²', 'RMSE', 'MAE'],
            loc='center',
            cellLoc='center'
        )

        # Set table style
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)

        # Set header style
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4472C4')
            elif j == 0:  # First column
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#D9E1F2')
            elif i % 2 == 1:  # Odd rows
                cell.set_facecolor('#E9EDF4')

        plt.title('Model Performance Metrics Across Datasets', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'performance_metrics_table.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Save metrics as CSV
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(os.path.join(self.save_dir, 'performance_metrics.csv'), index=False)

        # Save as HTML table
        html_table = metrics_df.to_html(index=False)
        with open(os.path.join(self.save_dir, 'performance_metrics.html'), 'w') as f:
            f.write("<html><head><style>")
            f.write("table {border-collapse: collapse; width: 100%; margin: 20px 0;}")
            f.write("th {background-color: #4472C4; color: white; font-weight: bold; text-align: center; padding: 10px;}")
            f.write("td {padding: 8px; text-align: center; border: 1px solid #ddd;}")
            f.write("tr:nth-child(even) {background-color: #E9EDF4;}")
            f.write("tr:hover {background-color: #ddd;}")
            f.write("</style></head><body>")
            f.write("<h2>Model Performance Metrics</h2>")
            f.write(html_table)
            f.write("</body></html>")

    def train(self, num_epochs, eval_every=1, unfreeze_at_epoch=None):
        """Complete training process"""
        for epoch in range(num_epochs):
            if self.early_stop:
                print("Early stopping triggered!")
                break

            # Unfreeze part of CNN layers at specified epoch
            if unfreeze_at_epoch and epoch == unfreeze_at_epoch:
                print(f"Epoch {epoch+1}: Unfreezing last layers of CNN for fine-tuning")
                self.model.unfreeze_last_layers(num_layers=2)
                # Adjust learning rate to 1/10 of original to avoid destroying pretrained features
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")

            # Training phase
            train_loss = self.train_epoch()

            # Validation phase (based on eval_every parameter)
            if (epoch + 1) % eval_every == 0:
                val_loss, predictions, true_values = self.validate(self.val_loader)

                # Update learning rate
                self.scheduler.step()

                # Record history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['learning_rates'].append(current_lr)

                # Print results
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}")

                # Calculate validation R^2 and RMSE
                r2 = self.calculate_r2(predictions, true_values)
                rmse = np.sqrt(np.mean((predictions - true_values) ** 2))
                print(f"Val R²: {r2:.4f}, RMSE: {rmse:.4f}")

                # Save best model
                if val_loss < self.history['best_val_loss']:
                    self.history['best_val_loss'] = val_loss
                    self.history['best_epoch'] = epoch + 1
                    self.save_checkpoint(f'best_model.pth')
                    print(f"New best model saved with validation loss: {val_loss:.4f}")
                    # Reset early stopping counter
                    self.patience_counter = 0
                else:
                    # Increment early stopping counter
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"Early stopping after {self.patience} epochs without improvement")
                        self.early_stop = True

                # Save checkpoints every 5 evaluation cycles
                if (epoch + 1) % (eval_every * 5) == 0:
                    self.save_checkpoint(f'epoch_{epoch+1}.pth')

                # Plot and save current prediction results
                self.plot_predictions(predictions, true_values, epoch+1)

                # Plot learning curves every 10 evaluation cycles
                if (epoch + 1) % (eval_every * 10) == 0:
                    self.plot_learning_curves()

        # Save training history
        self.save_history()
        print(f"Training completed! Best validation loss: {self.history['best_val_loss']:.4f} at epoch {self.history['best_epoch']}")

        # If test set exists, evaluate it using the best model
        if self.test_loader:
            self.evaluate_test_set()

    def calculate_r2(self, predictions, true_values):
        """Calculate R² coefficient of determination"""
        mean_true = np.mean(true_values)
        ss_tot = np.sum((true_values - mean_true) ** 2)
        ss_res = np.sum((true_values - predictions) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))  # add small value to prevent division by zero
        return r2

    def evaluate_test_set(self):
        """Evaluate test set performance"""
        print("\nEvaluating on test set...")
        # Load best model first
        self.load_checkpoint('best_model.pth')

        # Evaluate test set
        test_loss, test_preds, test_true = self.validate(self.test_loader, desc='Testing')

        # Calculate test set metrics
        test_r2 = self.calculate_r2(test_preds, test_true)
        test_rmse = np.sqrt(np.mean((test_preds - test_true) ** 2))
        test_mae = np.mean(np.abs(test_preds - test_true))

        # Output test results
        print(f"Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  R²: {test_r2:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE: {test_mae:.4f}")

        # Plot test set predictions
        self.plot_predictions(test_preds, test_true, epoch='test', save_path=os.path.join(self.save_dir, 'test_predictions.png'))

        # Save test results
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
        """Save checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))

    def load_checkpoint(self, filename):
        """Load checkpoint"""
        checkpoint = torch.load(os.path.join(self.save_dir, filename), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Only load optimizer and scheduler if they exist and training is continuing
        if hasattr(self, 'optimizer') and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if hasattr(self, 'scheduler') and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        return checkpoint

    def plot_predictions(self, predictions, true_values, epoch, save_path=None):
        """Plot prediction results"""
        plt.figure(figsize=(10, 8))

        # Calculate performance metrics
        r2 = self.calculate_r2(predictions, true_values)
        rmse = np.sqrt(np.mean((predictions - true_values) ** 2))

        # Plot scatter plot
        plt.scatter(true_values, predictions, alpha=0.6, s=40)

        # Add perfect prediction line
        min_val = min(np.min(true_values), np.min(predictions))
        max_val = max(np.max(true_values), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        # Add regression line
        z = np.polyfit(true_values, predictions, 1)
        p = np.poly1d(z)
        plt.plot(true_values, p(true_values), 'g-', lw=1.5, alpha=0.7)

        # Add chart title and axis labels
        if isinstance(epoch, str):
            title = f'Predictions vs True Values - {epoch}'
        else:
            title = f'Predictions vs True Values - Epoch {epoch}'
        plt.title(f'{title}\nR² = {r2:.4f}, RMSE = {rmse:.4f}')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.grid(True, alpha=0.3)

        # Set axis range
        margin = (max_val - min_val) * 0.05  # 5% margin
        plt.xlim(min_val - margin, max_val + margin)
        plt.ylim(min_val - margin, max_val + margin)

        # Save image
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, f'predictions_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_learning_curves(self):
        """Plot learning curves"""
        epochs = range(1, len(self.history['train_loss']) + 1)

        plt.figure(figsize=(12, 10))

        # Plot loss curves
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

        # Plot learning rate curve
        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.history['learning_rates'], 'g-')
        plt.title('Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.yscale('log')  # log scale makes it easier to see learning rate changes
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def save_history(self):
        """Save training history"""
        history_file = os.path.join(self.save_dir, 'training_history.json')
        # Convert NumPy arrays to Python lists for JSON serialization
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
    """Visualize the effect of different preprocessing techniques"""
    # Original image
    original = Image.open(image_path).convert('RGB')

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Save original image
    original.save(os.path.join(save_dir, '01_original.png'))

    # Apply various preprocessing methods
    # 1. Fixed angle rotation
    rotated = FixedRotation(p=1.0)(original)
    rotated.save(os.path.join(save_dir, '02_rotated.png'))

    # 2. Adaptive edge enhancement
    edge_enhanced = AdaptiveEdgeEnhancer(p=1.0)(original)
    edge_enhanced.save(os.path.join(save_dir, '03_edge_enhanced.png'))

    # 3. Contrast enhancement
    contrast_enhanced = ContrastTextureEnhancer(p=1.0)(original)
    contrast_enhanced.save(os.path.join(save_dir, '04_contrast_enhanced.png'))

    # 4. Complete enhancement chain
    transform = transforms.Compose([
        AdaptiveEdgeEnhancer(p=1.0),
        ContrastTextureEnhancer(p=1.0)
    ])
    fully_enhanced = transform(original)
    fully_enhanced.save(os.path.join(save_dir, '05_fully_enhanced.png'))

    # 5. Comparison visualization
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    # Display images
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

    # Remove ticks
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '00_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Preprocessing visualizations saved to {save_dir}")

def inference_example(model_path, image_path, device='cuda'):
    """Inference example, showing how to use the saved model for prediction"""
    # Load model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Create model instance
    model = FrozenCNNRegressor(backbone='densenet121', pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        prediction = model(input_tensor).item()

    return prediction

def main():
    """Main function"""
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set paths - adapt to the new dataset structure
    base_dir = "/home/lsy/gbx_cropping_ws/src/image_tools/second_batch_of_samples/whole_second_batch_fifteen_samples"
    raw_dataset_path = os.path.join(base_dir, "cropping")            # Directory with all original images and labels.csv
    labels_file = os.path.join(raw_dataset_path, "labels.csv")      # Labels file path
    split_dataset_path = os.path.join(base_dir, "split_dataset")    # Path to store split dataset
    augmented_dataset_path = os.path.join(base_dir, "augmented_dataset")
    save_dir = "whole_second_batch_fifteen_samples/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # Check if labels file exists
    if not os.path.exists(labels_file):
        raise ValueError(f"Labels file not found: {labels_file}")

    print(f"Found labels file: {labels_file}")

    # Create experiment folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(save_dir, f"frozen_cnn_experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Split dataset
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

    # First do data augmentation
    print("Augmenting training set...")
    train_augmenter = DatasetAugmenter(augmentation_factor=15, is_training=True)  # Generate 5 augmented versions for each image
    train_augmenter.augment_dataset(
        os.path.join(split_dataset_path, 'train'),
        os.path.join(augmented_dataset_path, 'train'),
        is_training=True
    )

    # Augment validation set (with light transformation)
    print("Augmenting validation set...")
    val_augmenter = DatasetAugmenter(augmentation_factor=15, is_training=True)  # Only original images for validation
    val_augmenter.augment_dataset(
        os.path.join(split_dataset_path, 'val'),
        os.path.join(augmented_dataset_path, 'val'),
        is_training=True
    )

    # Augment test set (with light transformation)
    print("Augmenting test set...")
    test_augmenter = DatasetAugmenter(augmentation_factor=5, is_training=False)  # Only original images for test
    test_augmenter.augment_dataset(
        os.path.join(split_dataset_path, 'test'),
        os.path.join(augmented_dataset_path, 'test'),
        is_training=False
    )

    # Visualize some preprocessing steps
    # Find the first image for visualization
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

    # Set up data transforms for training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load augmented datasets
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

    # Create data loaders
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

    # Choose which backbone to use
    backbone = 'densenet121'  # Options: 'resnet34', 'densenet121', 'mobilenet_v2', etc.

    # Create frozen CNN+FC model with increased dropout
    model = FrozenCNNRegressor(backbone=backbone, pretrained=True, initial_value=15.0, dropout_rate=0.5)

    # Verify which layers are frozen
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Created frozen CNN+FC model using {backbone} backbone")
    print(f"Trainable parameters: {trainable_params:,} / Total parameters: {total_params:,} ({trainable_params/total_params:.2%})")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=0.001,
        save_dir=experiment_dir
    )

    # Train model, unfreeze last few layers at epoch 80 for fine-tuning
    trainer.train(num_epochs=200, eval_every=1, unfreeze_at_epoch=80)

    # Evaluate model on all datasets
    trainer.evaluate_all_datasets()

    # Save configuration information
    config = {
        'backbone': backbone,
        'augmentation_factor': 5,
        'batch_size': 32,
        'initial_lr': 0.001,
        'num_epochs': 200,
        'unfreeze_at_epoch': 80,
        'experiment_timestamp': timestamp,
        'model_type': 'FrozenCNN_with_FC',
        'dropout_rate': 0.5,
        'stratified_sampling': True
    }

    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    # Create inference example code
    inference_code = """
# Inference example code
import torch
from torchvision import transforms
from PIL import Image

def predict_image(model_path, image_path, device='cuda'):
    # Load model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model instance
    from your_model_file import FrozenCNNRegressor  # Import your model class
    model = FrozenCNNRegressor(backbone='densenet121', pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    return prediction

# Usage example
# prediction = predict_image('best_model.pth', 'test_image.jpg')
# print(f"Predicted value: {prediction:.2f}")
"""

    with open(os.path.join(experiment_dir, 'inference_example.py'), 'w') as f:
        f.write(inference_code)

    print(f"Training complete! Results saved to {experiment_dir}")
    print("Tip: Use the saved 'best_model.pth' file for inference, example code saved to inference_example.py")

if __name__ == "__main__":
    main()
