# datasets.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


class RegressionDataset(Dataset):
    """回归数据集"""
    def __init__(self, data_dir, labels_file=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # 加载标签文件
        if labels_file is not None and os.path.exists(labels_file):
            print(f"使用提供的标签文件: {labels_file}")
            df = pd.read_csv(labels_file)
        else:
            # 尝试找到默认标签文件
            subset_name = os.path.basename(data_dir)
            default_file = os.path.join(data_dir, f"{subset_name}_labels.csv")
            alt_file = os.path.join(data_dir, "labels.csv")

            if os.path.exists(default_file):
                print(f"使用默认标签文件: {default_file}")
                df = pd.read_csv(default_file)
            elif os.path.exists(alt_file):
                print(f"使用替代标签文件: {alt_file}")
                df = pd.read_csv(alt_file)
            else:
                raise ValueError(f"找不到有效的标签文件")

        # 过滤掉不存在的图片
        valid_entries = []
        print("过滤不存在的图片...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="验证图片"):
            img_path = os.path.join(data_dir, row['image_name'])
            if os.path.exists(img_path):
                valid_entries.append(row)
            else:
                print(f"跳过不存在的图片: {img_path}")

        # 创建只包含有效条目的数据帧
        self.labels_df = pd.DataFrame(valid_entries)
        print(f"原始条目数: {len(df)}, 有效条目数: {len(self.labels_df)}")

        if len(self.labels_df) == 0:
            raise ValueError("没有找到有效的图片，请检查数据目录和标签文件")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # 获取图像路径和标签
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.data_dir, row['image_name'])
        label = row['label']

        # 读取图像
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

class DatasetSplitter:
    """将原始数据集分割为训练、验证和测试集"""
    def __init__(self, source_dir, labels_file, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        self.source_dir = source_dir
        self.labels_file = labels_file
        self.target_dir = target_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        # 确保比例之和为1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "比例之和必须为1"

    def split_dataset(self):
        """执行数据集分割"""
        print(f"从 {self.source_dir} 分割数据集")

        # 创建目标目录
        os.makedirs(os.path.join(self.target_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.target_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.target_dir, 'test'), exist_ok=True)

        # 读取标签文件
        df = pd.read_csv(self.labels_file)
        print(f"在标签文件中找到 {len(df)} 条记录")

        # 验证所有图片是否存在
        valid_entries = []
        for idx, row in df.iterrows():
            img_path = os.path.join(self.source_dir, row['image_name'])
            if os.path.exists(img_path):
                valid_entries.append(row)
            else:
                print(f"警告: 未找到图像: {img_path}")

        # 创建只包含有效条目的数据帧
        valid_df = pd.DataFrame(valid_entries)
        print(f"找到 {len(valid_df)} 张有效图像")

        if len(valid_df) == 0:
            raise ValueError("未找到有效图像。请检查源目录和标签文件。")

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

        print(f"分割数据集: {len(train_df)} 训练, {len(val_df)} 验证, {len(test_df)} 测试")

        # 分析数据分布
        self._analyze_distribution(train_df, val_df, test_df)

        # 保存子集
        self._save_subset(train_df, 'train')
        self._save_subset(val_df, 'val')
        self._save_subset(test_df, 'test')

    def _analyze_distribution(self, train_df, val_df, test_df):
        """分析各数据集的标签分布"""
        print("\n分析标签分布:")

        # 计算各数据集的标签范围和统计信息
        train_min, train_max = train_df['label'].min(), train_df['label'].max()
        val_min, val_max = val_df['label'].min(), val_df['label'].max()
        test_min, test_max = test_df['label'].min(), test_df['label'].max()

        print(f"训练集标签范围: {train_min:.2f} 到 {train_max:.2f}, 均值={train_df['label'].mean():.2f}, 标准差={train_df['label'].std():.2f}")
        print(f"验证集标签范围: {val_min:.2f} 到 {val_max:.2f}, 均值={val_df['label'].mean():.2f}, 标准差={val_df['label'].std():.2f}")
        print(f"测试集标签范围: {test_min:.2f} 到 {test_max:.2f}, 均值={test_df['label'].mean():.2f}, 标准差={test_df['label'].std():.2f}")

        # 使用直方图可视化分布
        import matplotlib.pyplot as plt
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
        """保存数据子集"""
        target_dir = os.path.join(self.target_dir, subset_name)

        # 复制图像并保存标签
        for idx, row in tqdm(df.iterrows(), desc=f"复制 {subset_name} 集", total=len(df)):
            # 构建源路径和目标路径
            src_path = os.path.join(self.source_dir, row['image_name'])
            dst_path = os.path.join(target_dir, row['image_name'])

            # 复制图像
            try:
                image = Image.open(src_path)
                image.save(dst_path)
            except Exception as e:
                print(f"复制 {src_path} 时出错: {e}")
                continue

        # 保存标签文件
        df.to_csv(os.path.join(target_dir, f'{subset_name}_labels.csv'), index=False)
        print(f"将 {len(df)} 张图像保存到 {subset_name} 集")

class DatasetAugmenter:
    """数据集增强器"""
    def __init__(self, augmentation_factor=5, is_training=True):
        from transforms import get_training_transform, get_validation_transform
        self.augmentation_factor = augmentation_factor

        # 根据是否为训练数据确定转换
        if is_training:
            self.transform = get_training_transform(border_width=70)
        else:
            # 对验证/测试使用更轻量的转换
            self.transform = get_validation_transform(border_width=70)

    def augment_dataset(self, source_dir, target_dir, is_training=True):
        """增强数据集并保存"""
        os.makedirs(target_dir, exist_ok=True)

        # 读取原始标签文件
        subset_name = os.path.basename(source_dir)  # 'train', 'val', 或 'test'
        labels_file = os.path.join(source_dir, f'{subset_name}_labels.csv')

        if not os.path.exists(labels_file):
            raise ValueError(f"未找到标签文件: {labels_file}")

        print(f"从 {labels_file} 读取标签")
        original_df = pd.read_csv(labels_file)
        print(f"在标签文件中找到 {len(original_df)} 条记录")

        # 用于存储新的标签
        new_records = []

        # 对每张图片进行处理
        for idx, row in tqdm(original_df.iterrows(), desc=f"增强 {subset_name} 数据集"):
            img_path = os.path.join(source_dir, row['image_name'])

            if not os.path.exists(img_path):
                print(f"警告: 未找到图像: {img_path}")
                continue

            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"打开图像 {img_path} 时出错: {e}")
                continue

            # 保存原始图片
            original_name = f"orig_{row['image_name']}"
            from torchvision import transforms
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

        print(f"增强 {subset_name} 数据集: {len(original_df)} 原始图像 -> {len(new_df)} 总图像")