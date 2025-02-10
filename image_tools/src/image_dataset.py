import os
import pandas as pd
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageDatasetOrganizer:
    def __init__(self, image_folder, csv_path, output_base_path):
        """
        初始化数据组织器
        image_folder: 图片所在文件夹路径
        csv_path: 标签CSV文件路径
        output_base_path: 输出数据集的基础路径
        """
        self.image_folder = image_folder
        self.csv_path = csv_path
        self.output_base_path = output_base_path

        # 创建输出目录
        os.makedirs(output_base_path, exist_ok=True)

    def get_image_name_from_id(self, id_num):
        """根据ID生成对应的图片文件名"""
        # 将浮点数ID转换为整数字符串
        id_str = str(int(float(id_num)))  # 先转换为浮点数，再转整数，最后转字符串
        return f"cropped_raw_{id_str}.jpg"

    def load_and_verify_data(self):
        """加载并验证数据"""
        # 读取CSV文件
        self.labels_df = pd.read_csv(self.csv_path)

        # 获取所有图片文件
        self.image_files = {f: f for f in os.listdir(self.image_folder)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))}

        # 验证数据匹配
        missing_images = []
        self.valid_data = []

        for index, row in self.labels_df.iterrows():
            # 转换ID并生成图片名称
            id_num = str(int(float(row['ID'])))
            image_name = self.get_image_name_from_id(id_num)

            if image_name in self.image_files:
                self.valid_data.append({
                    'image_path': os.path.join(self.image_folder, image_name),
                    'label': row['ComputeRate'],
                    'image_name': image_name
                })
            else:
                missing_images.append(f"ID: {id_num}, Image: {image_name}")

        # 打印验证结果
        print(f"找到的有效数据对: {len(self.valid_data)}")
        if missing_images:
            print("\n缺失图片:")
            for item in missing_images[:10]:
                print(item)
            if len(missing_images) > 10:
                print(f"... 还有 {len(missing_images) - 10} 个未显示")

        return len(self.valid_data) > 0

    def prepare_dataset(self, test_size=0.2, val_size=0.2, random_state=42):
        """准备训练、验证和测试数据集"""
        # 首先分割出测试集
        train_val_data, test_data = train_test_split(
            self.valid_data,
            test_size=test_size,
            random_state=random_state
        )

        # 从剩余数据中分割出验证集
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_size,
            random_state=random_state
        )

        # 保存数据集划分
        self.save_dataset_split(train_data, 'train')
        self.save_dataset_split(val_data, 'val')
        self.save_dataset_split(test_data, 'test')

        # 创建数据集信息文件
        self.create_dataset_info(train_data, val_data, test_data)

    def save_dataset_split(self, data, split_name):
        """保存数据集划分"""
        split_dir = os.path.join(self.output_base_path, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # 保存图片和标签
        labels = []
        for item in data:
            # 复制图片
            dest_path = os.path.join(split_dir, item['image_name'])
            shutil.copy2(item['image_path'], dest_path)

            # 收集标签信息（只保存图片名称和标签）
            labels.append({
                'image_name': item['image_name'],
                'label': item['label']
            })

        # 保存标签CSV
        pd.DataFrame(labels).to_csv(
            os.path.join(split_dir, f'{split_name}_labels.csv'),
            index=False
        )

    def create_dataset_info(self, train_data, val_data, test_data):
        """创建数据集信息文件"""
        info = {
            'dataset_size': {
                'total': len(self.valid_data),
                'train': len(train_data),
                'val': len(val_data),
                'test': len(test_data)
            },
            'label_statistics': self.calculate_label_statistics(self.valid_data),
            'data_splits': {
                'train': [item['image_name'] for item in train_data],
                'val': [item['image_name'] for item in val_data],
                'test': [item['image_name'] for item in test_data]
            }
        }

        # 保存数据集信息
        pd.DataFrame([info]).to_json(
            os.path.join(self.output_base_path, 'dataset_info.json'),
            orient='records'
        )

    @staticmethod
    def calculate_label_statistics(data):
        """计算标签统计信息"""
        labels = [item['label'] for item in data]
        return {
            'mean': np.mean(labels),
            'std': np.std(labels),
            'min': np.min(labels),
            'max': np.max(labels),
            'count': len(labels)
        }

class RegressionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        初始化数据集
        data_dir: 数据目录路径
        transform: 图像变换
        """
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

def main():
    # 设置路径
    image_folder = "/home/lsy/gbx_cropping_ws/src/image_tools/cropping"
    csv_path = "/home/lsy/gbx_cropping_ws/src/image_tools/src/data_with_compute_rate.csv"
    output_base_path = "/home/lsy/gbx_cropping_ws/src/image_tools/dataset"

    # 创建数据组织器
    organizer = ImageDatasetOrganizer(image_folder, csv_path, output_base_path)

    # 加载和验证数据
    if organizer.load_and_verify_data():
        # 准备数据集
        organizer.prepare_dataset()

        # 示例：创建数据加载器
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 创建训练集
        train_dataset = RegressionDataset(
            os.path.join(output_base_path, 'train'),
            transform=transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4
        )

        print("数据集准备完成！")
        print(f"训练集大小: {len(train_dataset)}")

    else:
        print("数据验证失败，请检查输入文件")

if __name__ == "__main__":
    main()