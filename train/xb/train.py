import os
import shutil
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# 设置路径
PROCESSED_DATA_DIR = '/home/siyang_liu/gbx_cropping_ws/train/whole/augmented_dataset'
MODEL_SAVE_PATH = '/home/siyang_liu/gbx_cropping_ws/train/xb/whole_fifteen_epoch200/model.pth'

# 超参数设置
BATCH_SIZE = 64
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集定义
class MPPDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx]['image_name'])
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.labels_df.iloc[idx]['label'], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

# 数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据加载
train_dataset = MPPDataset(os.path.join(PROCESSED_DATA_DIR, 'train/train_labels.csv'),
                           os.path.join(PROCESSED_DATA_DIR, 'train'),
                           transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = MPPDataset(os.path.join(PROCESSED_DATA_DIR, 'val/val_labels.csv'),
                         os.path.join(PROCESSED_DATA_DIR, 'val'),
                         transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 模型定义
from torchvision.models import ResNet50_Weights
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1)
)
model = model.to(DEVICE)

# 损失函数与优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练函数
def train_model():
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            # 不使用squeeze，使用完全控制的方法
            outputs = model(images)
            outputs = outputs.reshape(-1)  # 确保形状正确
            labels = labels.reshape(-1)    # 确保形状正确

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}')

    # 确保保存目录存在
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print('Training completed and model saved.')

if __name__ == '__main__':
    train_model()