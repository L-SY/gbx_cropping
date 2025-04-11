import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
from tqdm import tqdm

# 设置路径
PROCESSED_DATA_DIR = '/home/siyang_liu/gbx_cropping_ws/train/first_second/augmented_dataset'
MODEL_SAVE_DIR = '/home/siyang_liu/gbx_cropping_ws/train/xb/first_second'
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')

# 超参数设置
BATCH_SIZE = 64
NUM_EPOCHS = 200
INITIAL_LR = 1e-4  # 初始学习率
MIN_LR = 1e-6  # 最小学习率
WEIGHT_DECAY = 1e-4  # 权重衰减，帮助正则化
DROPOUT_RATE = 0.3  # 还原为原始的Dropout率
PATIENCE = 20  # 早停的耐心值
LR_PATIENCE = 5  # 学习率调整的耐心值
LR_FACTOR = 0.5  # 学习率衰减因子
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 确保模型保存目录存在
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

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

# 还原为原始的模型定义
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(DROPOUT_RATE),
    nn.Linear(256, 1)
)
model = model.to(DEVICE)

# 损失函数与优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)

# 学习率调度器 - 根据验证损失进行调整
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=LR_FACTOR,
    patience=LR_PATIENCE,
    verbose=True,
    min_lr=MIN_LR
)

# 早停工具类
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement
            path (str): Path to save the best model
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
            return True  # Model improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # Model did not improve

    def save_checkpoint(self, model):
        """Save model checkpoint"""
        torch.save(model.state_dict(), self.path)

    def load_checkpoint(self, model):
        """Load best model checkpoint"""
        model.load_state_dict(torch.load(self.path))

# 获取当前学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# 训练函数
def train_model():
    # 初始化早停工具
    early_stopping = EarlyStopping(patience=PATIENCE, path=BEST_MODEL_PATH)

    # 存储训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }

    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{NUM_EPOCHS}')

        for images, labels in train_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images).reshape(-1)
            labels = labels.reshape(-1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * images.size(0)
            train_bar.set_postfix({'loss': loss.item(), 'lr': get_lr(optimizer)})

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)
        history['learning_rates'].append(get_lr(optimizer))

        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{NUM_EPOCHS}')

        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images).reshape(-1)
                labels = labels.reshape(-1)

                loss = criterion(outputs, labels)
                total_val_loss += loss.item() * images.size(0)
                val_bar.set_postfix({'loss': loss.item()})

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        history['val_loss'].append(avg_val_loss)

        # 更新学习率
        current_lr = get_lr(optimizer)
        scheduler.step(avg_val_loss)
        new_lr = get_lr(optimizer)

        if new_lr != current_lr:
            print(f'Learning rate decreased from {current_lr:.6f} to {new_lr:.6f}')

        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'LR: {get_lr(optimizer):.6f}')

        # 检查是否需要早停
        model_improved = early_stopping(avg_val_loss, model)
        if model_improved:
            print(f'Validation loss improved to {avg_val_loss:.4f}. Model saved!')

        if early_stopping.early_stop:
            print(f"Early stopping triggered! No improvement for {PATIENCE} consecutive epochs.")
            break

    # 加载最佳模型用于评估
    early_stopping.load_checkpoint(model)
    print(f'Loaded best model with validation loss: {early_stopping.best_loss:.4f}')

    return history

# 评估函数
def evaluate_model():
    # 加载最佳模型
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()

    total_val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images).reshape(-1)
            labels = labels.reshape(-1)

            loss = criterion(outputs, labels)
            total_val_loss += loss.item() * images.size(0)

    avg_val_loss = total_val_loss / len(val_loader.dataset)
    print(f'Best model validation loss: {avg_val_loss:.4f}')

# 可视化训练结果
def plot_training_history(history):
    import matplotlib.pyplot as plt

    # 创建带有两个Y轴的图
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 第一个Y轴：损失
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(history['val_loss'], 'r-', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True)

    # 第二个Y轴：学习率
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate')
    ax2.plot(history['learning_rates'], 'g-', label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor='g')

    # 设置图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Training History')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, 'training_history.png'))
    plt.close()

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # 开始训练
    history = train_model()

    # 评估最佳模型
    evaluate_model()

    # 可视化训练历史（需要matplotlib）
    try:
        plot_training_history(history)
        print(f"Training history plot saved to {os.path.join(MODEL_SAVE_DIR, 'training_history.png')}")
    except Exception as e:
        print(f"Could not plot training history: {e}")
