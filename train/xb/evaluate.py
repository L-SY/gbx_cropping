import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score

# 设置路径
PROCESSED_DATA_DIR = '/home/siyang_liu/gbx_cropping_ws/train/whole/augmented_dataset'
MODEL_SAVE_PATH = '/home/siyang_liu/gbx_cropping_ws/train/xb/whole_fifteen_epoch200/best_model.pth'
OUTPUT_DIR = '/home/siyang_liu/gbx_cropping_ws/train/xb/whole_fifteen_epoch20_test'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集定义 (增加返回文件名)
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

        return image, label, self.labels_df.iloc[idx]['image_name']  # 返回文件名以便追踪

# 数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据加载
test_dataset = MPPDataset(os.path.join(PROCESSED_DATA_DIR, 'test/test_labels.csv'),
                          os.path.join(PROCESSED_DATA_DIR, 'test'),
                          transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 模型定义
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(256, 1)
)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model = model.to(DEVICE)
model.eval()

# 评估函数（记录文件名、真实值和预测值）
def evaluate_model():
    predictions, true_labels, filenames = [], [], []

    with torch.no_grad():
        for images, labels, files in test_loader:
            images = images.to(DEVICE)
            outputs = model(images).squeeze().cpu().numpy()
            if outputs.ndim == 0:
                outputs = [outputs.item()]
            else:
                outputs = outputs.tolist()

            predictions.extend(outputs)
            true_labels.extend(labels.numpy().tolist())
            filenames.extend(files)  # 保存文件名

    mse = mean_squared_error(true_labels, predictions)
    r2 = r2_score(true_labels, predictions)

    print(f'MSE: {mse:.4f}')
    print(f'R^2 Score: {r2:.4f}')

    # 绘制预测 vs 实际密度散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(true_labels, predictions, alpha=0.7)
    plt.xlabel('Actual Density')
    plt.ylabel('Predicted Density')
    plt.title('Actual vs Predicted Density')
    plt.plot([min(true_labels), max(true_labels)], [min(true_labels), max(true_labels)], 'r--')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'density_prediction_plot2.png'))
    print(f'Plot saved to {os.path.join(OUTPUT_DIR, "density_prediction_plot2.png")}')

    # 保存预测详情到CSV以便排查
    results_df = pd.DataFrame({
        'filename': filenames,
        'actual_density': true_labels,
        'predicted_density': predictions,
        'abs_error': [abs(t - p) for t, p in zip(true_labels, predictions)]
    })

    results_df.to_csv(os.path.join(OUTPUT_DIR, 'prediction_details.csv'), index=False)
    print(f'Detailed predictions saved to {os.path.join(OUTPUT_DIR, "prediction_details.csv")}')

    # 找到误差最大的前5个点便于查看
    worst_predictions = results_df.sort_values('abs_error', ascending=False).head(5)
    print("\nWorst predictions:")
    print(worst_predictions)

if __name__ == '__main__':
    evaluate_model()