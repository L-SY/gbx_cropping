import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score

# 设置路径
PROCESSED_DATA_DIR = '/home/siyang_liu/gbx_cropping_ws/train/hand_cropping/third/augmented_dataset'
MODEL_SAVE_PATH = '/home/siyang_liu/gbx_cropping_ws/train/xb/hand_cropping/first_second/best_model.pth'
OUTPUT_DIR = '/home/siyang_liu/gbx_cropping_ws/train/xb/hand_cropping/first_second/evaluate/2third'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DROPOUT_RATE = 0.5  # 保持与训练代码相同的Dropout率

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
test_dataset = MPPDataset(os.path.join(PROCESSED_DATA_DIR, 'all/all_labels.csv'),
                          os.path.join(PROCESSED_DATA_DIR, 'all'),
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
            outputs = model(images).reshape(-1).cpu().numpy()

            # 保证输出是可迭代的
            if outputs.ndim == 0:
                outputs = [outputs.item()]
            else:
                outputs = outputs.tolist()

            predictions.extend(outputs)
            true_labels.extend(labels.numpy().tolist())
            filenames.extend(files)  # 保存文件名

    # 计算评估指标
    mse = mean_squared_error(true_labels, predictions)
    r2 = r2_score(true_labels, predictions)
    mae = sum(abs(t - p) for t, p in zip(true_labels, predictions)) / len(true_labels)

    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'R² Score: {r2:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')

    # 绘制预测 vs 实际密度散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(true_labels, predictions, alpha=0.7)
    plt.xlabel('Actual Density')
    plt.ylabel('Predicted Density')
    plt.title('Actual vs Predicted Density')

    # 添加对角线（理想预测线）
    min_val = min(min(true_labels), min(predictions))
    max_val = max(max(true_labels), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')

    # 添加线性回归拟合线
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(true_labels, predictions)
    plt.plot(true_labels, [slope*x + intercept for x in true_labels], 'g-',
             label=f'Fit: y={slope:.4f}x+{intercept:.4f} (R²={r_value**2:.4f})')

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'density_prediction_scatter.png'), dpi=300)
    print(f'Scatter plot saved to {os.path.join(OUTPUT_DIR, "density_prediction_scatter.png")}')

    # 绘制预测误差直方图
    errors = [p - t for p, t in zip(predictions, true_labels)]
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, color='blue')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'error_distribution.png'), dpi=300)
    print(f'Error distribution plot saved to {os.path.join(OUTPUT_DIR, "error_distribution.png")}')

    # 保存预测详情到CSV以便排查
    results_df = pd.DataFrame({
        'filename': filenames,
        'actual_density': true_labels,
        'predicted_density': predictions,
        'error': errors,
        'abs_error': [abs(e) for e in errors]
    })

    results_df.to_csv(os.path.join(OUTPUT_DIR, 'prediction_details.csv'), index=False)
    print(f'Detailed predictions saved to {os.path.join(OUTPUT_DIR, "prediction_details.csv")}')

    # 找到误差最大的前10个点便于查看
    worst_predictions = results_df.sort_values('abs_error', ascending=False).head(10)
    print("\nWorst 10 predictions:")
    print(worst_predictions[['filename', 'actual_density', 'predicted_density', 'abs_error']])

    # 统计摘要
    print("\nError summary statistics:")
    error_stats = results_df['error'].describe()
    print(error_stats)

    # 将最差的预测结果可视化
    try:
        n_worst = 5  # 展示最差的5个预测
        plt.figure(figsize=(15, 10))

        for i, (_, row) in enumerate(worst_predictions.head(n_worst).iterrows()):
            img_path = os.path.join(PROCESSED_DATA_DIR, 'all', row['filename'])
            img = Image.open(img_path).convert('RGB')

            plt.subplot(1, n_worst, i+1)
            plt.imshow(img)
            plt.title(f"Act: {row['actual_density']:.2f}\nPred: {row['predicted_density']:.2f}")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'worst_predictions_images.png'), dpi=300)
        print(f'Images of worst predictions saved to {os.path.join(OUTPUT_DIR, "worst_predictions_images.png")}')
    except Exception as e:
        print(f"Could not create worst predictions visualization: {e}")

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Model path: {MODEL_SAVE_PATH}")
    print("Starting evaluation...")
    evaluate_model()
