import os
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# 模型定义函数（与训练脚本保持一致）
def get_model(model_name: str, dropout_rate: float):
    model_name = model_name.lower()
    model_func = getattr(models, model_name, None)
    if model_func is None:
        raise ValueError(f"Unsupported model: {model_name}")
    model = model_func(weights=None)
    if 'densenet' in model_name:
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )
    else:
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )
    return model

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
        return image, label, self.labels_df.iloc[idx]['image_name']

# 评估函数
def evaluate_model(model, dataloader, device, output_dir, data_root):
    model.eval()
    predictions, true_labels, filenames = [], [], []

    with torch.no_grad():
        for images, labels, files in dataloader:
            images = images.to(device)
            outputs = model(images).reshape(-1).cpu().numpy()
            outputs = outputs.tolist() if outputs.ndim > 0 else [outputs.item()]
            predictions.extend(outputs)
            true_labels.extend(labels.numpy().tolist())
            filenames.extend(files)

    # 指标计算
    mse = mean_squared_error(true_labels, predictions)
    r2 = r2_score(true_labels, predictions)
    mae = sum(abs(t - p) for t, p in zip(true_labels, predictions)) / len(true_labels)

    print(f'MSE: {mse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}')

    # 绘图目录准备
    os.makedirs(output_dir, exist_ok=True)

    # 预测 vs 实际
    plt.figure(figsize=(10, 8))
    plt.scatter(true_labels, predictions, alpha=0.7)
    plt.xlabel('Actual Density')
    plt.ylabel('Predicted Density')
    plt.title('Actual vs Predicted Density')
    min_val = min(min(true_labels), min(predictions))
    max_val = max(max(true_labels), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    slope, intercept, r_value, _, _ = stats.linregress(true_labels, predictions)
    plt.plot(true_labels, [slope * x + intercept for x in true_labels], 'g-',
             label=f'Fit: y={slope:.2f}x+{intercept:.2f} (R²={r_value**2:.2f})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'density_prediction_scatter.png'))

    # 误差分布图
    errors = [p - t for p, t in zip(predictions, true_labels)]
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, color='blue')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error (Pred - True)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))

    # 保存csv
    results_df = pd.DataFrame({
        'filename': filenames,
        'actual_density': true_labels,
        'predicted_density': predictions,
        'error': errors,
        'abs_error': [abs(e) for e in errors]
    })
    results_df.to_csv(os.path.join(output_dir, 'prediction_details.csv'), index=False)

    # 输出最差预测
    worst_predictions = results_df.sort_values('abs_error', ascending=False).head(10)
    print("\nWorst 10 predictions:")
    print(worst_predictions[['filename', 'actual_density', 'predicted_density', 'abs_error']])

    # 展示最差图像
    try:
        plt.figure(figsize=(15, 10))
        for i, (_, row) in enumerate(worst_predictions.head(5).iterrows()):
            img_path = os.path.join(data_root, row['filename'])
            img = Image.open(img_path).convert('RGB')
            plt.subplot(1, 5, i+1)
            plt.imshow(img)
            plt.title(f"A:{row['actual_density']:.1f}\nP:{row['predicted_density']:.1f}")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'worst_predictions_images.png'), dpi=300)
    except Exception as e:
        print(f"Failed to show worst prediction images: {e}")

# 主函数入口
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate regression model on test dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset (must contain all/ folder)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth model')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save evaluation outputs')
    parser.add_argument('--model', type=str, default='resnet50', help='Model name (resnet50, densenet121, etc.)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = MPPDataset(os.path.join(args.data_dir, 'test/test_labels.csv'),
                              os.path.join(args.data_dir, 'test'),
                              transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = get_model(args.model, args.dropout).to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    evaluate_model(model, test_loader, DEVICE, args.output_dir, os.path.join(args.data_dir, 'test'))
