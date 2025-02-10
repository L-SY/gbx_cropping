import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seed
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
np.random.seed(42)

class RegressionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Read label file
        labels_file = os.path.join(data_dir, f'{os.path.basename(data_dir)}_labels.csv')
        self.labels_df = pd.read_csv(labels_file)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.data_dir, row['image_name'])
        label = row['label']

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

class ImageRegressor(nn.Module):
    def __init__(self, backbone='resnet34', pretrained=True):
        super(ImageRegressor, self).__init__()
        # Load pretrained model
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        # Modify the final fully connected layer for regression
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

class ModelEvaluator:
    def __init__(self, model, device, test_loader):
        self.model = model.to(device)
        self.device = device
        self.test_loader = test_loader

    def compute_metrics(self):
        """Calculate multiple evaluation metrics"""
        self.model.eval()
        predictions = []
        true_values = []
        absolute_errors = []
        relative_errors = []

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Evaluating'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                # Collect predictions and true values
                pred_np = outputs.cpu().numpy()
                true_np = labels.cpu().numpy()

                predictions.extend(pred_np)
                true_values.extend(true_np)

                # Calculate errors
                abs_error = np.abs(pred_np - true_np)
                rel_error = abs_error / (true_np + 1e-10) * 100

                absolute_errors.extend(abs_error)
                relative_errors.extend(rel_error)

        predictions = np.array(predictions)
        true_values = np.array(true_values)
        absolute_errors = np.array(absolute_errors)
        relative_errors = np.array(relative_errors)

        # Calculate metrics
        metrics = {
            'MSE': np.mean((predictions - true_values) ** 2),
            'RMSE': np.sqrt(np.mean((predictions - true_values) ** 2)),
            'MAE': np.mean(absolute_errors),
            'MAPE': np.mean(relative_errors),
            'Max_Error': np.max(absolute_errors),
            'Min_Error': np.min(absolute_errors),
            'Std_Error': np.std(absolute_errors),
            'R2': self.r2_score(true_values, predictions)
        }

        return metrics, predictions, true_values

    @staticmethod
    def r2_score(y_true, y_pred):
        """Calculate R² score"""
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-10))

    def analyze_performance(self):
        """Analyze model performance and generate report"""
        metrics, predictions, true_values = self.compute_metrics()

        # Print metrics
        print("\n=== Model Performance Evaluation Report ===")
        print(f"Mean Squared Error (MSE): {metrics['MSE']:.4f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
        print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%")
        print(f"Maximum Error: {metrics['Max_Error']:.4f}")
        print(f"Minimum Error: {metrics['Min_Error']:.4f}")
        print(f"Standard Deviation of Error: {metrics['Std_Error']:.4f}")
        print(f"R² Score: {metrics['R2']:.4f}")

        # Plot detailed analysis
        self.plot_analysis(predictions, true_values, metrics)

        return metrics

    def plot_analysis(self, predictions, true_values, metrics):
        """Plot detailed analysis charts"""
        plt.figure(figsize=(15, 10))

        # 1. Predictions vs True Values scatter plot
        plt.subplot(2, 2, 1)
        plt.scatter(true_values, predictions, alpha=0.5)
        plt.plot([min(true_values), max(true_values)],
                 [min(true_values), max(true_values)],
                 'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Predictions vs True Values')

        # 2. Error distribution histogram
        plt.subplot(2, 2, 2)
        errors = predictions - true_values
        plt.hist(errors, bins=30)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')

        # 3. Bland-Altman plot
        plt.subplot(2, 2, 3)
        mean = (predictions + true_values) / 2
        diff = predictions - true_values
        plt.scatter(mean, diff, alpha=0.5)
        plt.axhline(y=np.mean(diff), color='r', linestyle='-')
        plt.axhline(y=np.mean(diff) + 1.96*np.std(diff), color='r', linestyle='--')
        plt.axhline(y=np.mean(diff) - 1.96*np.std(diff), color='r', linestyle='--')
        plt.xlabel('Mean (Prediction + True)/2')
        plt.ylabel('Difference (Prediction - True)')
        plt.title('Bland-Altman Plot')

        # 4. Relative error box plot
        plt.subplot(2, 2, 4)
        relative_errors = np.abs(predictions - true_values) / (true_values + 1e-10) * 100
        plt.boxplot(relative_errors)
        plt.ylabel('Relative Error (%)')
        plt.title('Relative Error Distribution')

        plt.tight_layout()
        try:
            plt.savefig('model_analysis.png', dpi=300, bbox_inches='tight')
            print("Analysis plots saved as 'model_analysis.png'")
        except Exception as e:
            print(f"Error saving plots: {str(e)}")
        finally:
            plt.close()

def evaluate_model():
    """Main function for model evaluation"""
    # Load model and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load trained model
    model = ImageRegressor(backbone='resnet34')
    try:
        checkpoint = torch.load('checkpoints/best_model.pth', weights_only=True)
        try:
            model.load_state_dict(checkpoint)
        except:
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise Exception("Unable to load model weights, checkpoint format incorrect")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, False

    # Prepare test data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = RegressionDataset(
        os.path.join("/home/lsy/gbx_cropping_ws/src/image_tools/dataset", 'test'),
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create evaluator and perform evaluation
    evaluator = ModelEvaluator(model, device, test_loader)
    metrics = evaluator.analyze_performance()

    # Check if model meets requirements
    is_model_acceptable = (
            metrics['R2'] > 0.9 and
            metrics['MAPE'] < 10.0 and
            metrics['Max_Error'] < 5.0
    )

    if is_model_acceptable:
        print("\nModel performance meets requirements!")
        print("\nDetailed Analysis:")
        print(f"1. R² Score {metrics['R2']:.4f} > 0.9: Strong correlation between predictions and true values")
        print(f"2. MAPE {metrics['MAPE']:.2f}% < 10%: Prediction errors within acceptable range")
        print(f"3. Max Error {metrics['Max_Error']:.4f} < 5.0: No severe prediction anomalies")
    else:
        print("\nModel performance needs improvement:")
        if metrics['R2'] <= 0.9:
            print(f"- R² Score {metrics['R2']:.4f} <= 0.9: Consider increasing training data or adjusting model structure")
        if metrics['MAPE'] >= 10.0:
            print(f"- MAPE {metrics['MAPE']:.2f}% >= 10%: Consider adjusting learning rate or increasing training epochs")
        if metrics['Max_Error'] >= 5.0:
            print(f"- Max Error {metrics['Max_Error']:.4f} >= 5.0: Consider using stronger data augmentation or regularization")

    return metrics, is_model_acceptable

if __name__ == "__main__":
    # Run evaluation
    metrics, is_acceptable = evaluate_model()