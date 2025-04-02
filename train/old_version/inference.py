# 导入所需的库
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import glob
from pathlib import Path

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 定义内部黑色边框添加类（与训练代码保持一致）
class InnerBlackBorderAdder(object):
    """Add black border inside the image"""
    def __init__(self, border_width=15):
        self.border_width = border_width

    def __call__(self, img):
        width, height = img.size
        bordered_img = img.copy()
        draw = ImageDraw.Draw(bordered_img)

        draw.rectangle([(0, 0), (width, self.border_width)], fill="black")
        draw.rectangle([(0, height - self.border_width), (width, height)], fill="black")
        draw.rectangle([(0, 0), (self.border_width, height)], fill="black")
        draw.rectangle([(width - self.border_width, 0), (width, height)], fill="black")

        return bordered_img

# 定义回归数据集类
class RegressionDataset(Dataset):
    """Custom dataset class"""
    def __init__(self, data_dir, labels_file=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # If a labels file is provided, read it
        if labels_file:
            if os.path.exists(labels_file):
                self.labels_df = pd.read_csv(labels_file)
            else:
                raise ValueError(f"Labels file not found: {labels_file}")
        else:
            # Try to find the subset name and read the corresponding labels file
            subset_name = os.path.basename(data_dir)  # 'train', 'val', or 'test'
            labels_file = os.path.join(data_dir, f'{subset_name}_labels.csv')

            if os.path.exists(labels_file):
                self.labels_df = pd.read_csv(labels_file)
            else:
                # If no labels file, assume we only need predictions
                self.labels_df = None
                # Get all image files in the directory
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(glob.glob(os.path.join(data_dir, f'*{ext}')))

                # Create a dataframe without labels
                self.labels_df = pd.DataFrame({
                    'image_name': [os.path.basename(f) for f in image_files],
                    'label': [float('nan')] * len(image_files)
                })

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Get image path and label
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.data_dir, row['image_name'])

        # Read image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Cannot read image {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='gray')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # If there's a label, return image and label; otherwise just return image
        if not pd.isna(row['label']):
            return image, torch.tensor(row['label'], dtype=torch.float32)
        else:
            return image, torch.tensor(float('nan'), dtype=torch.float32)

# 定义冻结CNN回归模型（与训练代码保持一致）
class FrozenCNNRegressor(nn.Module):
    """Texture regression model using frozen CNN feature extractor and trainable FC layers"""
    def __init__(self, backbone='densenet121', pretrained=True, initial_value=15.0, dropout_rate=0.5):
        super(FrozenCNNRegressor, self).__init__()

        # Load pretrained backbone network
        if backbone == 'densenet121':
            base_model = models.densenet121(weights='DEFAULT' if pretrained else None)
            self.features = base_model.features
            feature_dim = base_model.classifier.in_features  # 1024
        elif backbone == 'densenet169':
            base_model = models.densenet169(weights='DEFAULT' if pretrained else None)
            self.features = base_model.features
            feature_dim = base_model.classifier.in_features  # 1664
        elif backbone == 'resnet18':
            base_model = models.resnet18(weights='DEFAULT' if pretrained else None)
            # Remove global average pooling and fully connected layers
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            feature_dim = 512
        elif backbone == 'resnet34':
            base_model = models.resnet34(weights='DEFAULT' if pretrained else None)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            feature_dim = 512
        elif backbone == 'resnet50':
            base_model = models.resnet50(weights='DEFAULT' if pretrained else None)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            feature_dim = 2048
        elif backbone == 'mobilenet_v2':
            base_model = models.mobilenet_v2(weights='DEFAULT' if pretrained else None)
            self.features = base_model.features
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Regression head
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
        # Extract features
        features = self.features(x)

        # Global average pooling
        pooled = self.global_pool(features)

        # Regression prediction
        output = self.regressor(pooled).squeeze()

        return output

class ModelInference:
    """Model inference class"""
    def __init__(self, model_path, backbone='densenet121', device='cuda', border_width=70, output_dir='inference_results'):
        # 设置设备
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 保存输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 加载模型
        self.model = self.load_model(model_path, backbone)

        # 设置转换 - 确保与训练时完全一致
        # 注意：这里我们先添加黑色边框，再进行resize，然后转为tensor并进行标准化
        # 这与训练代码保持一致
        self.transform = transforms.Compose([
            InnerBlackBorderAdder(border_width=border_width),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print("Preprocessing pipeline initialized with:")
        print(f"  - Border width: {border_width}")
        print(f"  - Image size: 224x224")

    def load_model(self, model_path, backbone):
        """Load model"""
        try:
            print(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)

            # 创建模型实例
            model = FrozenCNNRegressor(
                backbone=backbone,
                pretrained=False,  # 不需要预训练权重，我们会加载自己训练的权重
                initial_value=15.0,
                dropout_rate=0.5
            )

            # 加载模型权重 - 处理不同的checkpoint格式
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded model state dict from checkpoint")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded entire model (not a checkpoint)")

            model.to(self.device)
            model.eval()  # 确保模型处于评估模式
            print(f"Model successfully loaded and set to evaluation mode")
            return model
        except Exception as e:
            raise Exception(f"Error loading model: {e}")

    def predict_single_image(self, image_path):
        """Predict a single image"""
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            print(f"Original image size: {original_size}")

            # 预处理图像 - 完全按照训练时的预处理流程
            input_tensor = self.transform(image)
            print(f"Transformed tensor shape: {input_tensor.shape}")

            # 添加批次维度并移至指定设备
            input_tensor = input_tensor.unsqueeze(0).to(self.device)

            # 执行推理
            with torch.no_grad():
                prediction = self.model(input_tensor).item()

            print(f"Raw prediction: {prediction}")
            return prediction
        except Exception as e:
            print(f"Error predicting image {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict_multiple_images(self, image_paths):
        """Predict multiple images"""
        predictions = []
        for path in tqdm(image_paths, desc="Predicting multiple images"):
            pred = self.predict_single_image(path)
            predictions.append({
                'image_path': path,
                'prediction': pred
            })

        return predictions

    def evaluate_dataset(self, data_dir, labels_file=None, batch_size=32, num_workers=4):
        """Evaluate entire dataset and identify samples with largest errors"""
        # Create dataset and dataloader
        dataset = RegressionDataset(data_dir, labels_file, transform=self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        print(f"Evaluating dataset with {len(dataset)} images")
        print(f"Using batch size: {batch_size}")

        # Store predictions and true labels
        all_predictions = []
        all_true_values = []
        image_names = []

        # Perform inference
        self.model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluating dataset")):
                images = images.to(self.device)

                # Get image names for current batch
                batch_indices = list(range(i * batch_size, min((i + 1) * batch_size, len(dataset))))
                batch_image_names = [dataset.labels_df.iloc[idx]['image_name'] for idx in batch_indices]

                # Predict
                outputs = self.model(images)

                # Store results
                all_predictions.extend(outputs.cpu().numpy())
                all_true_values.extend(labels.cpu().numpy())
                image_names.extend(batch_image_names)

        # Create results dataframe
        results_df = pd.DataFrame({
            'image_name': image_names,
            'prediction': all_predictions,
            'true_value': all_true_values
        })

        # Calculate metrics (if true labels exist)
        metrics = {}
        has_labels = not np.isnan(all_true_values).all()

        if has_labels:
            # Filter out NaN values
            valid_indices = ~np.isnan(all_true_values)
            valid_results_df = results_df[valid_indices].copy()

            # Calculate absolute errors
            valid_results_df['abs_error'] = np.abs(valid_results_df['prediction'] - valid_results_df['true_value'])
            valid_results_df['error'] = valid_results_df['prediction'] - valid_results_df['true_value']

            # Sort by absolute error (descending)
            valid_results_df = valid_results_df.sort_values('abs_error', ascending=False)

            # Extract top N samples with largest errors
            top_n_errors = 5
            largest_errors_df = valid_results_df.head(top_n_errors)

            print(f"\n===== Top {top_n_errors} Samples with Largest Errors =====")
            for idx, row in largest_errors_df.iterrows():
                print(f"Image: {row['image_name']}")
                print(f"  True Value: {row['true_value']:.2f}")
                print(f"  Prediction: {row['prediction']:.2f}")
                print(f"  Abs Error: {row['abs_error']:.2f}")
                print(f"  Error: {row['error']:.2f} ({'underestimated' if row['error'] < 0 else 'overestimated'})")
                print("-" * 50)

            # Save these images for further analysis
            error_analysis_dir = os.path.join(self.output_dir, 'error_analysis')
            os.makedirs(error_analysis_dir, exist_ok=True)

            # Copy the images with largest errors to the analysis directory
            for idx, row in largest_errors_df.iterrows():
                img_path = os.path.join(data_dir, row['image_name'])
                if os.path.exists(img_path):
                    # Load and save the image with error information in the filename
                    img = Image.open(img_path)
                    error_info = f"true_{row['true_value']:.2f}_pred_{row['prediction']:.2f}_err_{row['error']:.2f}"
                    save_path = os.path.join(error_analysis_dir, f"{os.path.splitext(row['image_name'])[0]}_{error_info}.png")
                    img.save(save_path)

                    # Also save a visualization with the error information
                    self.visualize_prediction(img, row['true_value'], row['prediction'], save_path.replace('.png', '_viz.png'))

            print(f"\nError analysis images saved to: {error_analysis_dir}")

            # Save detailed error analysis to CSV
            largest_errors_df.to_csv(os.path.join(error_analysis_dir, 'largest_errors.csv'), index=False)

            # Calculate standard metrics
            valid_preds = valid_results_df['prediction'].values
            valid_true = valid_results_df['true_value'].values

            metrics['mse'] = mean_squared_error(valid_true, valid_preds)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(valid_true, valid_preds)
            metrics['r2'] = r2_score(valid_true, valid_preds)
            metrics['mean_error'] = np.mean(valid_preds - valid_true)
            metrics['std_error'] = np.std(valid_preds - valid_true)
            metrics['max_error'] = np.max(np.abs(valid_preds - valid_true))
            metrics['min_error'] = np.min(np.abs(valid_preds - valid_true))

            # Calculate error distribution
            errors = valid_preds - valid_true
            metrics['error_percentiles'] = {
                '10%': np.percentile(np.abs(errors), 10),
                '25%': np.percentile(np.abs(errors), 25),
                '50%': np.percentile(np.abs(errors), 50),
                '75%': np.percentile(np.abs(errors), 75),
                '90%': np.percentile(np.abs(errors), 90),
                '95%': np.percentile(np.abs(errors), 95),
                '99%': np.percentile(np.abs(errors), 99)
            }

            # Calculate proportion of samples within different error ranges
            metrics['error_ranges'] = {
                '<0.5': np.mean(np.abs(errors) < 0.5) * 100,
                '<1.0': np.mean(np.abs(errors) < 1.0) * 100,
                '<2.0': np.mean(np.abs(errors) < 2.0) * 100,
                '<5.0': np.mean(np.abs(errors) < 5.0) * 100,
                '>5.0': np.mean(np.abs(errors) >= 5.0) * 100
            }

            # Visualize results
            self.visualize_evaluation_results(valid_true, valid_preds, metrics)

        return results_df, metrics

    def visualize_evaluation_results(self, true_values, predictions, metrics):
        """
        Visualize evaluation results with scatter plots and error histograms
        """
        # Create output directory for plots
        plots_dir = os.path.join(self.output_dir, 'evaluation_plots')
        os.makedirs(plots_dir, exist_ok=True)

        # 1. Scatter plot of predictions vs true values
        plt.figure(figsize=(10, 8))
        plt.scatter(true_values, predictions, alpha=0.6)

        # Add perfect prediction line
        min_val = min(min(true_values), min(predictions))
        max_val = max(max(true_values), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'Predictions vs True Values\nR² = {metrics["r2"]:.4f}, RMSE = {metrics["rmse"]:.4f}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'scatter_plot.png'), dpi=200)
        plt.close()

        # 2. Error histogram
        errors = predictions - true_values
        plt.figure(figsize=(10, 8))
        plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution\nMean Error = {metrics["mean_error"]:.4f}, Std Dev = {metrics["std_error"]:.4f}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'error_histogram.png'), dpi=200)
        plt.close()

        # 3. Error vs true value
        plt.figure(figsize=(10, 8))
        plt.scatter(true_values, errors, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('True Values')
        plt.ylabel('Prediction Error')
        plt.title('Error vs True Values')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'error_vs_true.png'), dpi=200)
        plt.close()

        # 4. Bland-Altman plot (agreement analysis)
        mean_values = (true_values + predictions) / 2
        plt.figure(figsize=(10, 8))
        plt.scatter(mean_values, errors, alpha=0.6)
        plt.axhline(y=metrics["mean_error"], color='r', linestyle='-', label=f'Mean error: {metrics["mean_error"]:.4f}')
        plt.axhline(y=metrics["mean_error"] + 1.96 * metrics["std_error"], color='g', linestyle='--',
                    label=f'+1.96 SD: {metrics["mean_error"] + 1.96 * metrics["std_error"]:.4f}')
        plt.axhline(y=metrics["mean_error"] - 1.96 * metrics["std_error"], color='g', linestyle='--',
                    label=f'-1.96 SD: {metrics["mean_error"] - 1.96 * metrics["std_error"]:.4f}')
        plt.xlabel('Mean of True and Predicted Values')
        plt.ylabel('Prediction Error')
        plt.title('Bland-Altman Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'bland_altman.png'), dpi=200)
        plt.close()

        # 5. Summary metrics table as an image
        plt.figure(figsize=(10, 6))
        plt.axis('off')

        # Create text for the table
        table_text = "Evaluation Metrics Summary\n\n"
        table_text += f"MSE: {metrics['mse']:.4f}\n"
        table_text += f"RMSE: {metrics['rmse']:.4f}\n"
        table_text += f"MAE: {metrics['mae']:.4f}\n"
        table_text += f"R²: {metrics['r2']:.4f}\n"
        table_text += f"Mean Error: {metrics['mean_error']:.4f}\n"
        table_text += f"Std Error: {metrics['std_error']:.4f}\n"
        table_text += f"Max Abs Error: {metrics['max_error']:.4f}\n"
        table_text += f"Min Abs Error: {metrics['min_error']:.4f}\n\n"

        table_text += "Error Percentiles:\n"
        for percentile, value in metrics['error_percentiles'].items():
            table_text += f"  {percentile}: {value:.4f}\n"

        table_text += "\nSamples within Error Ranges:\n"
        for range_name, percentage in metrics['error_ranges'].items():
            table_text += f"  {range_name}: {percentage:.2f}%\n"

        plt.text(0.1, 0.5, table_text, fontsize=12, va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'metrics_summary.png'), dpi=200)
        plt.close()

        print(f"Evaluation plots saved to: {plots_dir}")

    def visualize_prediction(self, image, true_value, prediction, save_path):
        """Visualize image with prediction and true value"""
        # Create a figure
        plt.figure(figsize=(10, 8))

        # Display the image
        plt.imshow(image)

        # Add text with prediction information
        error = prediction - true_value
        error_text = f"Error: {error:.2f} ({'underestimated' if error < 0 else 'overestimated'})"

        plt.title(f"True: {true_value:.2f} | Predicted: {prediction:.2f}\n{error_text}", fontsize=14)
        plt.axis('off')

        # Save the figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Texture Regression Model Inference')

    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model weights file')
    parser.add_argument('--backbone', type=str, default='densenet121',
                        choices=['densenet121', 'densenet169', 'resnet18', 'resnet34', 'resnet50', 'mobilenet_v2'],
                        help='Backbone network selection (default: densenet121)')
    parser.add_argument('--border_width', type=int, default=70,
                        help='Inner black border width (default: 70)')

    # Inference mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--evaluate_dataset', type=str,
                       help='Path to dataset directory for evaluation')
    group.add_argument('--predict_image', type=str,
                       help='Path to single image for prediction')
    group.add_argument('--predict_directory', type=str,
                       help='Path to directory containing multiple images for prediction')

    # Other parameters
    parser.add_argument('--labels_file', type=str, default=None,
                        help='Path to labels file (if different from default)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Output directory (default: inference_results)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Do not use CUDA even if available')

    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()

    # Set device
    device = 'cpu' if args.no_cuda else 'cuda'

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 打印系统信息
    print("=" * 50)
    print("System Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")
    print("=" * 50)

    # Initialize model inferencer
    inferencer = ModelInference(
        model_path=args.model_path,
        backbone=args.backbone,
        device=device,
        border_width=args.border_width,
        output_dir=args.output_dir
    )

    # Execute the appropriate inference task based on command line arguments
    if args.evaluate_dataset:
        print(f"Evaluating dataset: {args.evaluate_dataset}")
        results_df, metrics = inferencer.evaluate_dataset(
            data_dir=args.evaluate_dataset,
            labels_file=args.labels_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        # Save results
        results_df.to_csv(os.path.join(args.output_dir, 'evaluation_results.csv'), index=False)

        # Print metrics
        if metrics:
            print("\nEvaluation Metrics:")
            print(f"R²: {metrics['r2']:.4f}")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"MAE: {metrics['mae']:.4f}")
            print(f"Mean Error: {metrics['mean_error']:.4f}")
            print(f"Error Std Dev: {metrics['std_error']:.4f}")
            print(f"Max Error: {metrics['max_error']:.4f}")

            print("\nSample Proportion by Error Range:")
            for range_name, percentage in metrics['error_ranges'].items():
                print(f"  {range_name}: {percentage:.2f}%")

        print(f"\nDetailed results saved to: {os.path.join(args.output_dir, 'evaluation_results')}")

    elif args.predict_image:
        print(f"Predicting single image: {args.predict_image}")
        prediction = inferencer.predict_single_image(args.predict_image)

        print(f"Prediction result: {prediction:.4f}")

        # Save result
        with open(os.path.join(args.output_dir, 'single_prediction.txt'), 'w') as f:
            f.write(f"Image: {args.predict_image}\n")
            f.write(f"Predicted value: {prediction:.4f}\n")

    elif args.predict_directory:
        print(f"Predicting images in directory: {args.predict_directory}")

        # Get all image files in the directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(args.predict_directory, f'*{ext}')))

        print(f"Found {len(image_files)} image files")

        # Predict all images
        predictions = inferencer.predict_multiple_images(image_files)

        # Create results dataframe
        results_df = pd.DataFrame(predictions)

        # Save results
        results_df.to_csv(os.path.join(args.output_dir, 'batch_predictions.csv'), index=False)

        print(f"Prediction results saved to: {os.path.join(args.output_dir, 'batch_predictions.csv')}")

if __name__ == "__main__":
    main()
