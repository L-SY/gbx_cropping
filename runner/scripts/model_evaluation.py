# model_evaluator.py
import os
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# PyTorch相关导入
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

class InnerBlackBorderAdder(object):
    """在图像内部添加黑色边框"""
    def __init__(self, border_width=15):
        self.border_width = border_width

    def __call__(self, img):
        # 直接应用边框，不进行随机性检查
        width, height = img.size
        bordered_img = img.copy()
        draw = ImageDraw.Draw(bordered_img)

        draw.rectangle([(0, 0), (width, self.border_width)], fill="black")
        draw.rectangle([(0, height - self.border_width), (width, height)], fill="black")
        draw.rectangle([(0, 0), (self.border_width, height)], fill="black")
        draw.rectangle([(width - self.border_width, 0), (width, height)], fill="black")

        return bordered_img

class AdaptiveEdgeEnhancer(object):
    """Adaptive edge enhancer"""
    def __init__(self, alpha=1.5, beta=0.5, p=1.0):
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
    def __init__(self, clip_limit=3.0, tile_grid_size=(8, 8), p=1.0):
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

class RegressionDataset(Dataset):
    """Custom dataset class for regression tasks"""
    def __init__(self, data_dir, labels_file=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Read labels file
        if labels_file is None:
            # 尝试查找标签文件
            potential_files = [
                os.path.join(data_dir, 'labels.csv'),                    # 标准标签文件
                os.path.join(data_dir, f'{os.path.basename(data_dir)}_labels.csv')  # 以目录名命名的标签文件
            ]

            for file_path in potential_files:
                if os.path.exists(file_path):
                    self.labels_file = file_path
                    break
            else:
                raise FileNotFoundError(f"Cannot find labels file, paths searched: {potential_files}")
        else:
            self.labels_file = labels_file
            if not os.path.exists(self.labels_file):
                raise FileNotFoundError(f"Provided labels file does not exist: {self.labels_file}")

        print(f"Using labels file: {self.labels_file}")
        self.labels_df = pd.read_csv(self.labels_file)

        # 验证标签文件格式
        if 'image_name' not in self.labels_df.columns or 'label' not in self.labels_df.columns:
            raise ValueError(f"Invalid labels file format. Must contain 'image_name' and 'label' columns. Found: {self.labels_df.columns.tolist()}")

        # 验证图像文件是否存在
        self.valid_indices = []
        missing_files = []
        for idx, row in self.labels_df.iterrows():
            img_path = os.path.join(self.data_dir, row['image_name'])
            if os.path.exists(img_path):
                self.valid_indices.append(idx)
            else:
                missing_files.append(row['image_name'])

        if len(missing_files) > 0:
            print(f"Warning: Could not find {len(missing_files)} images, they will be excluded from the dataset.")
            if len(missing_files) <= 10:
                print(f"Missing files: {missing_files}")
            else:
                print(f"Sample of missing files: {missing_files[:10]}...")

        print(f"Loaded {len(self.valid_indices)} valid samples")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get image path and label from valid indices
        row = self.labels_df.iloc[self.valid_indices[idx]]
        img_path = os.path.join(self.data_dir, row['image_name'])
        label = row['label']

        # Read and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

class ModelEvaluator:
    """Model evaluator for pretrained regression models"""
    def __init__(self, model_path, data_dir, labels_file=None, output_dir=None, device=None,
                 backbone="densenet121", batch_size=32, initial_value=15.0, apply_preprocessing=True):

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Set output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"evaluation_results_{timestamp}"
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Evaluation results will be saved to: {self.output_dir}")

        # Load model
        self.model_path = model_path
        self.backbone = backbone
        self.initial_value = initial_value
        self.model = self._load_model()

        # Data-related parameters
        self.data_dir = data_dir
        self.labels_file = labels_file
        self.batch_size = batch_size
        self.apply_preprocessing = apply_preprocessing

        # Create data loader
        self.test_loader = self._create_data_loader()

        # Evaluation results
        self.metrics = {}
        self.predictions = None
        self.true_values = None

    def _load_model(self):
        """Load pretrained model"""
        try:
            print(f"Loading model from {self.model_path}...")

            # Create model instance
            model = FrozenCNNRegressor(
                backbone=self.backbone,
                pretrained=False,
                initial_value=self.initial_value
            )

            # Load weights
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Try different loading methods
            try:
                # Direct state dict loading
                model.load_state_dict(checkpoint)
                print("Successfully loaded model weights directly")
            except:
                # Try loading from a dict containing state_dict
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("Loaded model weights from model_state_dict")
                else:
                    # List top-level keys in checkpoint
                    print(f"Checkpoint contains the following keys: {list(checkpoint.keys())}")
                    raise ValueError("Could not recognize model weights in checkpoint")

            model.to(self.device)
            model.eval()  # Set to evaluation mode
            return model

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def _create_data_loader(self):
        """Create data loader"""
        print(f"Preparing dataset: {self.data_dir}")

        # Define transformations
        if self.apply_preprocessing:
            print("Applying image preprocessing enhancements")
            transforms_list = [
                # Preprocessing transforms
                InnerBlackBorderAdder(border_width=70),
                AdaptiveEdgeEnhancer(alpha=1.7, beta=0.4),
                ContrastTextureEnhancer(clip_limit=3.0, tile_grid_size=(8, 8)),
                # Basic transforms
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        else:
            print("Applying only basic transforms")
            transforms_list = [
                # Basic transforms
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]

        transform = transforms.Compose(transforms_list)

        # Create dataset
        dataset = RegressionDataset(
            data_dir=self.data_dir,
            labels_file=self.labels_file,
            transform=transform
        )

        # Create data loader
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        print(f"Created test dataset with {len(dataset)} samples")
        return loader

    def evaluate(self):
        """Evaluate model performance"""
        print("\nStarting model evaluation...")
        self.model.eval()

        all_predictions = []
        all_true_values = []
        all_absolute_errors = []

        total_time = 0
        total_images = 0

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluation progress"):
                images, labels = images.to(self.device), labels.to(self.device)

                # Record inference time
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                outputs = self.model(images)
                end_time.record()

                # Wait for GPU operations to complete
                torch.cuda.synchronize()

                # Calculate batch inference time (ms to seconds)
                batch_time = start_time.elapsed_time(end_time) / 1000
                total_time += batch_time
                total_images += images.size(0)

                # Collect predictions and true values
                predictions = outputs.cpu().numpy()
                true_values = labels.cpu().numpy()

                all_predictions.extend(predictions)
                all_true_values.extend(true_values)

                # Calculate absolute errors
                abs_errors = np.abs(predictions - true_values)
                all_absolute_errors.extend(abs_errors)

        # Convert to NumPy arrays for calculations
        all_predictions = np.array(all_predictions)
        all_true_values = np.array(all_true_values)
        all_absolute_errors = np.array(all_absolute_errors)

        # Save for further analysis
        self.predictions = all_predictions
        self.true_values = all_true_values

        # Calculate evaluation metrics
        mse = mean_squared_error(all_true_values, all_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_true_values, all_predictions)
        r2 = r2_score(all_true_values, all_predictions)

        # Calculate additional metrics
        mape = np.mean(np.abs((all_true_values - all_predictions) / all_true_values)) * 100
        max_error = np.max(all_absolute_errors)
        min_error = np.min(all_absolute_errors)
        std_error = np.std(all_absolute_errors)
        median_error = np.median(all_absolute_errors)
        q1_error = np.percentile(all_absolute_errors, 25)
        q3_error = np.percentile(all_absolute_errors, 75)

        # Calculate average inference time
        avg_time_per_image = total_time / total_images

        # Summarize metrics
        self.metrics = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape),
            'R2': float(r2),
            'Max_Error': float(max_error),
            'Min_Error': float(min_error),
            'Median_Error': float(median_error),
            'Q1_Error': float(q1_error),
            'Q3_Error': float(q3_error),
            'Std_Error': float(std_error),
            'Avg_Inference_Time': float(avg_time_per_image),
            'Total_Samples': int(len(all_true_values)),
            'Total_Time': float(total_time),
            'Dataset_Path': self.data_dir,
            'Model_Path': self.model_path,
            'Evaluation_Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return self.metrics

    def print_metrics(self):
        """Print evaluation metrics"""
        if not self.metrics:
            print("Please run evaluate() method first")
            return

        print("\n======== Model Evaluation Results ========")
        print(f"Dataset: {self.data_dir}")
        print(f"Sample count: {self.metrics['Total_Samples']}")
        print(f"Evaluation time: {self.metrics['Evaluation_Time']}")
        print("\n----- Accuracy Metrics -----")
        print(f"Mean Squared Error (MSE): {self.metrics['MSE']:.6f}")
        print(f"Root Mean Squared Error (RMSE): {self.metrics['RMSE']:.6f}")
        print(f"Mean Absolute Error (MAE): {self.metrics['MAE']:.6f}")
        print(f"Mean Absolute Percentage Error (MAPE): {self.metrics['MAPE']:.2f}%")
        print(f"Coefficient of Determination (R²): {self.metrics['R2']:.6f}")
        print("\n----- Error Distribution -----")
        print(f"Maximum Error: {self.metrics['Max_Error']:.6f}")
        print(f"Minimum Error: {self.metrics['Min_Error']:.6f}")
        print(f"Median Error: {self.metrics['Median_Error']:.6f}")
        print(f"First Quartile Error: {self.metrics['Q1_Error']:.6f}")
        print(f"Third Quartile Error: {self.metrics['Q3_Error']:.6f}")
        print(f"Error Standard Deviation: {self.metrics['Std_Error']:.6f}")
        print("\n----- Performance Metrics -----")
        print(f"Average Inference Time: {self.metrics['Avg_Inference_Time']*1000:.2f} ms/image")
        print(f"Total Inference Time: {self.metrics['Total_Time']:.2f} seconds")
        print("============================")

        # Judge if model performance is acceptable
        print("\n----- Model Quality Assessment -----")
        if self.metrics['R2'] > 0.9:
            print("✓ R² > 0.9: Model has strong predictive power")
        else:
            print(f"✗ R² = {self.metrics['R2']:.4f} < 0.9: Model has weaker predictive power")

        if self.metrics['MAPE'] < 10.0:
            print("✓ MAPE < 10%: Prediction errors are within acceptable range")
        else:
            print(f"✗ MAPE = {self.metrics['MAPE']:.2f}% > 10%: Prediction relative errors are significant")

        if self.metrics['Max_Error'] < 5.0:
            print("✓ Max Error < 5.0: No severe prediction anomalies")
        else:
            print(f"✗ Max Error = {self.metrics['Max_Error']:.4f} > 5.0: Contains significant prediction anomalies")

    def generate_visualizations(self):
        """Generate visualization charts"""
        if self.predictions is None or self.true_values is None:
            print("Please run evaluate() method first")
            return

        print("\nGenerating visualization charts...")

        # 1. True vs Predicted scatter compute
        self._plot_true_vs_predicted()

        # 2. Error distribution histogram
        self._plot_error_distribution()

        # 3. Bland-Altman compute (difference compute)
        self._plot_bland_altman()

        # 4. Error boxplot
        self._plot_error_boxplot()

        # 5. Comprehensive analysis
        self._plot_comprehensive_analysis()

        # 6. Error vs True compute
        self._plot_error_vs_true()

        print(f"All visualization charts have been saved to: {self.output_dir}")

    def save_results(self):
        """Save evaluation results"""
        if not self.metrics:
            print("Please run evaluate() method first")
            return

        print("\nSaving evaluation results...")

        # Save metrics as JSON
        metrics_file = os.path.join(self.output_dir, 'evaluation_metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=4, ensure_ascii=False)

        # Save prediction results as CSV
        if self.predictions is not None and self.true_values is not None:
            results_df = pd.DataFrame({
                'true_value': self.true_values,
                'predicted_value': self.predictions,
                'absolute_error': np.abs(self.predictions - self.true_values),
                'relative_error': np.abs((self.predictions - self.true_values) / self.true_values) * 100
            })

            results_file = os.path.join(self.output_dir, 'prediction_results.csv')
            results_df.to_csv(results_file, index=False)

            # Find samples with largest errors
            top_errors = results_df.nlargest(10, 'absolute_error')
            top_errors_file = os.path.join(self.output_dir, 'top_10_errors.csv')
            top_errors.to_csv(top_errors_file, index=False)

            print(f"Evaluation metrics saved to: {metrics_file}")
            print(f"Prediction results saved to: {results_file}")
            print(f"Top 10 largest errors saved to: {top_errors_file}")

    def _plot_true_vs_predicted(self):
        """Plot true vs predicted values scatter compute"""
        plt.figure(figsize=(10, 8))

        # Scatter compute
        plt.scatter(self.true_values, self.predictions, alpha=0.6, s=30)

        # Ideal line (diagonal)
        min_val = min(np.min(self.true_values), np.min(self.predictions))
        max_val = max(np.max(self.true_values), np.max(self.predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Line')

        # Fit line
        z = np.polyfit(self.true_values, self.predictions, 1)
        p = np.poly1d(z)
        plt.plot(self.true_values, p(self.true_values), 'g-', lw=1.5, alpha=0.7, label=f'Fit Line (y={z[0]:.4f}x+{z[1]:.4f})')

        # Set title and labels
        plt.title(f'Predicted vs True Values\nR² = {self.metrics["R2"]:.4f}, RMSE = {self.metrics["RMSE"]:.4f}')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add text information
        text = f'Sample count: {len(self.true_values)}\n'
        text += f'MAE: {self.metrics["MAE"]:.4f}\n'
        text += f'MAPE: {self.metrics["MAPE"]:.2f}%'
        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Save chart
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'true_vs_predicted.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_error_distribution(self):
        """Plot error distribution histogram"""
        plt.figure(figsize=(10, 8))

        # Calculate errors
        errors = self.predictions - self.true_values

        # Draw histogram
        counts, bins, patches = plt.hist(errors, bins=30, alpha=0.7, color='blue', density=True)

        # Add kernel density estimation
        kde_x = np.linspace(min(errors), max(errors), 1000)
        kde = np.zeros_like(kde_x)

        bandwidth = 0.5  # Bandwidth parameter, can be adjusted
        for e in errors:
            kde += np.exp(-0.5 * ((kde_x - e) / bandwidth)**2) / (bandwidth * np.sqrt(2 * np.pi))
        kde /= len(errors)

        plt.plot(kde_x, kde, 'r-', lw=2, label='Kernel Density Estimation')

        # Add vertical lines for mean and median
        plt.axvline(x=np.mean(errors), color='g', linestyle='--', lw=2, label=f'Mean: {np.mean(errors):.4f}')
        plt.axvline(x=np.median(errors), color='m', linestyle='--', lw=2, label=f'Median: {np.median(errors):.4f}')

        # Set title and labels
        plt.title('Prediction Error Distribution')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency Density')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add text information
        text = f'Sample count: {len(errors)}\n'
        text += f'Mean: {np.mean(errors):.4f}\n'
        text += f'Std Dev: {np.std(errors):.4f}\n'
        text += f'Skewness: {float(pd.Series(errors).skew()):.4f}\n'
        text += f'Kurtosis: {float(pd.Series(errors).kurtosis()):.4f}'
        plt.text(0.95, 0.95, text, transform=plt.gca().transAxes,
                 horizontalalignment='right', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Save chart
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_bland_altman(self):
        """Plot Bland-Altman compute (difference compute)"""
        plt.figure(figsize=(10, 8))

        # Calculate means and differences
        mean = (self.predictions + self.true_values) / 2
        diff = self.predictions - self.true_values

        # Calculate mean difference and standard deviation
        md = np.mean(diff)
        sd = np.std(diff)

        # Draw scatter compute
        plt.scatter(mean, diff, alpha=0.6, s=30)

        # Add mean line and 1.96 standard deviation interval
        plt.axhline(md, color='r', linestyle='-', lw=2, label=f'Mean Bias: {md:.4f}')
        plt.axhline(md + 1.96*sd, color='r', linestyle='--', lw=2, label=f'Upper Limit (+1.96SD): {md + 1.96*sd:.4f}')
        plt.axhline(md - 1.96*sd, color='r', linestyle='--', lw=2, label=f'Lower Limit (-1.96SD): {md - 1.96*sd:.4f}')

        # Set title and labels
        plt.title('Bland-Altman Plot (Difference Plot)')
        plt.xlabel('Mean ((Predicted + True)/2)')
        plt.ylabel('Difference (Predicted - True)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add text information
        percent_within = 100 * np.sum((diff >= md - 1.96*sd) & (diff <= md + 1.96*sd)) / len(diff)
        text = f'Sample count: {len(diff)}\n'
        text += f'Mean bias: {md:.4f}\n'
        text += f'SD of bias: {sd:.4f}\n'
        text += f'Samples within 95% CI: {percent_within:.1f}%'
        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Save chart
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'bland_altman.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_error_boxplot(self):
        """Plot error boxplot"""
        plt.figure(figsize=(10, 6))

        # Calculate absolute and relative errors
        abs_errors = np.abs(self.predictions - self.true_values)
        rel_errors = np.abs((self.predictions - self.true_values) / self.true_values) * 100

        # Create boxplot data
        data = [abs_errors, rel_errors]
        labels = ['Absolute Error', 'Relative Error (%)']

        # Draw boxplot
        box = plt.boxplot(data, labels=labels, patch_artist=True,
                          widths=0.6, showmeans=True, meanline=True)

        # Set colors
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        # Set title and labels
        plt.title('Error Distribution Boxplot')
        plt.ylabel('Error Values')
        plt.grid(True, axis='y', alpha=0.3)

        # Add text information
        for i, (errors, label) in enumerate(zip(data, labels)):
            x_pos = i + 1  # Boxplot position
            text = f'Median: {np.median(errors):.4f}\n'
            text += f'Mean: {np.mean(errors):.4f}\n'
            text += f'Q1: {np.percentile(errors, 25):.4f}\n'
            text += f'Q3: {np.percentile(errors, 75):.4f}\n'
            text += f'Max: {np.max(errors):.4f}'

            # Add text next to the boxplot
            plt.text(x_pos + 0.3, np.median(errors), text,
                     verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Save chart
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'error_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_comprehensive_analysis(self):
        """Plot comprehensive analysis chart"""
        plt.figure(figsize=(16, 12))

        # 1. True vs Predicted scatter compute
        plt.subplot(2, 2, 1)
        plt.scatter(self.true_values, self.predictions, alpha=0.6, s=30)
        min_val = min(np.min(self.true_values), np.min(self.predictions))
        max_val = max(np.max(self.true_values), np.max(self.predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        z = np.polyfit(self.true_values, self.predictions, 1)
        p = np.poly1d(z)
        plt.plot(self.true_values, p(self.true_values), 'g-', lw=1.5, alpha=0.7)
        plt.title(f'Predicted vs True Values\nR² = {self.metrics["R2"]:.4f}')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.grid(True, alpha=0.3)

        # 2. Error distribution histogram
        plt.subplot(2, 2, 2)
        errors = self.predictions - self.true_values
        plt.hist(errors, bins=30, alpha=0.7)
        plt.axvline(x=np.mean(errors), color='r', linestyle='--', lw=2)
        plt.title(f'Error Distribution\nMean = {np.mean(errors):.4f}, SD = {np.std(errors):.4f}')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        # 3. Bland-Altman compute
        plt.subplot(2, 2, 3)
        mean = (self.predictions + self.true_values) / 2
        diff = self.predictions - self.true_values
        md = np.mean(diff)
        sd = np.std(diff)
        plt.scatter(mean, diff, alpha=0.6, s=30)
        plt.axhline(md, color='r', linestyle='-', lw=2)
        plt.axhline(md + 1.96*sd, color='r', linestyle='--', lw=2)
        plt.axhline(md - 1.96*sd, color='r', linestyle='--', lw=2)
        plt.title('Bland-Altman Plot (Difference Plot)')
        plt.xlabel('Mean ((Predicted + True)/2)')
        plt.ylabel('Difference (Predicted - True)')
        plt.grid(True, alpha=0.3)

        # 4. Relative error vs true value compute
        plt.subplot(2, 2, 4)
        rel_errors = np.abs((self.predictions - self.true_values) / self.true_values) * 100
        plt.scatter(self.true_values, rel_errors, alpha=0.6, s=30)
        plt.axhline(y=5, color='g', linestyle='--', lw=2, label='5% Error')
        plt.axhline(y=10, color='orange', linestyle='--', lw=2, label='10% Error')
        plt.axhline(y=20, color='r', linestyle='--', lw=2, label='20% Error')
        plt.title(f'Relative Error vs True Values\nMAPE = {self.metrics["MAPE"]:.2f}%')
        plt.xlabel('True Values')
        plt.ylabel('Relative Error (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save chart
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_error_vs_true(self):
        """Plot error vs true value relationship"""
        plt.figure(figsize=(10, 8))

        # Calculate errors
        errors = self.predictions - self.true_values

        # Draw scatter compute
        plt.scatter(self.true_values, errors, alpha=0.6, s=30)

        # Add zero error line
        plt.axhline(y=0, color='r', linestyle='-', lw=2)

        # Add ±5% error interval lines
        for percent in [5, 10]:
            upper_line = self.true_values * percent / 100
            lower_line = -self.true_values * percent / 100
            plt.plot(self.true_values, upper_line, 'g--', alpha=0.7, label=f'+{percent}%' if percent==5 else None)
            plt.plot(self.true_values, lower_line, 'g--', alpha=0.7, label=f'-{percent}%' if percent==5 else None)

        # Fit line
        z = np.polyfit(self.true_values, errors, 1)
        p = np.poly1d(z)
        plt.plot(self.true_values, p(self.true_values), 'b-', lw=1.5, alpha=0.7,
                 label=f'Trend Line (y={z[0]:.4f}x+{z[1]:.4f})')

        # Set title and labels
        plt.title('Prediction Error vs True Values')
        plt.xlabel('True Values')
        plt.ylabel('Prediction Error')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add text information
        in_5_percent = 100 * np.sum(np.abs(errors) <= 0.05 * np.abs(self.true_values)) / len(errors)
        in_10_percent = 100 * np.sum(np.abs(errors) <= 0.10 * np.abs(self.true_values)) / len(errors)
        text = f'Sample count: {len(errors)}\n'
        text += f'Errors within ±5%: {in_5_percent:.1f}%\n'
        text += f'Errors within ±10%: {in_10_percent:.1f}%\n'

        if abs(z[0]) > 0.05:  # If slope is significant
            text += f'Warning: Errors show correlation with true values ({z[0]:.4f}),\nsuggesting potential bias in the model'

        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Save chart
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'error_vs_true.png'), dpi=300, bbox_inches='tight')
        plt.close()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate pretrained regression model')

    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to pretrained model file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to test dataset directory')

    # Optional arguments
    parser.add_argument('--labels_file', type=str, default=None,
                        help='Path to labels file (default: auto-detect)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to output directory (default: auto-generate)')
    parser.add_argument('--backbone', type=str, default='densenet121',
                        choices=['densenet121', 'densenet169', 'resnet18', 'resnet34', 'resnet50', 'mobilenet_v2'],
                        help='Backbone network type (default: densenet121)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--initial_value', type=float, default=15.0,
                        help='Initial output value (default: 15.0)')
    parser.add_argument('--no_preprocessing', action='store_true',
                        help='Disable image preprocessing enhancements')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU evaluation (default: use GPU if available)')

    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()

    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        data_dir=args.data_dir,
        labels_file=args.labels_file,
        output_dir=args.output_dir,
        device='cpu' if args.cpu else 'cuda',
        backbone=args.backbone,
        batch_size=args.batch_size,
        initial_value=args.initial_value,
        apply_preprocessing=not args.no_preprocessing
    )

    # Perform evaluation
    evaluator.evaluate()

    # Print evaluation metrics
    evaluator.print_metrics()

    # Generate visualization charts
    evaluator.generate_visualizations()

    # Save evaluation results
    evaluator.save_results()

    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
