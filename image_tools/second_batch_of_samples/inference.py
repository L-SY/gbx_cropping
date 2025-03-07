#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch Image Texture Analysis Tool
Usage: python inference.py --model_path MODEL_PATH --image_dir IMAGE_DIR [--output_dir OUTPUT_DIR] [--device DEVICE] [--csv_path CSV_PATH]
"""

import os
import sys
import argparse
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
from tqdm import tqdm
import glob
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define model class - Modified to match the saved model structure
class FrozenCNNRegressor(nn.Module):
    """Texture regression model using frozen CNN feature extractor and trainable FC layers"""
    def __init__(self, backbone='densenet121', pretrained=True, initial_value=15.0):
        super(FrozenCNNRegressor, self).__init__()

        # Load pre-trained backbone network
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
            # Remove global avg pooling and fc layers
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

        # Regression head with L2 regularization effect (similar to Ridge regression) - Modified to match saved model structure
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),  # Changed to 128
            nn.BatchNorm1d(128),  # Changed to 128
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),   # Added layer
            nn.BatchNorm1d(64),   # Added layer
            nn.ReLU(),            # Added layer
            nn.Dropout(0.2),      # Added layer
            nn.Linear(64, 1)      # Final output layer
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

def load_model(model_path, device='cuda'):
    """Load pretrained model - Using a safer approach"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    try:
        # First try to load model weights only - safer approach
        try:
            print("Attempting to load model in safe mode (weights_only=True)...")
            # Create empty model
            model = FrozenCNNRegressor(backbone='densenet121', pretrained=False)

            # Try to load in safe mode with weights_only=True
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)

            # If it's a state dict, load directly
            if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
                model.load_state_dict(checkpoint, strict=False)  # Use strict=False to allow partial loading
            else:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            print("Model loaded successfully in safe mode")

        except Exception as e:
            print(f"Safe mode loading failed, trying compatibility mode: {e}")
            # If safe mode fails, use compatibility mode (not recommended, but ensures old models can be loaded)
            print("WARNING: Using unsafe loading method, ensure you trust this model file")
            checkpoint = torch.load(model_path, map_location=device)

            # If it's a complete model rather than a dict
            if not isinstance(checkpoint, dict):
                model = checkpoint
            # If it's a dict containing state_dict
            elif 'model_state_dict' in checkpoint:
                model = FrozenCNNRegressor(backbone='densenet121', pretrained=False)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                # If it's a direct state_dict
                model = FrozenCNNRegressor(backbone='densenet121', pretrained=False)
                model.load_state_dict(checkpoint, strict=False)

        model = model.to(device)
        model.eval()
        print(f"Model successfully loaded: {model_path}")
        return model

    except Exception as e:
        print(f"Model loading failed: {e}")
        sys.exit(1)

def get_transform():
    """Get image transformations"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def predict_single_image(model, image_path, transform, device):
    """Predict for a single image"""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Save original size for later visualization
        original_size = image.size

        # Apply transformations
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            prediction = model(input_tensor).item()

        return {
            'path': image_path,
            'prediction': prediction,
            'image': image,
            'original_size': original_size,
            'status': 'success'
        }
    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")
        return {
            'path': image_path,
            'prediction': None,
            'status': 'failed',
            'error': str(e)
        }

def generate_heatmap(prediction, min_val=0, max_val=30):
    """Generate heatmap color based on prediction value"""
    # Create custom colormap, from blue (low) to red (high)
    colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)

    # Normalize prediction to 0-1 range
    normalized = (prediction - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0, 1)  # Ensure it's in 0-1 range

    # Get RGB color
    rgb_color = cmap(normalized)[:3]  # Remove alpha channel

    # Convert RGB to 0-255 integers
    color_8bit = tuple(int(c * 255) for c in rgb_color)

    return color_8bit, normalized

def create_visualization(result, output_path, global_min=0, global_max=30, reference_value=None):
    """Create result visualization"""
    if result['status'] != 'success':
        return

    image = result['image']
    prediction = result['prediction']

    # Calculate color using global min and max
    color, normalized = generate_heatmap(prediction, global_min, global_max)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Show original image
    ax.imshow(image)

    # Add prediction value text with color, make it visible on any background
    if reference_value is not None:
        error = prediction - reference_value
        text = f"Predicted: {prediction:.2f} | Reference: {reference_value:.2f} | Error: {error:.2f}"
    else:
        text = f"Predicted: {prediction:.2f}"

    ax.text(10, 30, text, fontsize=16, weight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'))

    # Add colorbar indicator
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,
                               norm=plt.Normalize(vmin=global_min, vmax=global_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Prediction Range')

    # Mark current value on the colorbar
    cbar.ax.plot([0, 1], [prediction, prediction], 'k-', linewidth=2)

    # Turn off axes
    ax.axis('off')

    # Set title
    filename = os.path.basename(result['path'])
    ax.set_title(f"File: {filename} - Prediction: {prediction:.2f}", fontsize=14)

    # Save image
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def create_gallery(results, output_dir, rows=5, global_min=None, global_max=None, reference_values=None, image_id_map=None):
    """Create image gallery"""
    # Skip failed images
    valid_results = [r for r in results if r['status'] == 'success']

    if not valid_results:
        print("No valid results to generate gallery")
        return

    # If global range not specified, calculate from data
    if global_min is None:
        global_min = min(r['prediction'] for r in valid_results)
    if global_max is None:
        global_max = max(r['prediction'] for r in valid_results)

    # Sort by prediction value
    valid_results.sort(key=lambda x: x['prediction'])

    # Calculate layout
    n_images = len(valid_results)
    cols = (n_images + rows - 1) // rows  # Ceiling division

    # Create canvas
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    if n_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Add each image
    for i, result in enumerate(valid_results):
        if i < len(axes):
            ax = axes[i]
            ax.imshow(result['image'])

            # Add prediction value text
            pred = result['prediction']
            filename = os.path.basename(result['path'])

            # Check if we have a reference value for this image using image_id_map
            ref_value = None
            if image_id_map is not None and result['path'] in image_id_map:
                id_val = image_id_map[result['path']]
                if id_val in reference_values:
                    ref_value = reference_values[id_val]

            if ref_value is not None:
                error = pred - ref_value
                title = f"{filename}\nPred: {pred:.2f} | Ref: {ref_value:.2f} | Err: {error:.2f}"
            else:
                title = f"{filename}\nPrediction: {pred:.2f}"

            ax.set_title(title, fontsize=10)

            # Set border color reflecting prediction value
            color, _ = generate_heatmap(pred, global_min, global_max)
            for spine in ax.spines.values():
                spine.set_color(np.array(color)/255)
                spine.set_linewidth(5)

            # Turn off axes
            ax.axis('off')

    # Hide extra subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    # Adjust layout
    plt.tight_layout()

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,
                               norm=plt.Normalize(vmin=global_min, vmax=global_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                        pad=0.01, fraction=0.05, aspect=40)
    cbar.set_label('Prediction Range')

    # Add title
    fig.suptitle(f'Batch Prediction Results Overview - {n_images} Images', fontsize=16, y=1.02)

    # Save gallery
    gallery_path = os.path.join(output_dir, 'results_gallery.png')
    plt.savefig(gallery_path, dpi=150, bbox_inches='tight')
    print(f"Gallery saved to: {gallery_path}")
    plt.close(fig)

def extract_id_from_filename(filename):
    """Extract numeric ID from filename like 'cropped_raw_1.jpg'"""
    # Pattern to match numbers at the end of the filename before extension
    match = re.search(r'(\d+)(?:\.[^.]+)?$', filename)
    if match:
        return match.group(1)
    return None

def process_image_folder(model_path, image_dir, output_dir=None, device='cuda', n_workers=4, csv_path=None):
    """Process entire image folder"""
    start_time = time.time()

    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(image_dir), 'inference_results')

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)

    # Load reference values from CSV if provided
    reference_values = None
    if csv_path and os.path.exists(csv_path):
        try:
            print(f"Loading reference values from: {csv_path}")
            csv_data = pd.read_csv(csv_path)

            # Print out the columns in the CSV file for debugging
            print(f"CSV columns: {list(csv_data.columns)}")

            # Check for ID column
            id_col = None
            for col_name in ['ID', 'Id', 'id', 'sample_id', 'sampleid', 'sample', 'SampleId']:
                if col_name in csv_data.columns:
                    id_col = col_name
                    print(f"Using '{id_col}' as ID column")
                    break

            # Check for rate/value column
            rate_col = None
            for col_name in ['ComputedRate', 'computedrate', 'computed_rate', 'rate', 'density', 'value', 'prediction', 'target', 'label', 'Rate']:
                if col_name in csv_data.columns:
                    rate_col = col_name
                    print(f"Using '{rate_col}' as reference value column")
                    break

            # Create a dictionary of ID -> reference value
            if id_col is not None and rate_col is not None:
                reference_values = {}

                for _, row in csv_data.iterrows():
                    if pd.notna(row[id_col]) and pd.notna(row[rate_col]):
                        # Convert ID to string and remove decimal part if it's like "1.0"
                        id_raw = row[id_col]
                        if isinstance(id_raw, (int, float)):
                            id_value = str(int(id_raw) if id_raw == int(id_raw) else id_raw)
                        else:
                            id_value = str(id_raw)
                            if id_value.endswith('.0'):
                                id_value = id_value[:-2]

                        rate_value = float(row[rate_col])
                        reference_values[id_value] = rate_value

                print(f"Loaded {len(reference_values)} reference values from CSV")
                print(f"Example ID to rate mapping: {list(reference_values.items())[:3]}")
            else:
                print("Could not find suitable columns for ID and reference values")
                print("Required: a column for ID and a column for reference values")
                print("Available columns:", list(csv_data.columns))
        except Exception as e:
            print(f"Error loading reference values from CSV: {e}")
            reference_values = None

    # Load model
    model = load_model(model_path, device)
    transform = get_transform()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Get all image files
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))

    image_files = sorted(list(set(image_files)))  # Remove duplicates and sort

    if not image_files:
        print(f"No image files found in {image_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} image files")

    # Create image to ID mapping
    image_id_map = {}
    if reference_values:
        print("Sample image filenames:")
        for img_path in image_files[:5]:  # Print first 5 image filenames
            print(f"  - {os.path.basename(img_path)}")

        print("Sample IDs from CSV:")
        sample_ids = list(reference_values.keys())[:5]  # First 5 IDs
        for id_val in sample_ids:
            print(f"  - {id_val}")

        print("Extracting numeric IDs from filenames...")

        matched_count = 0
        for img_path in image_files:
            filename = os.path.basename(img_path)
            extracted_id = extract_id_from_filename(filename)

            if extracted_id and extracted_id in reference_values:
                image_id_map[img_path] = extracted_id
                matched_count += 1
                if matched_count <= 5:  # Print first 5 matches
                    print(f"✓ Matched: {filename} with ID {extracted_id} → Rate {reference_values[extracted_id]}")

        print(f"Matched {matched_count} out of {len(image_files)} images with reference values")

    # Batch predictions
    results = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Create tasks
        future_to_path = {
            executor.submit(
                predict_single_image, model, path, transform, device
            ): path for path in image_files
        }

        # Process results
        for future in tqdm(as_completed(future_to_path), total=len(image_files), desc="Processing images"):
            result = future.result()
            results.append(result)

    # Calculate success rate
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"Processing completed: {success_count}/{len(results)} images successfully processed")

    # Prepare CSV data
    csv_data = []
    valid_predictions = []
    errors = []

    for result in results:
        img_path = result['path']
        filename = os.path.basename(img_path)
        row = {
            'image_path': img_path,
            'status': result['status']
        }

        if result['status'] == 'success':
            row['prediction'] = result['prediction']
            valid_predictions.append(result['prediction'])

            # Find matching ID in the filename and get reference value if available
            if img_path in image_id_map:
                id_val = image_id_map[img_path]
                ref_value = reference_values[id_val]
                error = result['prediction'] - ref_value
                row['reference'] = ref_value
                row['error'] = error
                row['matched_id'] = id_val
                errors.append(error)
        else:
            row['error'] = result['error']

        csv_data.append(row)

    # Save CSV results
    csv_path = os.path.join(output_dir, 'results.csv')
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")

    # Generate visualizations for each image
    if valid_predictions:
        global_min = min(valid_predictions)
        global_max = max(valid_predictions)

        print("Generating visualizations for each image...")
        for result in tqdm(results):
            if result['status'] == 'success':
                img_path = result['path']
                base_name = os.path.basename(img_path)
                vis_path = os.path.join(output_dir, 'visualizations', f"{os.path.splitext(base_name)[0]}_result.png")

                # Get reference value if available
                ref_value = None
                if img_path in image_id_map:
                    id_val = image_id_map[img_path]
                    ref_value = reference_values[id_val]

                create_visualization(result, vis_path, global_min, global_max, ref_value)

        # Create gallery
        print("Generating image gallery...")
        create_gallery(results, output_dir, global_min=global_min, global_max=global_max,
                       reference_values=reference_values, image_id_map=image_id_map)

    # Generate statistics
    if valid_predictions:
        stats = {
            'processed_images': len(results),
            'successful': success_count,
            'failed': len(results) - success_count,
            'min_prediction': float(min(valid_predictions)),
            'max_prediction': float(max(valid_predictions)),
            'mean_prediction': float(np.mean(valid_predictions)),
            'median_prediction': float(np.median(valid_predictions)),
            'std_prediction': float(np.std(valid_predictions)),
            'processing_time_seconds': time.time() - start_time
        }

        # Add error statistics if reference values were provided
        if errors:
            stats.update({
                'mean_error': float(np.mean(errors)),
                'median_error': float(np.median(errors)),
                'max_abs_error': float(np.max(np.abs(errors))),
                'mean_abs_error': float(np.mean(np.abs(errors))),
                'std_error': float(np.std(errors)),
                'rmse': float(np.sqrt(np.mean(np.array(errors)**2)))
            })

        # Save statistics
        stats_path = os.path.join(output_dir, 'statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)

        print(f"Statistics saved to: {stats_path}")

        # Draw histograms
        # Prediction histogram
        plt.figure(figsize=(10, 6))
        plt.hist(valid_predictions, bins=20, alpha=0.7, color='royalblue')
        plt.axvline(np.mean(valid_predictions), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(valid_predictions):.2f}')
        plt.axvline(np.median(valid_predictions), color='g', linestyle='dashed', linewidth=2, label=f'Median: {np.median(valid_predictions):.2f}')
        plt.title('Prediction Value Distribution')
        plt.xlabel('Prediction Value')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        plt.legend()

        # Save histogram
        hist_path = os.path.join(output_dir, 'prediction_histogram.png')
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Prediction histogram saved to: {hist_path}")

        # If we have errors, generate error histogram
        if errors:
            plt.figure(figsize=(10, 6))
            plt.hist(errors, bins=20, alpha=0.7, color='salmon')
            plt.axvline(np.mean(errors), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
            plt.axvline(0, color='k', linestyle='solid', linewidth=2, label='Zero Error')
            plt.title('Error Distribution')
            plt.xlabel('Error (Predicted - Reference)')
            plt.ylabel('Frequency')
            plt.grid(alpha=0.3)
            plt.legend()

            # Save error histogram
            error_hist_path = os.path.join(output_dir, 'error_histogram.png')
            plt.savefig(error_hist_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Error histogram saved to: {error_hist_path}")

            # Generate scatter plot of predicted vs reference values
            if len(errors) > 1:
                # Collect pairs of reference and prediction values
                pred_ref_pairs = []
                for result in results:
                    if result['status'] == 'success' and result['path'] in image_id_map:
                        id_val = image_id_map[result['path']]
                        pred_ref_pairs.append((reference_values[id_val], result['prediction']))

                if pred_ref_pairs:
                    ref_vals, pred_vals = zip(*pred_ref_pairs)

                    plt.figure(figsize=(8, 8))
                    plt.scatter(ref_vals, pred_vals, alpha=0.7)

                    # Add identity line
                    min_val = min(min(ref_vals), min(pred_vals))
                    max_val = max(max(ref_vals), max(pred_vals))
                    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Identity Line')

                    # Add linear fit
                    if len(pred_ref_pairs) > 1:
                        z = np.polyfit(ref_vals, pred_vals, 1)
                        p = np.poly1d(z)
                        plt.plot(np.array([min_val, max_val]), p(np.array([min_val, max_val])),
                                 'r-', label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')

                    plt.title('Predicted vs Reference Values')
                    plt.xlabel('Reference Values')
                    plt.ylabel('Predicted Values')
                    plt.legend()
                    plt.grid(alpha=0.3)

                    # Save scatter plot
                    scatter_path = os.path.join(output_dir, 'pred_vs_ref_scatter.png')
                    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"Pred vs Ref scatter plot saved to: {scatter_path}")

    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    print(f"All results saved to: {output_dir}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Batch Image Texture Analysis Tool')
    parser.add_argument('--model_path', required=True, help='Path to pretrained model')
    parser.add_argument('--image_dir', required=True, help='Path to image directory')
    parser.add_argument('--output_dir', help='Path to output directory')
    parser.add_argument('--device', default='cuda', help='Compute device (cuda/cpu)')
    parser.add_argument('--workers', type=int, default=4, help='Number of processing threads')
    parser.add_argument('--csv_path', help='Path to CSV with reference values (optional)')

    args = parser.parse_args()

    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file does not exist: {args.model_path}")
        sys.exit(1)

    # Check if image directory exists
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory does not exist: {args.image_dir}")
        sys.exit(1)

    # Check if CSV file exists if provided
    if args.csv_path and not os.path.exists(args.csv_path):
        print(f"Warning: CSV file does not exist: {args.csv_path}")
        args.csv_path = None

    # Process image folder
    process_image_folder(
        model_path=args.model_path,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        device=args.device,
        n_workers=args.workers,
        csv_path=args.csv_path
    )

if __name__ == "__main__":
    main()
