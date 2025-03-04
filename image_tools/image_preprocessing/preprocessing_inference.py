#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image extraction and texture analysis tool
Feature: Extract square region from image, split into four small squares, and perform neural network prediction
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set matplotlib to use English fonts
plt.rcParams['font.sans-serif'] = ['Arial']  # Using Arial instead of SimHei
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue

# Define model class - correctly matching the architecture of the saved model
class FrozenCNNRegressor(nn.Module):
    """CNN Regressor with frozen feature extractor and trainable FC layers"""
    def __init__(self, backbone='densenet121', pretrained=True, initial_value=15.0):
        super(FrozenCNNRegressor, self).__init__()

        # Load pretrained backbone
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

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Modified regressor to match the saved model architecture
        # Using 128 units instead of 64 in the intermediate layer to match checkpoint
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),  # Changed from 64 to 128
            nn.BatchNorm1d(128),  # Changed from 64 to 128
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),   # Additional layer
            nn.BatchNorm1d(64),   # Additional layer
            nn.ReLU(),            # Additional layer
            nn.Dropout(0.2),      # Additional layer (if needed)
            nn.Linear(64, 1)      # Final layer
        )

        # Initialize final layer bias
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

def extract_and_split_square(image_path, output_folder='output'):
    """
    Read image, extract square area, and split it into four equal small squares

    Args:
        image_path: Image path
        output_folder: Output folder
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Unable to read image: {image_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort by area, find the largest contour (assumed to be our square)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Look for contours approximating a square
    square_contour = None
    for contour in contours:
        # Estimate contour's polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # If polygon has 4 points, we consider it might be our square
        if len(approx) == 4:
            square_contour = approx
            break

    if square_contour is None:
        print("Could not find appropriate square contour")
        # Visualize all found contours for debugging
        debug_img = img.copy()
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.title("All Detected Contours")
        plt.axis('off')
        plt.show()
        return None

    # Draw the found square contour
    contour_img = img.copy()
    cv2.drawContours(contour_img, [square_contour], -1, (0, 255, 0), 2)
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Square")
    plt.axis('off')
    plt.show()

    # Reorder contour points for perspective transform
    points = square_contour.reshape(4, 2)
    rect = order_points(points)

    # Calculate new image size (10cm corresponding to pixels)
    # For simplicity, we set each cm to correspond to 50 pixels
    square_size = 500  # 10cm * 50 pixels/cm = 500 pixels

    # Target points after transformation
    dst = np.array([
        [0, 0],  # Top-left
        [square_size - 1, 0],  # Top-right
        [square_size - 1, square_size - 1],  # Bottom-right
        [0, square_size - 1]  # Bottom-left
    ], dtype=np.float32)

    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)

    # Apply perspective transform
    warped = cv2.warpPerspective(img, M, (square_size, square_size))

    # Show corrected square
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title("Corrected Square")
    plt.axis('off')
    plt.show()

    # Split square into four small squares
    half_size = square_size // 2

    # Define four small square regions
    top_left = warped[0:half_size, 0:half_size]
    top_right = warped[0:half_size, half_size:square_size]
    bottom_left = warped[half_size:square_size, 0:half_size]
    bottom_right = warped[half_size:square_size, half_size:square_size]

    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save four small squares
    cv2.imwrite(f"{output_folder}/top_left.jpg", top_left)
    cv2.imwrite(f"{output_folder}/top_right.jpg", top_right)
    cv2.imwrite(f"{output_folder}/bottom_left.jpg", bottom_left)
    cv2.imwrite(f"{output_folder}/bottom_right.jpg", bottom_right)

    # Show four small squares
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(cv2.cvtColor(top_left, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Top Left")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(cv2.cvtColor(top_right, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title("Top Right")
    axs[0, 1].axis('off')

    axs[1, 0].imshow(cv2.cvtColor(bottom_left, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title("Bottom Left")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(cv2.cvtColor(bottom_right, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title("Bottom Right")
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.suptitle("Four Split Squares", fontsize=16, y=0.98)
    plt.show()

    print(f"Processing complete. Four small squares saved to {output_folder} folder.")
    return {
        'top_left': top_left,
        'top_right': top_right,
        'bottom_left': bottom_left,
        'bottom_right': bottom_right,
        'original_image': img,
        'warped_image': warped
    }

def order_points(pts):
    """
    Order four corner points of a quadrilateral to match [top-left, top-right, bottom-right, bottom-left]
    This is necessary for perspective transformation
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # Top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has smallest difference, bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def load_model(model_path, device='cuda'):
    """Load pretrained model - using safer approach per security guidelines"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    try:
        # According to PyTorch security guidelines, we should prefer weights_only=True
        # See: https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models
        try:
            # Create empty model with corrected architecture
            model = FrozenCNNRegressor(backbone='densenet121', pretrained=False)
            # Try to load with weights_only=True for safety
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)

            # If it's a state dict, load directly
            if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
                model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

        except Exception as e:
            print(f"Failed to load safely, trying compatibility mode: {e}")
            # If safe mode fails, use compatibility mode (not recommended, but ensures loading old models)
            print("Warning: Using unsafe loading method, ensure you trust this model file")

            # Per security guidelines, this is risky for untrusted models
            # Only use this for your own models
            checkpoint = torch.load(model_path, map_location=device)

            # If it's a complete model, not a dict
            if not isinstance(checkpoint, dict):
                model = checkpoint
            # If it's a dict containing state_dict
            elif 'model_state_dict' in checkpoint:
                model = FrozenCNNRegressor(backbone='densenet121', pretrained=False)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # If it's a direct state_dict
                model = FrozenCNNRegressor(backbone='densenet121', pretrained=False)
                model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()
        print(f"Model successfully loaded: {model_path}")
        return model

    except Exception as e:
        print(f"Model loading failed: {e}")
        sys.exit(1)

def get_transform():
    """Get image transformation"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def predict_image(model, image, transform, device):
    """Predict single image"""
    try:
        # Convert OpenCV image to PIL image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image

        # Apply transformation
        input_tensor = transform(image_pil).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            prediction = model(input_tensor).item()

        return prediction
    except Exception as e:
        print(f"Image prediction failed: {e}")
        return None

def generate_heatmap(prediction, min_val=0, max_val=30):
    """Generate heatmap color based on prediction value"""
    # Create custom colormap, from blue (low) to red (high)
    colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)

    # Normalize prediction to 0-1 range
    normalized = (prediction - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0, 1)  # Ensure in 0-1 range

    # Get RGB color
    rgb_color = cmap(normalized)[:3]  # Remove alpha channel

    # Convert RGB to 0-255 integers
    color_8bit = tuple(int(c * 255) for c in rgb_color)

    return color_8bit, normalized

def create_visualization_with_predictions(squares, predictions, output_path=None, min_val=0, max_val=30):
    """Create visualization with prediction results"""
    warped = squares['warped_image']
    half_size = warped.shape[0] // 2

    # Create visualization image with prediction results
    vis_img = warped.copy()

    # Add prediction result text and color indicators
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2

    # Top left
    color_tl, _ = generate_heatmap(predictions['top_left'], min_val, max_val)
    cv2.rectangle(vis_img, (10, 10), (half_size - 10, half_size - 10), color_tl, thickness)
    cv2.putText(vis_img, f"{predictions['top_left']:.2f}", (20, half_size // 2),
                font, font_scale, color_tl, thickness * 2)

    # Top right
    color_tr, _ = generate_heatmap(predictions['top_right'], min_val, max_val)
    cv2.rectangle(vis_img, (half_size + 10, 10), (vis_img.shape[1] - 10, half_size - 10), color_tr, thickness)
    cv2.putText(vis_img, f"{predictions['top_right']:.2f}", (half_size + 20, half_size // 2),
                font, font_scale, color_tr, thickness * 2)

    # Bottom left
    color_bl, _ = generate_heatmap(predictions['bottom_left'], min_val, max_val)
    cv2.rectangle(vis_img, (10, half_size + 10), (half_size - 10, vis_img.shape[0] - 10), color_bl, thickness)
    cv2.putText(vis_img, f"{predictions['bottom_left']:.2f}", (20, half_size + half_size // 2),
                font, font_scale, color_bl, thickness * 2)

    # Bottom right
    color_br, _ = generate_heatmap(predictions['bottom_right'], min_val, max_val)
    cv2.rectangle(vis_img, (half_size + 10, half_size + 10),
                  (vis_img.shape[1] - 10, vis_img.shape[0] - 10), color_br, thickness)
    cv2.putText(vis_img, f"{predictions['bottom_right']:.2f}", (half_size + 20, half_size + half_size // 2),
                font, font_scale, color_br, thickness * 2)

    # Draw image with prediction results
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title("Square Region Prediction Results", fontsize=18)
    plt.axis('off')

    # Draw colorbar
    ax = plt.gca()
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,
                               norm=plt.Normalize(vmin=min_val, vmax=max_val))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Predicted Value')

    # Save result
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Results saved to: {output_path}")

    plt.show()

    return vis_img

def process_image_with_model(image_path, model_path, output_dir=None, device='cuda'):
    """Process image and predict values for each small square"""
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(image_path), 'results')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract and split square
    print("Extracting and splitting square region...")
    squares = extract_and_split_square(image_path, output_dir)

    if squares is None:
        print("Cannot process this image, exiting")
        return

    # Load model
    print("Loading model...")
    model = load_model(model_path, device)
    transform = get_transform()

    # Predict each small square
    print("Making predictions...")
    predictions = {}

    predictions['top_left'] = predict_image(model, squares['top_left'], transform, device)
    predictions['top_right'] = predict_image(model, squares['top_right'], transform, device)
    predictions['bottom_left'] = predict_image(model, squares['bottom_left'], transform, device)
    predictions['bottom_right'] = predict_image(model, squares['bottom_right'], transform, device)

    # Find prediction value range
    valid_predictions = [p for p in predictions.values() if p is not None]
    if not valid_predictions:
        print("All predictions failed, cannot create visualization")
        return

    min_val = min(valid_predictions)
    max_val = max(valid_predictions)

    # Create visualization
    output_path = os.path.join(output_dir, 'prediction_result.png')
    create_visualization_with_predictions(squares, predictions, output_path, min_val, max_val)

    # Save prediction results
    results = {
        'image_path': image_path,
        'predictions': predictions,
        'min_val': min_val,
        'max_val': max_val,
        'average': sum(valid_predictions) / len(valid_predictions)
    }

    # Save results as JSON
    json_path = os.path.join(output_dir, 'predictions.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Prediction complete! Results saved to {output_dir}")
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Image Extraction and Texture Analysis Tool')
    parser.add_argument('--model_path', required=True, help='Pretrained model path')
    parser.add_argument('--image_path', required=True, help='Input image path')
    parser.add_argument('--output_dir', help='Output directory path')
    parser.add_argument('--device', default='cuda', help='Computing device (cuda/cpu)')

    args = parser.parse_args()

    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file does not exist: {args.model_path}")
        sys.exit(1)

    # Check if image file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file does not exist: {args.image_path}")
        sys.exit(1)

    # Process image and make predictions
    process_image_with_model(
        image_path=args.image_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device
    )

if __name__ == "__main__":
    main()
