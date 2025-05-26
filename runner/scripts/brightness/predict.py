import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import math
import torch
import torch.nn as nn
from torchvision import models, transforms
import argparse
from pathlib import Path

class InnerBlackBorderAdder(object):
    def __init__(self, border_percentage=0.05):
        self.border_percentage = border_percentage

    def __call__(self, img):
        width, height = img.size
        bordered_img = img.copy()
        draw = ImageDraw.Draw(bordered_img)

        border_width_h = int(width * self.border_percentage)
        border_width_v = int(height * self.border_percentage)

        draw.rectangle([(0, 0), (width, border_width_v)], fill="black")
        draw.rectangle([(0, height - border_width_v), (width, height)], fill="black")
        draw.rectangle([(0, 0), (border_width_h, height)], fill="black")
        draw.rectangle([(width - border_width_h, 0), (width, height)], fill="black")

        return bordered_img

def get_model(model_name: str, dropout_rate: float, freeze_backbone: bool = False):
    model_name = model_name.lower()
    model_func = getattr(models, model_name, None)
    if model_func is None:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model_func(weights=None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

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

def predict_density(model, image_path, transform, device):
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            model.eval()
            prediction = model(image_tensor).item()

        return prediction
    except Exception as e:
        print(f"预测图片 {image_path} 时出错: {e}")
        return None

def get_supported_image_files(folder_path):
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []

    folder = Path(folder_path)
    if not folder.exists():
        print(f"文件夹不存在: {folder_path}")
        return []

    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            image_files.append(file_path)

    image_files.sort(key=lambda x: x.name)
    return image_files

def calculate_grid_size(num_images):
    if num_images == 0:
        return 0, 0

    sqrt_num = math.sqrt(num_images)
    cols = math.ceil(sqrt_num)
    rows = math.ceil(num_images / cols)

    return rows, cols

def create_assembled_image(processed_images, target_size, padding, rows, cols):
    canvas_width = cols * target_size + (cols - 1) * padding
    canvas_height = rows * target_size + (rows - 1) * padding
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    for i, img_data in enumerate(processed_images):
        row = i // cols
        col = i % cols
        x = col * (target_size + padding)
        y = row * (target_size + padding)
        image = img_data['image']
        canvas[y:y+target_size, x:x+target_size] = image

    return canvas

def folder_prediction_and_assembly(input_folder, output_folder, model_path,
                                   model_name='resnet50', dropout_rate=0.5,
                                   freeze_backbone=False, image_size=224,
                                   target_size=300, add_border=True,
                                   border_percentage=0.05, padding=10):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = get_supported_image_files(input_folder)
    if not image_files:
        print(f"在文件夹 {input_folder} 中未找到支持的图片文件")
        return

    print(f"找到 {len(image_files)} 张图片")

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return

    print(f"加载模型: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = get_model(model_name, dropout_rate, freeze_backbone).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    transform = transforms.Compose([
        InnerBlackBorderAdder(border_percentage),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.220116, 0.220116, 0.220116],
            std=[0.178257, 0.178257, 0.178257]
        )
    ])

    if add_border:
        border_adder = InnerBlackBorderAdder(border_percentage)

    visual_transform = transforms.Compose([
        InnerBlackBorderAdder(border_percentage),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((target_size, target_size)),
    ])

    prediction_results = []
    original_images = []
    transformed_images = []

    print("开始预测...")
    for i, image_path in enumerate(image_files):
        print(f"处理图片 {i+1}/{len(image_files)}: {image_path.name}")

        try:
            pil_image = Image.open(image_path).convert('RGB')
            original_resized = pil_image.resize((target_size, target_size), Image.Resampling.LANCZOS)

            transformed_image = visual_transform(pil_image.copy())

            transformed_resized = transformed_image

            prediction = predict_density(model, image_path, transform, device)

            prediction_results.append({
                'filename': image_path.name,
                'prediction': prediction
            })

            original_images.append({
                'image': np.array(original_resized),
                'filename': image_path.name,
                'prediction': prediction
            })

            transformed_images.append({
                'image': np.array(transformed_resized),
                'filename': image_path.name,
                'prediction': prediction
            })

            if prediction is not None:
                print(f"  预测倍率: {prediction:.2f}")
            else:
                print(f"  预测失败")

        except Exception as e:
            print(f"处理图片 {image_path.name} 时出错: {e}")
            continue

    if not original_images:
        print("没有成功处理的图片")
        return

    rows, cols = calculate_grid_size(len(original_images))
    print(f"网格布局: {rows} 行 x {cols} 列")

    print("开始拼接图片...")

    original_canvas = create_assembled_image(original_images, target_size, padding, rows, cols)
    transformed_canvas = create_assembled_image(transformed_images, target_size, padding, rows, cols)

    original_output_path = os.path.join(output_folder, "assembled_original.jpg")
    transformed_output_path = os.path.join(output_folder, "assembled_transformed.jpg")

    original_canvas_bgr = cv2.cvtColor(original_canvas, cv2.COLOR_RGB2BGR)
    transformed_canvas_bgr = cv2.cvtColor(transformed_canvas, cv2.COLOR_RGB2BGR)

    cv2.imwrite(original_output_path, original_canvas_bgr)
    cv2.imwrite(transformed_output_path, transformed_canvas_bgr)

    results_file = os.path.join(output_folder, "prediction_results.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("图片预测结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"输入文件夹: {input_folder}\n")
        f.write(f"处理图片数量: {len(original_images)}\n")
        f.write(f"网格布局: {rows} 行 x {cols} 列\n")
        f.write(f"模型: {model_name}\n")
        f.write(f"设备: {device}\n\n")

        f.write("详细结果:\n")
        f.write("-" * 50 + "\n")
        for i, result in enumerate(prediction_results):
            f.write(f"{i+1:3d}. {result['filename']:<30} ")
            if result['prediction'] is not None:
                f.write(f"倍率: {result['prediction']:8.2f}")
            else:
                f.write("预测失败")
            f.write("\n")

        valid_predictions = [r['prediction'] for r in prediction_results if r['prediction'] is not None]
        if valid_predictions:
            f.write(f"\n统计信息:\n")
            f.write("-" * 30 + "\n")
            f.write(f"有效预测数量: {len(valid_predictions)}\n")
            f.write(f"平均倍率: {np.mean(valid_predictions):.2f}\n")
            f.write(f"最大倍率: {np.max(valid_predictions):.2f}\n")
            f.write(f"最小倍率: {np.min(valid_predictions):.2f}\n")
            f.write(f"标准差: {np.std(valid_predictions):.2f}\n")

    print(f"\n处理完成!")
    print(f"原始图片拼接保存为: {original_output_path}")
    print(f"Transform后图片拼接保存为: {transformed_output_path}")
    print(f"预测结果保存为: {results_file}")
    print(f"处理了 {len(original_images)} 张图片")

    if len([r for r in prediction_results if r['prediction'] is not None]) > 0:
        valid_predictions = [r['prediction'] for r in prediction_results if r['prediction'] is not None]
        print(f"平均预测倍率: {np.mean(valid_predictions):.2f}")

if __name__ == "__main__":
    input_folder_path = "/home/lsy/gbx_cropping_ws/src/runner/scripts/brightness/images/cs050/12mm/cropping"
    output_directory = "/home/lsy/gbx_cropping_ws/src/runner/scripts/brightness/images/cs050/12mm/output"
    model_file_path = "/home/lsy/gbx_cropping_ws/src/runner/scripts/v1tov2/best_model.pth"

    folder_prediction_and_assembly(
        input_folder=input_folder_path,
        output_folder=output_directory,
        model_path=model_file_path,
        model_name='resnet50',
        dropout_rate=0.5,
        freeze_backbone=False,
        image_size=224,
        target_size=300,
        add_border=True,
        border_percentage=0.1,
        padding=10
    )

    print("\n使用说明:")
    print("1. 修改 input_folder_path 为包含图片的文件夹路径")
    print("2. 修改 output_directory 为输出文件夹路径")
    print("3. 修改 model_file_path 为训练好的模型文件路径")
    print("4. 支持的图片格式: jpg, jpeg, png, bmp, tiff, tif")
    print("5. 图片会按文件名排序后进行处理")
    print("6. 输出包括:")
    print("   - assembled_original.jpg: 原始图片拼接")
    print("   - assembled_transformed.jpg: Transform后图片拼接")
    print("   - prediction_results.txt: 详细预测结果")