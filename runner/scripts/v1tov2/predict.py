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

        # 根据图片尺寸计算边框宽度
        border_width_h = int(width * self.border_percentage)  # 水平边框宽度
        border_width_v = int(height * self.border_percentage)  # 垂直边框宽度

        # 绘制四个边的黑色边框
        draw.rectangle([(0, 0), (width, border_width_v)], fill="black")  # 上边
        draw.rectangle([(0, height - border_width_v), (width, height)], fill="black")  # 下边
        draw.rectangle([(0, 0), (border_width_h, height)], fill="black")  # 左边
        draw.rectangle([(width - border_width_h, 0), (width, height)], fill="black")  # 右边

        return bordered_img

# 模型定义函数
def get_model(model_name: str, dropout_rate: float, freeze_backbone: bool = False):
    """
    获取预训练模型并修改最后一层用于回归任务
    """
    model_name = model_name.lower()
    model_func = getattr(models, model_name, None)
    if model_func is None:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model_func(weights=None)

    # 冻结backbone参数
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # 修改最后一层
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
    """
    对单张图片进行倍率预测
    """
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
    """
    获取文件夹中所有支持的图片文件
    """
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []

    folder = Path(folder_path)
    if not folder.exists():
        print(f"文件夹不存在: {folder_path}")
        return []

    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            image_files.append(file_path)

    # 按文件名排序
    image_files.sort(key=lambda x: x.name)
    return image_files

def calculate_grid_size(num_images):
    """
    计算最佳的网格布局 (尽量接近正方形)
    """
    if num_images == 0:
        return 0, 0

    # 计算最接近正方形的布局
    sqrt_num = math.sqrt(num_images)
    cols = math.ceil(sqrt_num)
    rows = math.ceil(num_images / cols)

    return rows, cols

def folder_prediction_and_assembly(input_folder, output_folder, model_path,
                                   model_name='resnet50', dropout_rate=0.5,
                                   freeze_backbone=False, image_size=224,
                                   target_size=300, add_border=True,
                                   border_percentage=0.05, padding=10):
    """
    对文件夹中的图片进行预测并拼接成一张大图

    参数:
    - input_folder: 输入文件夹路径
    - output_folder: 输出文件夹路径
    - model_path: 预训练模型路径
    - model_name: 模型名称 (resnet50, densenet121等)
    - dropout_rate: Dropout率
    - freeze_backbone: 是否冻结backbone
    - image_size: 模型输入图片尺寸
    - target_size: 拼接时每张图片的大小
    - add_border: 是否添加边框
    - border_percentage: 边框宽度百分比
    - padding: 图片间距
    """

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有图片文件
    image_files = get_supported_image_files(input_folder)
    if not image_files:
        print(f"在文件夹 {input_folder} 中未找到支持的图片文件")
        return

    print(f"找到 {len(image_files)} 张图片")

    # 加载模型
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return

    print(f"加载模型: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    model = get_model(model_name, dropout_rate, freeze_backbone).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # 创建图像预处理
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

    # 创建边框处理器
    if add_border:
        border_adder = InnerBlackBorderAdder(border_percentage)

    # 预测每张图片并准备拼接数据
    prediction_results = []
    processed_images = []

    print("开始预测...")
    for i, image_path in enumerate(image_files):
        print(f"处理图片 {i+1}/{len(image_files)}: {image_path.name}")

        # 加载图片
        try:
            # 使用PIL加载图片
            pil_image = Image.open(image_path).convert('RGB')

            # 调整图片大小
            pil_image = pil_image.resize((target_size, target_size), Image.Resampling.LANCZOS)

            # 添加边框
            if add_border:
                pil_image = border_adder(pil_image)

            # 转换为numpy数组
            image_array = np.array(pil_image)

            # 进行预测
            prediction = predict_density(model, image_path, transform, device)

            prediction_results.append({
                'filename': image_path.name,
                'prediction': prediction
            })

            processed_images.append({
                'image': image_array,
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

    if not processed_images:
        print("没有成功处理的图片")
        return

    # 计算网格布局
    rows, cols = calculate_grid_size(len(processed_images))
    print(f"网格布局: {rows} 行 x {cols} 列")

    # 创建拼接画布
    canvas_width = cols * target_size + (cols - 1) * padding
    canvas_height = rows * target_size + (rows - 1) * padding
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # 白色背景

    print("开始拼接图片...")

    # 拼接图片
    for i, img_data in enumerate(processed_images):
        row = i // cols
        col = i % cols

        # 计算位置
        x = col * (target_size + padding)
        y = row * (target_size + padding)

        # 放置图片
        image = img_data['image']
        canvas[y:y+target_size, x:x+target_size] = image

        # 添加文字标注
        canvas_pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(canvas_pil)

        try:
            font_size = max(16, target_size // 20)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # 准备文字内容
        filename = img_data['filename']
        prediction = img_data['prediction']

        text_lines = [filename]
        if prediction is not None:
            text_lines.append(f"倍率: {prediction:.2f}")

        # 计算文字位置 (图片下方)
        text_x = x + 5
        text_y = y + target_size + 5

        # 绘制文字
        # for j, text in enumerate(text_lines):
        #     current_y = text_y + j * (font_size + 2)
        #
        #     # 确保文字不超出画布
        #     if current_y + font_size > canvas_height:
        #         break
        #
        #     # 获取文字边界框
        #     bbox = draw.textbbox((text_x, current_y), text, font=font)
        #     text_width = bbox[2] - bbox[0]
        #     text_height = bbox[3] - bbox[1]
        #
        #     # 绘制半透明背景
        #     draw.rectangle([text_x-2, current_y-2, text_x+text_width+2, current_y+text_height+2],
        #                    fill=(255, 255, 255, 200))
        #
        #     # 绘制文字
        #     draw.text((text_x, current_y), text, fill=(0, 0, 0), font=font)

        # 转换回numpy数组
        canvas = np.array(canvas_pil)

    # 保存拼接结果
    output_path = os.path.join(output_folder, "assembled_predictions.jpg")
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, canvas_bgr)

    # 保存预测结果文件
    results_file = os.path.join(output_folder, "prediction_results.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("图片预测结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"输入文件夹: {input_folder}\n")
        f.write(f"处理图片数量: {len(processed_images)}\n")
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

        # 统计信息
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
    print(f"拼接图片保存为: {output_path}")
    print(f"预测结果保存为: {results_file}")
    print(f"处理了 {len(processed_images)} 张图片")

    if len([r for r in prediction_results if r['prediction'] is not None]) > 0:
        valid_predictions = [r['prediction'] for r in prediction_results if r['prediction'] is not None]
        print(f"平均预测倍率: {np.mean(valid_predictions):.2f}")

# 使用示例
if __name__ == "__main__":
    # 请修改以下路径
    input_folder_path = "/runner/scripts/test/brightness/images/cs020/12mm/cropping"  # 输入图片文件夹路径
    output_directory = "/home/lsy/gbx_cropping_ws/src/runner/scripts/brightness/images/cs020/12mm/output"        # 输出文件夹路径
    model_file_path = "/home/lsy/gbx_cropping_ws/src/runner/scripts/v1tov2/best_model.pth"        # 模型文件路径

    # 执行预测和拼接
    folder_prediction_and_assembly(
        input_folder=input_folder_path,
        output_folder=output_directory,
        model_path=model_file_path,
        model_name='resnet50',      # 模型名称
        dropout_rate=0.5,           # Dropout率
        freeze_backbone=False,      # 是否冻结backbone
        image_size=224,             # 模型输入图片尺寸
        target_size=300,            # 拼接时每张图片的大小
        add_border=True,            # 是否添加边框
        border_percentage=0.1,    # 边框宽度百分比
        padding=10                  # 图片间距
    )

    print("\n使用说明:")
    print("1. 修改 input_folder_path 为包含图片的文件夹路径")
    print("2. 修改 output_directory 为输出文件夹路径")
    print("3. 修改 model_file_path 为训练好的模型文件路径")
    print("4. 支持的图片格式: jpg, jpeg, png, bmp, tiff, tif")
    print("5. 图片会按文件名排序后进行处理")
    print("6. 输出包括:")
    print("   - assembled_predictions.jpg: 拼接后的大图")
    print("   - prediction_results.txt: 详细预测结果")