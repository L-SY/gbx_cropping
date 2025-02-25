#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量图像纹理分析工具
用法: python inference.py --model_path MODEL_PATH --image_dir IMAGE_DIR [--output_dir OUTPUT_DIR] [--device DEVICE]
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

# 定义模型类 - 直接在这里内联定义，而不是从外部导入
class FrozenCNNRegressor(nn.Module):
    """使用冻结CNN特征提取器和可训练FC层的纹理回归模型"""
    def __init__(self, backbone='densenet121', pretrained=True, initial_value=15.0):
        super(FrozenCNNRegressor, self).__init__()

        # 加载预训练骨干网络
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
            # 移除全局平均池化层和全连接层
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
            raise ValueError(f"不支持的骨干网络: {backbone}")

        # 全局平均池化层
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 带有L2正则化效果的回归头 (类似Ridge回归)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        # 初始化最后一层的偏置为指定值
        final_layer = self.regressor[-1]
        nn.init.constant_(final_layer.bias, initial_value)

    def forward(self, x):
        # 提取特征
        features = self.features(x)

        # 全局平均池化
        pooled = self.global_pool(features)

        # 回归预测
        output = self.regressor(pooled).squeeze()

        return output

def load_model(model_path, device='cuda'):
    """加载预训练模型 - 使用更安全的方式"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    try:
        # 首先尝试只加载模型权重 - 更安全的方式
        try:
            # 创建空模型
            model = FrozenCNNRegressor(backbone='densenet121', pretrained=False)

            # 尝试使用weights_only=True的安全模式加载
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)

            # 如果是状态字典，直接加载
            if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
                model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

        except Exception as e:
            print(f"使用安全模式加载失败，尝试兼容模式: {e}")
            # 如果安全模式失败，使用兼容模式(不推荐，但确保能加载旧模型)
            print("警告: 使用不安全的加载方式，确保您信任此模型文件")
            checkpoint = torch.load(model_path, map_location=device)

            # 如果是完整模型而非字典
            if not isinstance(checkpoint, dict):
                model = checkpoint
            # 如果是包含state_dict的字典
            elif 'model_state_dict' in checkpoint:
                model = FrozenCNNRegressor(backbone='densenet121', pretrained=False)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 如果是直接的state_dict
                model = FrozenCNNRegressor(backbone='densenet121', pretrained=False)
                model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()
        print(f"模型成功加载: {model_path}")
        return model

    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)

def get_transform():
    """获取图像变换"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def predict_single_image(model, image_path, transform, device):
    """预测单个图像"""
    try:
        # 加载图像
        image = Image.open(image_path).convert('RGB')

        # 保存原始尺寸用于后续可视化
        original_size = image.size

        # 应用变换
        input_tensor = transform(image).unsqueeze(0).to(device)

        # 执行推理
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
        print(f"处理图像失败 {image_path}: {e}")
        return {
            'path': image_path,
            'prediction': None,
            'status': 'failed',
            'error': str(e)
        }

def generate_heatmap(prediction, min_val=0, max_val=30):
    """根据预测值生成热力图颜色"""
    # 创建自定义colormap，从蓝色(低值)到红色(高值)
    colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)

    # 将预测值归一化到0-1范围
    normalized = (prediction - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0, 1)  # 确保在0-1范围内

    # 获取RGB颜色
    rgb_color = cmap(normalized)[:3]  # 去除alpha通道

    # 将RGB转换为0-255整数
    color_8bit = tuple(int(c * 255) for c in rgb_color)

    return color_8bit, normalized

def create_visualization(result, output_path, global_min=0, global_max=30):
    """创建结果可视化"""
    if result['status'] != 'success':
        return

    image = result['image']
    prediction = result['prediction']

    # 使用全局最小值和最大值计算颜色
    color, normalized = generate_heatmap(prediction, global_min, global_max)

    # 创建图像
    fig, ax = plt.subplots(figsize=(10, 6))

    # 显示原始图像
    ax.imshow(image)

    # 添加预测值文本，设置颜色，使其在任何背景下都可见
    text = f"预测值: {prediction:.2f}"
    ax.text(10, 30, text, fontsize=16, weight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'))

    # 添加色条指示器
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,
                               norm=plt.Normalize(vmin=global_min, vmax=global_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('预测值范围')

    # 在色条上标记当前值
    cbar.ax.plot([0, 1], [prediction, prediction], 'k-', linewidth=2)

    # 关闭坐标轴
    ax.axis('off')

    # 设置标题
    filename = os.path.basename(result['path'])
    ax.set_title(f"文件: {filename} - 预测值: {prediction:.2f}", fontsize=14)

    # 保存图像
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def create_gallery(results, output_dir, rows=5, global_min=None, global_max=None):
    """创建图像库"""
    # 跳过处理失败的图像
    valid_results = [r for r in results if r['status'] == 'success']

    if not valid_results:
        print("没有有效结果可以生成图像库")
        return

    # 如果未指定全局范围，则从数据计算
    if global_min is None:
        global_min = min(r['prediction'] for r in valid_results)
    if global_max is None:
        global_max = max(r['prediction'] for r in valid_results)

    # 按预测值排序
    valid_results.sort(key=lambda x: x['prediction'])

    # 计算布局
    n_images = len(valid_results)
    cols = (n_images + rows - 1) // rows  # 向上取整

    # 创建画布
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    if n_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # 添加每个图像
    for i, result in enumerate(valid_results):
        if i < len(axes):
            ax = axes[i]
            ax.imshow(result['image'])

            # 添加预测值文本
            pred = result['prediction']
            filename = os.path.basename(result['path'])
            ax.set_title(f"{filename}\n预测值: {pred:.2f}", fontsize=10)

            # 设置边框颜色反映预测值
            color, _ = generate_heatmap(pred, global_min, global_max)
            for spine in ax.spines.values():
                spine.set_color(np.array(color)/255)
                spine.set_linewidth(5)

            # 关闭坐标轴
            ax.axis('off')

    # 隐藏多余的子图
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    # 调整布局
    plt.tight_layout()

    # 添加色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,
                               norm=plt.Normalize(vmin=global_min, vmax=global_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                        pad=0.01, fraction=0.05, aspect=40)
    cbar.set_label('预测值范围')

    # 添加标题
    fig.suptitle(f'批量预测结果总览 - 共{n_images}张图像', fontsize=16, y=1.02)

    # 保存图像库
    gallery_path = os.path.join(output_dir, 'results_gallery.png')
    plt.savefig(gallery_path, dpi=150, bbox_inches='tight')
    print(f"图像库已保存到: {gallery_path}")
    plt.close(fig)

def process_image_folder(model_path, image_dir, output_dir=None, device='cuda', n_workers=4):
    """处理整个图像文件夹"""
    start_time = time.time()

    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(image_dir), 'inference_results')

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)

    # 加载模型
    model = load_model(model_path, device)
    transform = get_transform()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 获取所有图像文件
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))

    image_files = sorted(list(set(image_files)))  # 去重并排序

    if not image_files:
        print(f"在 {image_dir} 中未找到图像文件")
        sys.exit(1)

    print(f"找到 {len(image_files)} 个图像文件")

    # 批量预测
    results = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # 创建任务
        future_to_path = {
            executor.submit(
                predict_single_image, model, path, transform, device
            ): path for path in image_files
        }

        # 处理结果
        for future in tqdm(as_completed(future_to_path), total=len(image_files), desc="处理图像"):
            result = future.result()
            results.append(result)

    # 计算成功率
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"处理完成: {success_count}/{len(results)} 张图像成功处理")

    # 准备CSV数据
    csv_data = []
    valid_predictions = []

    for result in results:
        row = {
            'image_path': result['path'],
            'status': result['status']
        }

        if result['status'] == 'success':
            row['prediction'] = result['prediction']
            valid_predictions.append(result['prediction'])
        else:
            row['error'] = result['error']

        csv_data.append(row)

    # 保存CSV结果
    csv_path = os.path.join(output_dir, 'results.csv')
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"结果已保存到: {csv_path}")

    # 生成每个图像的可视化
    if valid_predictions:
        global_min = min(valid_predictions)
        global_max = max(valid_predictions)

        print("生成每张图像的可视化结果...")
        for result in tqdm(results):
            if result['status'] == 'success':
                base_name = os.path.basename(result['path'])
                vis_path = os.path.join(output_dir, 'visualizations', f"{os.path.splitext(base_name)[0]}_result.png")
                create_visualization(result, vis_path, global_min, global_max)

        # 创建图像库
        print("生成图像库...")
        create_gallery(results, output_dir, global_min=global_min, global_max=global_max)

    # 生成统计信息
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

        # 保存统计信息
        stats_path = os.path.join(output_dir, 'statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)

        print(f"统计信息已保存到: {stats_path}")

        # 绘制直方图
        plt.figure(figsize=(10, 6))
        plt.hist(valid_predictions, bins=20, alpha=0.7, color='royalblue')
        plt.axvline(np.mean(valid_predictions), color='r', linestyle='dashed', linewidth=2, label=f'均值: {np.mean(valid_predictions):.2f}')
        plt.axvline(np.median(valid_predictions), color='g', linestyle='dashed', linewidth=2, label=f'中位数: {np.median(valid_predictions):.2f}')
        plt.title('预测值分布')
        plt.xlabel('预测值')
        plt.ylabel('频率')
        plt.grid(alpha=0.3)
        plt.legend()

        # 保存直方图
        hist_path = os.path.join(output_dir, 'prediction_histogram.png')
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"预测值分布直方图已保存到: {hist_path}")

    elapsed_time = time.time() - start_time
    print(f"总处理时间: {elapsed_time:.2f} 秒")
    print(f"所有结果已保存到: {output_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量图像纹理分析工具')
    parser.add_argument('--model_path', required=True, help='预训练模型路径')
    parser.add_argument('--image_dir', required=True, help='图像目录路径')
    parser.add_argument('--output_dir', help='输出目录路径')
    parser.add_argument('--device', default='cuda', help='计算设备 (cuda/cpu)')
    parser.add_argument('--workers', type=int, default=4, help='处理线程数')

    args = parser.parse_args()

    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        sys.exit(1)

    # 检查图像目录是否存在
    if not os.path.exists(args.image_dir):
        print(f"错误: 图像目录不存在: {args.image_dir}")
        sys.exit(1)

    # 处理图像文件夹
    process_image_folder(
        model_path=args.model_path,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        device=args.device,
        n_workers=args.workers
    )

if __name__ == "__main__":
    main()
