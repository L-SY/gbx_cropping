# inference.py
"""
图像纹理回归推理脚本

使用方法:
    单张图像推理:
        python inference.py --model_path /path/to/model.pth --image_path /path/to/image.jpg

    目录批量推理:
        python inference.py --model_path /path/to/model.pth --predict_directory /path/to/images_folder

    评估数据集:
        python inference.py --model_path /path/to/model.pth --evaluate_dataset /path/to/dataset_folder

日期: 2025-04-02
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import glob

from models import FrozenCNNRegressor
from transforms import InnerBlackBorderAdder, get_inference_transform, get_validation_transform
from datasets import RegressionDataset
from torchvision import transforms

class ModelInference:
    """模型推理类"""
    def __init__(self, model_path, backbone='densenet121', device='cuda', border_width=70, output_dir='inference_results'):
        # 设置设备
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 保存输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 加载模型
        self.model = self.load_model(model_path, backbone)


        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 设置转换 - 确保与训练时完全一致
        self.transform = transform

        print("初始化预处理管道，设置:")
        print(f"  - 边框宽度: {border_width}")
        print(f"  - 图像大小: 224x224")

    def load_model(self, model_path, backbone):
        """加载模型"""
        try:
            print(f"从 {model_path} 加载模型")
            checkpoint = torch.load(model_path, map_location=self.device)

            # 创建模型实例
            model = FrozenCNNRegressor(
                backbone=backbone,
                pretrained=True,
                initial_value=15.0,
                dropout_rate=0.5
            )

            # 加载模型权重
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("已加载模型权重")
            else:
                model.load_state_dict(checkpoint)
                print("已加载整个模型")

            model.to(self.device)
            model.eval()  # 确保模型处于评估模式
            print(f"模型加载成功并设置为评估模式")
            return model
        except Exception as e:
            raise Exception(f"加载模型时出错: {e}")

    def predict_single_image(self, image_path):
        """预测单张图像"""
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            print(f"原始图像尺寸: {original_size}")

            # 预处理图像
            input_tensor = self.transform(image)
            print(f"转换后的张量形状: {input_tensor.shape}")

            # 添加批次维度并移至指定设备
            input_tensor = input_tensor.unsqueeze(0).to(self.device)

            # 执行推理
            with torch.no_grad():
                prediction = self.model(input_tensor).item()

            print(f"原始预测值: {prediction}")
            return prediction
        except Exception as e:
            print(f"预测图像 {image_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict_multiple_images(self, image_paths):
        """预测多张图像"""
        predictions = []
        for path in tqdm(image_paths, desc="预测多张图像"):
            pred = self.predict_single_image(path)
            predictions.append({
                'image_path': path,
                'prediction': pred
            })

        return predictions

    def evaluate_dataset(self, data_dir, labels_file=None, batch_size=32, num_workers=4):
        """评估整个数据集并识别误差最大的样本"""
        # 创建数据集和数据加载器
        dataset = RegressionDataset(data_dir, transform=self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        print(f"评估数据集，包含 {len(dataset)} 张图像")
        print(f"批次大小: {batch_size}")

        # 存储预测结果和真实标签
        all_predictions = []
        all_true_values = []
        image_names = []

        # 执行推理
        self.model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(dataloader, desc="评估数据集")):
                images = images.to(self.device)

                # 获取当前批次的图像名称
                batch_indices = list(range(i * batch_size, min((i + 1) * batch_size, len(dataset))))
                batch_image_names = [dataset.labels_df.iloc[idx]['image_name'] for idx in batch_indices]

                # 预测
                outputs = self.model(images)

                # 存储结果
                all_predictions.extend(outputs.cpu().numpy())
                all_true_values.extend(labels.cpu().numpy())
                image_names.extend(batch_image_names)

        # 创建结果数据帧
        results_df = pd.DataFrame({
            'image_name': image_names,
            'prediction': all_predictions,
            'true_value': all_true_values
        })

        # 计算指标(如果存在真实标签)
        metrics = {}
        has_labels = not np.isnan(all_true_values).all()

        if has_labels:
            # 过滤掉NaN值
            valid_indices = ~np.isnan(all_true_values)
            valid_results_df = results_df[valid_indices].copy()

            # 计算绝对误差
            valid_results_df['abs_error'] = np.abs(valid_results_df['prediction'] - valid_results_df['true_value'])
            valid_results_df['error'] = valid_results_df['prediction'] - valid_results_df['true_value']

            # 按绝对误差排序(降序)
            valid_results_df = valid_results_df.sort_values('abs_error', ascending=False)

            # 提取误差最大的N个样本
            top_n_errors = 5
            largest_errors_df = valid_results_df.head(top_n_errors)

            print(f"\n===== 误差最大的 {top_n_errors} 个样本 =====")
            for idx, row in largest_errors_df.iterrows():
                print(f"图像: {row['image_name']}")
                print(f"  真实值: {row['true_value']:.2f}")
                print(f"  预测值: {row['prediction']:.2f}")
                print(f"  绝对误差: {row['abs_error']:.2f}")
                print(f"  误差: {row['error']:.2f} ({'低估' if row['error'] < 0 else '高估'})")
                print("-" * 50)

            # 保存这些图像以供进一步分析
            error_analysis_dir = os.path.join(self.output_dir, 'error_analysis')
            os.makedirs(error_analysis_dir, exist_ok=True)

            # 复制误差最大的图像到分析目录
            for idx, row in largest_errors_df.iterrows():
                img_path = os.path.join(data_dir, row['image_name'])
                if os.path.exists(img_path):
                    # 加载并保存图像，在文件名中包含误差信息
                    img = Image.open(img_path)
                    error_info = f"true_{row['true_value']:.2f}_pred_{row['prediction']:.2f}_err_{row['error']:.2f}"
                    save_path = os.path.join(error_analysis_dir, f"{os.path.splitext(row['image_name'])[0]}_{error_info}.png")
                    img.save(save_path)

                    # 同时保存带有误差信息的可视化图像
                    self.visualize_prediction(img, row['true_value'], row['prediction'], save_path.replace('.png', '_viz.png'))

            print(f"\n误差分析图像已保存到: {error_analysis_dir}")

            # 将详细的误差分析保存为CSV
            largest_errors_df.to_csv(os.path.join(error_analysis_dir, 'largest_errors.csv'), index=False)

            # 计算标准指标
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

            # 计算误差分布
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

            # 计算不同误差范围内的样本比例
            metrics['error_ranges'] = {
                '<0.5': np.mean(np.abs(errors) < 0.5) * 100,
                '<1.0': np.mean(np.abs(errors) < 1.0) * 100,
                '<2.0': np.mean(np.abs(errors) < 2.0) * 100,
                '<5.0': np.mean(np.abs(errors) < 5.0) * 100,
                '>5.0': np.mean(np.abs(errors) >= 5.0) * 100
            }

            # 可视化结果
            self.visualize_evaluation_results(valid_true, valid_preds, metrics)

        return results_df, metrics

    def visualize_evaluation_results(self, true_values, predictions, metrics):
        """
        可视化评估结果，包括散点图和误差直方图
        """
        # 创建输出目录
        plots_dir = os.path.join(self.output_dir, 'evaluation_plots')
        os.makedirs(plots_dir, exist_ok=True)

        # 1. 散点图：预测值vs真实值
        plt.figure(figsize=(10, 8))
        plt.scatter(true_values, predictions, alpha=0.6)

        # 添加完美预测线
        min_val = min(min(true_values), min(predictions))
        max_val = max(max(true_values), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        plt.title(f'Predicted vs True Values\nR² = {metrics["r2"]:.4f}, RMSE = {metrics["rmse"]:.4f}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'scatter_plot.png'), dpi=200)
        plt.close()

    def visualize_prediction(self, image, true_value, prediction, save_path):
        """可视化带有预测和真实值的图像"""
        # 创建图形
        plt.figure(figsize=(10, 8))

        # 显示图像
        plt.imshow(image)

        # 添加带有预测信息的文本
        error = prediction - true_value
        error_text = f"Error: {error:.2f} ({'Underestimated' if error < 0 else 'Overestimated'})"

        plt.title(f"True Value: {true_value:.2f} | Predicted Value: {prediction:.2f}\n{error_text}", fontsize=14)
        plt.axis('off')

        # 保存图形
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='纹理回归模型推理')

    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--backbone', type=str, default='densenet121',
                        choices=['densenet121', 'densenet169', 'resnet18', 'resnet34', 'resnet50', 'mobilenet_v2'],
                        help='骨干网络选择 (默认: densenet121)')
    parser.add_argument('--border_width', type=int, default=70,
                        help='内部黑色边框宽度 (默认: 70)')

    # 推理模式
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--evaluate_dataset', type=str,
                       help='用于评估的数据集目录路径')
    group.add_argument('--image_path', type=str,
                       help='用于预测的单张图像路径')
    group.add_argument('--predict_directory', type=str,
                       help='包含多张图像的目录路径')

    # 其他参数
    parser.add_argument('--labels_file', type=str, default=None,
                        help='标签文件路径 (如果与默认路径不同)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小 (默认: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数 (默认: 4)')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='输出目录 (默认: inference_results)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='即使可用也不使用CUDA')

    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 设置设备
    device = 'cpu' if args.no_cuda else 'cuda'

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 打印系统信息
    print("=" * 50)
    print("系统信息:")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    print(f"使用设备: {device}")
    print("=" * 50)

    # 初始化模型推理器
    inferencer = ModelInference(
        model_path=args.model_path,
        backbone=args.backbone,
        device=device,
        border_width=args.border_width,
        output_dir=args.output_dir
    )

    # 根据命令行参数执行适当的推理任务
    if args.evaluate_dataset:
        print(f"评估数据集: {args.evaluate_dataset}")
        results_df, metrics = inferencer.evaluate_dataset(
            data_dir=args.evaluate_dataset,
            labels_file=args.labels_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        # 保存结果
        results_df.to_csv(os.path.join(args.output_dir, 'evaluation_results.csv'), index=False)

        # 打印指标
        if metrics:
            print("\n评估指标:")
            print(f"R²: {metrics['r2']:.4f}")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"MAE: {metrics['mae']:.4f}")
            print(f"平均误差: {metrics['mean_error']:.4f}")
            print(f"误差标准差: {metrics['std_error']:.4f}")
            print(f"最大误差: {metrics['max_error']:.4f}")

            print("\n不同误差范围内的样本比例:")
            for range_name, percentage in metrics['error_ranges'].items():
                print(f"  {range_name}: {percentage:.2f}%")

        print(f"\n详细结果已保存到: {os.path.join(args.output_dir, 'evaluation_results')}")

    elif args.image_path:
        print(f"预测单张图像: {args.image_path}")
        prediction = inferencer.predict_single_image(args.image_path)

        print(f"预测结果: {prediction:.4f}")

        # 保存结果
        with open(os.path.join(args.output_dir, 'single_prediction.txt'), 'w') as f:
            f.write(f"图像: {args.image_path}\n")
            f.write(f"预测值: {prediction:.4f}\n")

    elif args.predict_directory:
        print(f"预测目录中的图像: {args.predict_directory}")

        # 获取目录中的所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(args.predict_directory, f'*{ext}')))

        print(f"找到 {len(image_files)} 个图像文件")

        # 预测所有图像
        predictions = inferencer.predict_multiple_images(image_files)

        # 创建结果数据帧
        results_df = pd.DataFrame(predictions)

        # 保存结果
        results_df.to_csv(os.path.join(args.output_dir, 'batch_predictions.csv'), index=False)

        print(f"预测结果已保存到: {os.path.join(args.output_dir, 'batch_predictions.csv')}")

if __name__ == "__main__":
    main()
