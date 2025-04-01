import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import os
import sys
import argparse
from datetime import datetime

def process_data(input_csv_path, pytorch_dirs=None):
    # 创建输出目录
    base_dir = os.path.dirname(input_csv_path)
    compute_dir = os.path.join(base_dir, "processed")
    if not os.path.exists(compute_dir):
        os.makedirs(compute_dir)

    # 生成输出文件名
    filename = os.path.basename(input_csv_path)
    filename_without_ext = os.path.splitext(filename)[0]
    if filename_without_ext.startswith("raw_"):
        filename_without_ext = filename_without_ext[4:]
    output_csv = os.path.join(compute_dir, f"{filename_without_ext}_processed.csv")
    comparison_plot = os.path.join(compute_dir, f"{filename_without_ext}_comparison.png")
    distribution_plot = os.path.join(compute_dir, f"{filename_without_ext}_distribution.png")

    # 定义线性方程
    slope = -443.78698224852076
    intercept = 38.9792899408284

    def predict_compute_rate(density):
        return slope * density + intercept

    # 读取原始数据
    raw_data = pd.read_csv(input_csv_path)

    # 计算倍率并限制范围
    raw_data['ComputedRate'] = raw_data['Density'].apply(predict_compute_rate)
    raw_data['ComputedRate'] = raw_data['ComputedRate'].round(1)
    raw_data['ComputedRate'] = raw_data['ComputedRate'].clip(10, 25)

    # 保存处理后的CSV文件
    raw_data.to_csv(output_csv, index=False)
    print(f"Added computed rates and saved to {output_csv}")

    # 绘制对比图
    plt.figure(figsize=(12, 7))
    sorted_data = raw_data.sort_values(by='Density')
    plt.plot(sorted_data['Density'], sorted_data['ComputedRate'], 'ro-', label='Computed Rate', linewidth=2, markersize=6)
    if 'ReferenceRate' in raw_data.columns:
        plt.plot(sorted_data['Density'], sorted_data['ReferenceRate'], 'bo-', label='Reference Rate', linewidth=2, markersize=6)
    plt.title('Comparison between Reference Rate and Computed Rate', fontsize=16)
    plt.xlabel('Density', fontsize=14)
    plt.ylabel('Rate', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.figtext(0.15, 0.02, f'Compute Rate Formula: Rate = {slope:.4f} * Density + {intercept:.4f}', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(comparison_plot)
    plt.close()
    print(f"Saved comparison plot to {comparison_plot}")

    # 如果指定了 PyTorch 输出目录，则生成 labels.csv 文件
    if pytorch_dirs:
        for pytorch_dir in pytorch_dirs:
            if not os.path.exists(pytorch_dir):
                os.makedirs(pytorch_dir)

            # 从路径中提取"front"或"back"
            path_parts = pytorch_dir.split(os.sep)
            position = None
            for part in path_parts:
                if part in ["front", "back"]:
                    position = part
                    break

            # 如果没有找到"front"或"back"，使用默认值
            if position is None:
                position = "unknown"

            pytorch_labels_path = os.path.join(pytorch_dir, 'labels.csv')
            pytorch_labels = raw_data[['ID', 'ComputedRate']].copy()

            # 将位置信息添加到图像名称中
            pytorch_labels['image_name'] = pytorch_labels['ID'].apply(lambda x: f"cropped_{position}_{x}.jpg")

            pytorch_labels = pytorch_labels[['image_name', 'ComputedRate']]
            pytorch_labels.columns = ['image_name', 'label']
            pytorch_labels.to_csv(pytorch_labels_path, index=False)
            print(f"Generated PyTorch labels and saved to {pytorch_labels_path}")

    return output_csv, comparison_plot, distribution_plot if 'ReferenceRatio' in raw_data.columns else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理CSV文件并计算倍率')
    parser.add_argument('--input_csv', help='输入CSV文件的路径')
    parser.add_argument('--pytorch_dirs', nargs='+', help='输出PyTorch标签文件的目录列表', required=False)
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"错误: 找不到文件 {args.input_csv}")
        sys.exit(1)

    try:
        output_csv, comparison_plot, distribution_plot = process_data(args.input_csv, args.pytorch_dirs)
        print("\n处理完成!")
        print(f"- 处理后的CSV: {output_csv}")
        print(f"- 对比图: {comparison_plot}")
        if distribution_plot:
            print(f"- 分布分析图: {distribution_plot}")
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        sys.exit(1)
