# visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from transforms import InnerBlackBorderAdder, FixedRotation, AdaptiveEdgeEnhancer, ContrastTextureEnhancer

def visualize_preprocessing(image_path, save_dir):
    """可视化不同预处理技术的效果"""
    # 原始图像
    original = Image.open(image_path).convert('RGB')

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 保存原始图像
    original.save(os.path.join(save_dir, '01_original.png'))

    # 应用各种预处理方法
    # 0. 添加黑色边框
    border_added = InnerBlackBorderAdder(border_width=70)(original)
    border_added.save(os.path.join(save_dir, '02_border_added.png'))

    # 1. 固定角度旋转
    rotated = FixedRotation(p=1.0)(original)
    rotated.save(os.path.join(save_dir, '03_rotated.png'))

    # 2. 自适应边缘增强
    edge_enhanced = AdaptiveEdgeEnhancer(p=1.0)(original)
    edge_enhanced.save(os.path.join(save_dir, '04_edge_enhanced.png'))

    # 3. 对比度增强
    contrast_enhanced = ContrastTextureEnhancer(p=1.0)(original)
    contrast_enhanced.save(os.path.join(save_dir, '05_contrast_enhanced.png'))

    # 4. 完整增强链
    transform = transforms.Compose([
        InnerBlackBorderAdder(border_width=70),
        AdaptiveEdgeEnhancer(p=1.0),
        ContrastTextureEnhancer(p=1.0)
    ])
    fully_enhanced = transform(original)
    fully_enhanced.save(os.path.join(save_dir, '06_fully_enhanced.png'))

    # 5. 比较可视化
    fig, axes = plt.subplots(1, 6, figsize=(24, 5))

    # 显示图像
    axes[0].imshow(np.array(original))
    axes[0].set_title('Original')

    axes[1].imshow(np.array(border_added))
    axes[1].set_title('Border Added')

    axes[2].imshow(np.array(rotated))
    axes[2].set_title('Rotated')

    axes[3].imshow(np.array(edge_enhanced))
    axes[3].set_title('Edge Enhanced')

    axes[4].imshow(np.array(contrast_enhanced))
    axes[4].set_title('Contrast Enhanced')

    axes[5].imshow(np.array(fully_enhanced))
    axes[5].set_title('Fully Enhanced')

    # 移除刻度
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '00_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"预处理可视化保存到 {save_dir}")