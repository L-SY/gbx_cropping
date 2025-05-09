import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transforms import (
    AdaptiveEdgeEnhancer,
    ContrastTextureEnhancer,
    InnerBlackBorderAdder,
    FixedRotation
)
from torchvision import transforms


def visualize_transform_pipeline(image_path, output_path=None):
    assert os.path.exists(image_path), f"图像路径不存在: {image_path}"
    img = Image.open(image_path).convert('RGB')
    steps = [("Original", img)]

    # 固定旋转
    img = FixedRotation(p=1.0)(img)
    steps.append(("Rotated", img))

    # 添加黑边
    img = InnerBlackBorderAdder(border_width=70)(img)
    steps.append(("Black Border", img))

    # 边缘增强
    img = AdaptiveEdgeEnhancer(p=1.0)(img)
    steps.append(("Edge Enhanced", img))

    # 对比度增强
    img = ContrastTextureEnhancer(p=1.0)(img)
    steps.append(("Contrast Enhanced", img))

    # Resize + ToTensor + Normalize (不用于显示，但可选)
    transform_final = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    final_tensor = transform_final(img)
    final_img = transforms.ToPILImage()(final_tensor)
    steps.append(("Resized + ToTensor", final_img))

    # 可视化
    fig, axs = plt.subplots(1, len(steps), figsize=(4 * len(steps), 5))
    for i, (title, im) in enumerate(steps):
        axs[i].imshow(im)
        axs[i].set_title(title)
        axs[i].axis('off')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"保存对比图至: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize transformation steps on a single image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_path", type=str, default=None, help="Optional path to save output image")
    args = parser.parse_args()

    visualize_transform_pipeline(args.image_path, args.output_path)
