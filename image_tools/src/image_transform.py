import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F

def show_transform_steps(image_path):
    """
    展示所有预处理步骤的效果
    """
    # 1) 读取原图
    original_img = Image.open(image_path).convert('RGB')

    # 2) 随机旋转
    rotated_img = transforms.RandomRotation(degrees=30)(original_img)

    # 3) 随机锐度调整
    tensor_for_sharp = transforms.ToTensor()(rotated_img)
    sharpened_tensor = F.adjust_sharpness(tensor_for_sharp, 10)  # 使用1.5作为示例锐度因子
    sharpened_img = transforms.ToPILImage()(sharpened_tensor)

    # 4) 颜色变换
    color_jittered = transforms.ColorJitter(
        brightness=(1.0, 1.2),  # 只在原始亮度基础上增加0-20%
        contrast=(0.8, 1.2),
        saturation=(0.8, 1.2),
        hue=0.02
    )(sharpened_img)

    # 5) Resize
    resized_img = transforms.Resize((224, 224))(color_jittered)

    # 6) ToTensor
    tensor_img = transforms.ToTensor()(resized_img)

    # 7) Normalize
    normalize_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    normalized_tensor = normalize_transform(tensor_img)

    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Transform comparison of each step", fontsize=16)

    # 展示所有步骤
    images = [
        (original_img, "Original"),
        (rotated_img, "After Rotation"),
        (sharpened_img, "After Sharpness"),
        (color_jittered, "After ColorJitter"),
        (resized_img, "Resized (224×224)"),
    ]

    # 前5张图片
    for idx, (img, title) in enumerate(images):
        row = idx // 4
        col = idx % 4
        axes[row, col].imshow(img)
        axes[row, col].set_title(title)
        axes[row, col].axis("off")

    # ToTensor结果
    tensor_vis = tensor_img.permute(1, 2, 0).numpy()
    axes[1, 1].imshow(tensor_vis)
    axes[1, 1].set_title("After ToTensor()")
    axes[1, 1].axis("off")

    # Normalize结果
    normalized_vis = normalized_tensor.permute(1, 2, 0).numpy()
    normalized_vis = np.clip(normalized_vis, 0, 1)
    axes[1, 2].imshow(normalized_vis)
    axes[1, 2].set_title("After Normalize")
    axes[1, 2].axis("off")

    # 隐藏多余的子图
    axes[1, 3].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_folder = "/home/lsy/gbx_cropping_ws/src/image_tools/dataset/train"
    sample_image = "cropped_raw_2.jpg"
    image_path = os.path.join(train_folder, sample_image)

    show_transform_steps(image_path)