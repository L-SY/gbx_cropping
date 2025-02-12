import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def show_transform_steps(image_path):
    """
    从指定 image_path 读取图像，依次执行:
      1) 原图
      2) Resize(224, 224)
      3) ToTensor()
      4) Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    并分别显示在同一张图上。
    """
    # 1) 读取原图
    original_img = Image.open(image_path).convert('RGB')

    # 2) Resize
    resized_img = transforms.Resize((224, 224))(original_img)

    # 3) ToTensor (结果是 [C, H, W] 范围在 [0,1])
    tensor_img = transforms.ToTensor()(resized_img)

    # 4) Normalize (使用 ImageNet 预训练默认平均值和标准差)
    normalize_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
    normalized_tensor = normalize_transform(tensor_img)

    # 开始可视化
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Transform comparison of each step", fontsize=16)

    # (a) 原始图像
    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # (b) Resize 后图像
    axes[1].imshow(resized_img)
    axes[1].set_title("Resized (224×224)")
    axes[1].axis("off")

    # (c) ToTensor() 后的图像
    #     张量转到 [H, W, C] 并用 .numpy() 转为 NumPy 数组以可视化
    tensor_c = tensor_img.permute(1, 2, 0).numpy()
    # 因为它本身已经在 [0,1] 范围，可以直接显示
    axes[2].imshow(tensor_c)
    axes[2].set_title("After ToTensor()")
    axes[2].axis("off")

    # (d) Normalize 后的图像
    #     归一化后张量数值可能包含负值，这里用 clip 到 [0,1] 范围便于可视化
    normalized_c = normalized_tensor.permute(1, 2, 0).numpy()
    normalized_c = np.clip(normalized_c, 0, 1)
    axes[3].imshow(normalized_c)
    axes[3].set_title("After Normalize")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 假设 train 文件夹路径
    train_folder = "/home/lsy/gbx_cropping_ws/src/image_tools/dataset/train"  # 请替换成自己的train目录
    sample_image = "cropped_raw_2.jpg"               # 请替换成自己train目录下的一张有效图片
    image_path = os.path.join(train_folder, sample_image)

    show_transform_steps(image_path)