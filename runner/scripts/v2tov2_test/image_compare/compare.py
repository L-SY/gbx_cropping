import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_transform_images(folder_path, image_size=224):
    """
    加载文件夹中的两个图片并应用相同的transform
    """
    # 定义transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 获取文件夹中的图片文件
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if len(image_files) < 2:
        raise ValueError(f"文件夹中至少需要2个图片文件，但只找到了 {len(image_files)} 个")

    # 取前两个图片
    img1_path = os.path.join(folder_path, image_files[0])
    img2_path = os.path.join(folder_path, image_files[1])

    # 加载并处理图片
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    # 应用transform
    img1_transformed = transform(img1)
    img2_transformed = transform(img2)

    return img1_transformed, img2_transformed, image_files[0], image_files[1]

def denormalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    将标准化后的tensor反标准化，用于显示
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def tensor_to_image(tensor):
    """
    将tensor转换为可显示的numpy数组
    """
    # 反标准化
    denorm_tensor = denormalize_tensor(tensor)
    # 限制值在0-1之间
    denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
    # 转换为numpy并调整维度顺序
    return denorm_tensor.permute(1, 2, 0).numpy()

def compare_images(folder_path, image_size=224, save_path=None):
    """
    比较两个图片的处理结果
    """
    try:
        # 加载和处理图片
        img1_tensor, img2_tensor, name1, name2 = load_and_transform_images(folder_path, image_size)

        # 转换为可显示的格式
        img1_display = tensor_to_image(img1_tensor)
        img2_display = tensor_to_image(img2_tensor)

        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 显示第一个图片
        axes[0].imshow(img1_display)
        axes[0].set_title(f'Image 1: {name1}')
        axes[0].axis('off')

        # 显示第二个图片
        axes[1].imshow(img2_display)
        axes[1].set_title(f'Image 2: {name2}')
        axes[1].axis('off')

        # 显示差异图
        diff = np.abs(img1_display - img2_display)
        axes[2].imshow(diff)
        axes[2].set_title('Difference (Absolute)')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison image saved to: {save_path}")

        plt.show()

        # 计算一些统计信息
        mse = torch.mean((img1_tensor - img2_tensor) ** 2).item()
        print(f"\nStatistics:")
        print(f"Mean Squared Error (MSE): {mse:.6f}")
        print(f"Image 1 tensor shape: {img1_tensor.shape}")
        print(f"Image 2 tensor shape: {img2_tensor.shape}")
        print(f"Processed image size: {image_size}x{image_size}")

        return img1_tensor, img2_tensor

    except Exception as e:
        print(f"Error during processing: {e}")
        return None, None

# 使用示例
if __name__ == "__main__":
    # 设置文件夹路径
    folder_path = "/home/lsy/gbx_cropping_ws/src/runner/scripts/v2tov2_test/image_compare"  # 替换为你的文件夹路径
    image_size = 224  # 可以调整图片大小

    # 进行图片对比
    print("Processing images...")
    img1_tensor, img2_tensor = compare_images(
        folder_path=folder_path,
        image_size=image_size,
        save_path="image_comparison.png"  # 可选：保存对比图
    )

    if img1_tensor is not None:
        print("Processing completed!")

        # 如果需要进一步处理tensor，可以在这里继续
        # 例如：计算相似度、特征提取等

        # 计算余弦相似度
        img1_flat = img1_tensor.flatten()
        img2_flat = img2_tensor.flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            img1_flat.unsqueeze(0),
            img2_flat.unsqueeze(0)
        ).item()
        print(f"Cosine similarity: {cosine_sim:.6f}")