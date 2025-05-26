import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
from tqdm import tqdm
import numpy as np

def calculate_dataset_statistics(dataset_path, image_size=224, use_grayscale=False, batch_size=32):
    """
    计算数据集的均值和标准差

    Args:
        dataset_path: 图片文件夹路径
        image_size: resize后的图片大小
        use_grayscale: 是否转为灰度图
        batch_size: 批处理大小，避免内存溢出
    """

    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']

    # 获取所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))
        image_files.extend(glob.glob(os.path.join(dataset_path, '**', ext.upper()), recursive=True))

    if len(image_files) == 0:
        raise ValueError(f"No images found in {dataset_path}")

    print(f"🔍 Found {len(image_files)} images in {dataset_path}")

    # 设置transform
    if use_grayscale:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    # 初始化统计变量
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    total_pixels = 0

    print("📊 Calculating statistics...")

    # 分批处理避免内存溢出
    for i in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
        batch_files = image_files[i:i + batch_size]
        batch_tensors = []

        # 加载当前批次的图片
        for img_path in batch_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                batch_tensors.append(img_tensor)
            except Exception as e:
                print(f"⚠️  Warning: Could not load {img_path}: {e}")
                continue

        if batch_tensors:
            # 将批次转为tensor
            batch = torch.stack(batch_tensors)  # Shape: (batch_size, 3, H, W)

            # 累积统计
            batch_sum = batch.sum(dim=[0, 2, 3])  # 对batch, height, width求和
            batch_squared_sum = (batch ** 2).sum(dim=[0, 2, 3])
            batch_pixels = batch.shape[0] * batch.shape[2] * batch.shape[3]

            channel_sum += batch_sum
            channel_squared_sum += batch_squared_sum
            total_pixels += batch_pixels

    # 计算最终统计量
    mean = channel_sum / total_pixels
    var = (channel_squared_sum / total_pixels) - (mean ** 2)
    std = torch.sqrt(var)

    return mean, std, len(image_files)

def display_statistics(mean, std, num_images, dataset_path):
    """
    显示统计结果
    """
    print("\n" + "="*60)
    print(f"📈 Dataset Statistics for {dataset_path}")
    print("="*60)

    print(f"📁 Total images processed: {num_images}")
    print(f"🎨 Channel-wise statistics:")

    channels = ['Red  ', 'Green', 'Blue ']
    for i, channel in enumerate(channels):
        print(f"   {channel}: mean = {mean[i]:.6f}, std = {std[i]:.6f}")

    print(f"\n🔧 Use these values in your transform:")
    print(f"transforms.Normalize(")
    print(f"    mean=[{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}],")
    print(f"    std=[{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
    print(f")")

    # 与ImageNet对比
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
    imagenet_std = torch.tensor([0.229, 0.224, 0.225])

    print(f"\n📊 Comparison with ImageNet:")
    print(f"                Your Dataset    ImageNet        Difference")
    print(f"   Red   mean:  {mean[0]:.6f}        0.485000        {abs(mean[0]-imagenet_mean[0]):.6f}")
    print(f"   Green mean:  {mean[1]:.6f}        0.456000        {abs(mean[1]-imagenet_mean[1]):.6f}")
    print(f"   Blue  mean:  {mean[2]:.6f}        0.406000        {abs(mean[2]-imagenet_mean[2]):.6f}")
    print(f"   Red   std:   {std[0]:.6f}        0.229000        {abs(std[0]-imagenet_std[0]):.6f}")
    print(f"   Green std:   {std[1]:.6f}        0.224000        {abs(std[1]-imagenet_std[1]):.6f}")
    print(f"   Blue  std:   {std[2]:.6f}        0.225000        {abs(std[2]-imagenet_std[2]):.6f}")

    # 给出建议
    mean_diff = torch.mean(torch.abs(mean - imagenet_mean)).item()
    std_diff = torch.mean(torch.abs(std - imagenet_std)).item()

    print(f"\n💡 Recommendations:")
    if mean_diff > 0.1 or std_diff > 0.05:
        print(f"   ✅ Use your custom statistics (significant difference from ImageNet)")
        print(f"   ❌ Avoid ImageNet normalization for your dataset")
    else:
        print(f"   ✅ Your data is similar to ImageNet, both normalizations should work")
        print(f"   ✅ ImageNet normalization is fine to use")

def compare_normalizations(dataset_path, custom_mean, custom_std, sample_size=5):
    """
    对比不同标准化方法的效果
    """
    print(f"\n" + "="*60)
    print(f"🔬 Normalization Comparison on Sample Images")
    print("="*60)

    # 获取几张样本图片
    image_files = glob.glob(os.path.join(dataset_path, "*.jpg")) + \
                  glob.glob(os.path.join(dataset_path, "*.png"))

    if len(image_files) == 0:
        print("No sample images found for comparison")
        return

    sample_files = image_files[:min(sample_size, len(image_files))]

    # 不同的标准化方法
    transforms_dict = {
        "No Normalization": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),
        "ImageNet Norm": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "Custom Norm": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=custom_mean.tolist(), std=custom_std.tolist())
        ]),
        "Simple Norm": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    }

    print(f"📊 Processing {len(sample_files)} sample images...")

    for method_name, transform in transforms_dict.items():
        all_tensors = []

        for img_path in sample_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                all_tensors.append(img_tensor)
            except Exception as e:
                continue

        if all_tensors:
            batch = torch.stack(all_tensors)
            mean_val = batch.mean().item()
            std_val = batch.std().item()
            min_val = batch.min().item()
            max_val = batch.max().item()

            print(f"\n{method_name:15}: range=[{min_val:7.3f}, {max_val:7.3f}], "
                  f"mean={mean_val:7.3f}, std={std_val:6.3f}")

def save_statistics(mean, std, num_images, save_path="./dataset_stats.txt"):
    """
    保存统计结果到文件
    """
    with open(save_path, 'w') as f:
        f.write(f"Dataset Statistics\n")
        f.write(f"==================\n")
        f.write(f"Total images: {num_images}\n")
        f.write(f"Mean: [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]\n")
        f.write(f"Std:  [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]\n\n")
        f.write(f"PyTorch Transform Code:\n")
        f.write(f"transforms.Normalize(\n")
        f.write(f"    mean=[{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}],\n")
        f.write(f"    std=[{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]\n")
        f.write(f")\n")

    print(f"💾 Statistics saved to {save_path}")

# 主函数
if __name__ == "__main__":
    # 配置参数
    dataset_path = "/home/siyang_liu/gbx_cropping_ws/paper/images/augmented_dataset"  # 修改为你的数据集路径
    image_size = 224    # 修改为你需要的图片大小
    use_grayscale = True  # 是否转为灰度图
    batch_size = 32     # 批处理大小，如果内存不够可以调小

    print("🚀 Dataset Statistics Calculator")
    print(f"📁 Dataset path: {dataset_path}")
    print(f"🖼️  Image size: {image_size}x{image_size}")
    print(f"🎨 Grayscale: {use_grayscale}")
    print(f"📦 Batch size: {batch_size}")

    try:
        # 计算统计量
        mean, std, num_images = calculate_dataset_statistics(
            dataset_path=dataset_path,
            image_size=image_size,
            use_grayscale=use_grayscale,
            batch_size=batch_size
        )

        # 显示结果
        display_statistics(mean, std, num_images, dataset_path)

        # 保存结果
        save_statistics(mean, std, num_images)

        # 对比不同标准化方法
        compare_normalizations(dataset_path, mean, std)

        print(f"\n✅ Analysis complete! Use your custom normalization for best results.")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Tips:")
        print("   - Check if the dataset path exists")
        print("   - Make sure there are images in the folder")
        print("   - Try reducing batch_size if out of memory")