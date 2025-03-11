import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import argparse
from tqdm import tqdm
import math

def add_black_border(img, border_width=70):
    """在图像内部添加黑色边框"""
    width, height = img.size
    bordered_img = img.copy()
    draw = ImageDraw.Draw(bordered_img)

    draw.rectangle([(0, 0), (width, border_width)], fill="black")
    draw.rectangle([(0, height - border_width), (width, height)], fill="black")
    draw.rectangle([(0, 0), (border_width, height)], fill="black")  # 修复：使用参数而不是self
    draw.rectangle([(width - border_width, 0), (width, height)], fill="black")

    return bordered_img

def create_montage(dataset_path, output_path, border_width=70, thumbnail_size=(224, 224),
                   max_images=None, grid_size=None, background_color="white",
                   sort_ascending=True, add_labels=True, font_size=12):
    """
    创建一个按照label值排序的图片拼图

    参数:
        dataset_path: 包含图片和labels.csv的目录路径
        output_path: 输出拼图的文件路径
        border_width: 黑色边框宽度
        thumbnail_size: 缩略图大小
        max_images: 最大包含的图片数量 (None表示全部)
        grid_size: 网格大小 (None表示自动计算)
        background_color: 背景颜色
        sort_ascending: 是否按升序排列
        add_labels: 是否添加标签值
        font_size: 标签字体大小
    """
    # 检查输出路径是否是目录
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "montage.jpg")
        print(f"输出路径是目录，将保存到：{output_path}")

    # 读取标签文件
    labels_file = os.path.join(dataset_path, "labels.csv")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"标签文件未找到: {labels_file}")

    df = pd.read_csv(labels_file)
    print(f"读取到 {len(df)} 个图片条目")

    # 按标签值排序
    df = df.sort_values('label', ascending=sort_ascending)

    # 限制图片数量
    if max_images is not None and max_images < len(df):
        df = df.iloc[:max_images]
        print(f"限制为前 {max_images} 张图片")

    # 验证所有图片是否存在
    valid_entries = []
    for idx, row in df.iterrows():
        img_path = os.path.join(dataset_path, row['image_name'])
        if os.path.exists(img_path):
            valid_entries.append(row)
        else:
            print(f"警告: 图片未找到: {img_path}")

    df = pd.DataFrame(valid_entries)
    print(f"找到 {len(df)} 个有效图片")

    if len(df) == 0:
        raise ValueError("没有有效图片，请检查数据集路径")

    # 确定网格大小
    if grid_size is None:
        # 计算最接近的正方形网格
        grid_cols = int(math.ceil(math.sqrt(len(df))))
        grid_rows = int(math.ceil(len(df) / grid_cols))
    else:
        grid_rows, grid_cols = grid_size

    print(f"网格大小: {grid_rows} 行 x {grid_cols} 列")

    # 创建画布
    canvas_width = grid_cols * thumbnail_size[0]
    canvas_height = grid_rows * thumbnail_size[1]
    canvas = Image.new('RGB', (canvas_width, canvas_height), background_color)

    try:
        # 如果需要添加标签，尝试导入PIL的ImageFont模块
        font = None
        if add_labels:
            try:
                from PIL import ImageFont
                try:
                    # 尝试加载默认字体
                    font = ImageFont.truetype("arial.ttf", font_size)
                except IOError:
                    try:
                        # 尝试使用PIL默认字体
                        font = ImageFont.load_default()
                    except:
                        print("无法加载字体，将使用简单文本")
            except ImportError:
                print("ImageFont模块不可用，无法添加标签")
    except Exception as e:
        print(f"设置字体时出错: {e}")
        add_labels = False

    # 处理每张图片并添加到画布
    for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="创建拼图")):
        # 计算在网格中的位置
        grid_x = i % grid_cols
        grid_y = i // grid_cols

        # 如果位置超出网格，跳过
        if grid_y >= grid_rows:
            print(f"警告: 图片 {row['image_name']} 超出网格范围，将被跳过")
            continue

        # 加载图片
        img_path = os.path.join(dataset_path, row['image_name'])
        try:
            img = Image.open(img_path).convert('RGB')

            # 添加黑色边框
            img = add_black_border(img, border_width)

            # 调整大小
            img = img.resize(thumbnail_size, Image.LANCZOS)

            # 如果需要添加标签
            if add_labels:
                draw = ImageDraw.Draw(img)
                label_text = f"{row['label']:.2f}"

                # 绘制文本背景
                text_width = font_size * len(label_text) * 0.7  # 估算文本宽度
                text_height = font_size * 1.5
                text_x = 5
                text_y = 5
                draw.rectangle(
                    [(text_x, text_y), (text_x + text_width, text_y + text_height)],
                    fill=(0, 0, 0, 180)  # 半透明黑色背景
                )

                # 绘制文本
                draw.text(
                    (text_x + 2, text_y + 2),
                    label_text,
                    fill=(255, 255, 255),  # 白色文本
                    font=font
                )

            # 计算在画布上的位置
            x = grid_x * thumbnail_size[0]
            y = grid_y * thumbnail_size[1]

            # 将图片粘贴到画布上
            canvas.paste(img, (x, y))

        except Exception as e:
            print(f"处理图片时出错 {img_path}: {e}")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 保存最终拼图
    canvas.save(output_path, quality=95)
    print(f"拼图已保存到: {output_path}")

    # 返回用于显示的拼图
    return canvas

def main():
    parser = argparse.ArgumentParser(description="创建按标签值排序的图片拼图")
    parser.add_argument("--dataset_path", type=str, required=True, help="包含图片和labels.csv的目录路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出拼图的文件路径(可以是文件或目录)")
    parser.add_argument("--border_width", type=int, default=70, help="黑色边框宽度 (默认: 70)")
    parser.add_argument("--thumbnail_size", type=int, nargs=2, default=[224, 224], help="缩略图大小 (默认: 224 224)")
    parser.add_argument("--max_images", type=int, default=None, help="最大包含的图片数量 (默认: 所有)")
    parser.add_argument("--grid_cols", type=int, default=None, help="网格列数 (默认: 自动计算)")
    parser.add_argument("--grid_rows", type=int, default=None, help="网格行数 (默认: 自动计算)")
    parser.add_argument("--background", type=str, default="white", help="背景颜色 (默认: white)")
    parser.add_argument("--descending", action="store_true", help="按降序排列 (默认: 升序)")
    parser.add_argument("--no_labels", action="store_true", help="不显示标签值 (默认: 显示)")
    parser.add_argument("--font_size", type=int, default=12, help="标签字体大小 (默认: 12)")

    args = parser.parse_args()

    # 设置网格大小
    grid_size = None
    if args.grid_cols is not None and args.grid_rows is not None:
        grid_size = (args.grid_rows, args.grid_cols)

    # 创建拼图
    create_montage(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        border_width=args.border_width,
        thumbnail_size=tuple(args.thumbnail_size),
        max_images=args.max_images,
        grid_size=grid_size,
        background_color=args.background,
        sort_ascending=not args.descending,
        add_labels=not args.no_labels,
        font_size=args.font_size
    )

if __name__ == "__main__":
    main()
