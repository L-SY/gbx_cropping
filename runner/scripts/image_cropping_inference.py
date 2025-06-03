import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import math
import torch
import torch.nn as nn
from torchvision import models, transforms
import argparse

class InnerBlackBorderAdder(object):
    def __init__(self, border_percentage=0.05):
        self.border_percentage = border_percentage

    def __call__(self, img):
        width, height = img.size
        bordered_img = img.copy()
        draw = ImageDraw.Draw(bordered_img)

        # 根据图片尺寸计算边框宽度
        border_width_h = int(width * self.border_percentage)  # 水平边框宽度
        border_width_v = int(height * self.border_percentage)  # 垂直边框宽度

        # 绘制四个边的黑色边框
        draw.rectangle([(0, 0), (width, border_width_v)], fill="black")  # 上边
        draw.rectangle([(0, height - border_width_v), (width, height)], fill="black")  # 下边
        draw.rectangle([(0, 0), (border_width_h, height)], fill="black")  # 左边
        draw.rectangle([(width - border_width_h, 0), (width, height)], fill="black")  # 右边

        return bordered_img

# 模型定义函数
def get_model(model_name: str, dropout_rate: float, freeze_backbone: bool = False):
    """
    获取预训练模型并修改最后一层用于回归任务
    """
    model_name = model_name.lower()
    model_func = getattr(models, model_name, None)
    if model_func is None:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model_func(weights=None)

    # 冻结backbone参数
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # 修改最后一层
    if 'densenet' in model_name:
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )
    else:
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

    return model

def predict_density(model, image_path, transform, device):
    """
    对单张图片进行密度预测
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            model.eval()
            prediction = model(image_tensor).item()

        return prediction
    except Exception as e:
        print(f"预测图片 {image_path} 时出错: {e}")
        return None

def image_cropping(image_path, output_folder, actual_width_cm=10, crop_size_cm=5, start_from_left=False,
                   add_border=True, border_percentage=0.05, model_path=None, model_name='resnet50',
                   dropout_rate=0.5, freeze_backbone=False, image_size=224):
    """
    图片裁切工具

    参数:
    - image_path: 输入图片路径
    - output_folder: 输出文件夹路径
    - actual_width_cm: 图片实际宽度(厘米)
    - crop_size_cm: 每个裁切区域的大小(厘米)
    - start_from_left: 是否从左边开始裁切区域 (True: 从图片左边开始裁切, False: 从图片右边开始裁切)
    - add_border: 是否为每个裁切图片添加内边框 (True: 添加, False: 不添加)
    - border_percentage: 内边框宽度百分比 (0.05 = 5%)
    - model_path: 预训练模型路径 (如果提供则进行密度预测)
    - model_name: 模型名称 (resnet50, densenet121等)
    - dropout_rate: Dropout率
    - freeze_backbone: 是否冻结backbone
    - image_size: 模型输入图片尺寸
    """

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return

    height, width = image.shape[:2]
    print(f"原图尺寸: {width} x {height} 像素")

    # 判断哪个是短边（宽度）
    if width < height:
        # 横向是短边，作为实际宽度
        actual_short_side = width
        actual_long_side = height
        print(f"检测到竖长图片，短边(宽度): {width}像素，长边(高度): {height}像素")
    else:
        # 纵向是短边，作为实际宽度
        actual_short_side = height
        actual_long_side = width
        print(f"检测到横长图片，短边(宽度): {height}像素，长边(高度): {width}像素")

    # 计算像素与厘米的比例（基于短边）
    pixels_per_cm = actual_short_side / actual_width_cm
    print(f"像素密度: {pixels_per_cm:.2f} 像素/厘米")

    # 计算实际长边尺寸
    actual_long_side_cm = actual_long_side / pixels_per_cm

    if width < height:
        print(f"图片实际尺寸: {actual_width_cm} x {actual_long_side_cm:.2f} 厘米 (宽x高)")
        actual_height_cm = actual_long_side_cm
    else:
        print(f"图片实际尺寸: {actual_long_side_cm:.2f} x {actual_width_cm} 厘米 (宽x高)")
        actual_height_cm = actual_width_cm

    # 计算裁切区域的像素大小
    crop_size_pixels = int(crop_size_cm * pixels_per_cm)
    print(f"每个裁切区域: {crop_size_pixels} x {crop_size_pixels} 像素")

    # 计算可以裁切的行数和列数
    if width < height:
        # 竖长图片：宽度是短边
        available_width = width
        available_height = height
        cols = int(actual_width_cm / crop_size_cm)  # 列数（水平方向）
        rows = int(actual_long_side_cm / crop_size_cm)  # 行数（垂直方向）
    else:
        # 横长图片：高度是短边
        available_width = width
        available_height = height
        cols = int(actual_long_side_cm / crop_size_cm)  # 列数（水平方向）
        rows = int(actual_width_cm / crop_size_cm)  # 行数（垂直方向）

    # 计算实际使用的像素区域
    used_width = cols * crop_size_pixels
    used_height = rows * crop_size_pixels

    # 计算剩余像素
    remaining_width = available_width - used_width
    remaining_height = available_height - used_height

    print(f"图片总尺寸: {available_width} x {available_height} 像素")
    print(f"使用区域: {used_width} x {used_height} 像素")
    print(f"剩余像素: 宽度{remaining_width}像素, 高度{remaining_height}像素")

    # 计算裁切起始偏移量
    if start_from_left:
        # 从左边开始，剩余像素留在右边
        offset_x = 0
        print(f"从左边开始裁切，剩余{remaining_width}像素留在右边")
    else:
        # 从右边开始，剩余像素留在左边
        offset_x = remaining_width
        print(f"从右边开始裁切，剩余{remaining_width}像素留在左边")

    # 垂直方向始终从上开始（也可以添加参数控制）
    offset_y = 0

    print(f"可裁切区域: {rows} 行 x {cols} 列 = {rows * cols} 个区域")
    print(f"裁切起始位置: {'从图片左边开始' if start_from_left else '从图片右边开始'}")
    print(f"编号顺序: 从右上开始向左")
    print(f"内边框处理: {'添加' if add_border else '不添加'}{f' (宽度{border_percentage*100:.1f}%)' if add_border else ''}")

    # 初始化模型和预处理
    model = None
    transform = None
    device = None

    if model_path and os.path.exists(model_path):
        print(f"加载模型: {model_path}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        # 创建模型
        model = get_model(model_name, dropout_rate, freeze_backbone).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        # 创建图像预处理
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.220116, 0.220116, 0.220116],
                std=[0.178257, 0.178257, 0.178257]
            )
        ])
        print("模型加载成功，将进行密度预测")
    else:
        print("未提供模型路径或模型文件不存在，跳过密度预测")

    # 创建边框处理器
    if add_border:
        border_adder = InnerBlackBorderAdder(border_percentage)

    # 创建标号图
    labeled_image = image.copy()

    # 设置字体（尝试使用系统字体，如果没有则使用默认字体）
    try:
        # 对于中文支持，可以指定中文字体路径
        font_size = max(20, crop_size_pixels // 10)
    except:
        font_size = max(20, crop_size_pixels // 10)

    # 裁切图片并编号
    cropped_images = []
    processed_crops = []  # 存储处理后的裁切图片用于重组

    for row in range(rows):
        for col in range(cols):
            # 计算裁切区域的坐标（加上偏移量）
            x1 = offset_x + col * crop_size_pixels
            y1 = offset_y + row * crop_size_pixels
            x2 = x1 + crop_size_pixels
            y2 = y1 + crop_size_pixels

            # 确保不超出图片边界
            x2 = min(x2, width)
            y2 = min(y2, height)

            # 编号规则：从右上开始向左（与原需求一致）
            numbering_col = cols - 1 - col
            crop_number = row * cols + numbering_col + 1

            # 生成文件名
            crop_filename = f"crop_{crop_number:03d}.jpg"

            # 裁切图片
            cropped = image[y1:y2, x1:x2]

            # 处理裁切的图片（添加内边框）
            if add_border:
                # 将OpenCV图片转换为PIL格式
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cropped_rgb)

                # 添加内边框
                bordered_img = border_adder(pil_img)

                # 转换回OpenCV格式保存
                bordered_array = np.array(bordered_img)
                bordered_cv = cv2.cvtColor(bordered_array, cv2.COLOR_RGB2BGR)

                # 保存带边框的图片
                crop_path = os.path.join(output_folder, crop_filename)
                cv2.imwrite(crop_path, bordered_cv)

                # 存储处理后的图片用于重组
                processed_crops.append({
                    'image': bordered_cv,
                    'row': row,
                    'col': col,
                    'number': crop_number,
                    'filename': crop_filename
                })
            else:
                # 直接保存原始裁切图片
                crop_path = os.path.join(output_folder, crop_filename)
                cv2.imwrite(crop_path, cropped)

                # 存储原始裁切图片用于重组
                processed_crops.append({
                    'image': cropped,
                    'row': row,
                    'col': col,
                    'number': crop_number,
                    'filename': crop_filename
                })

            # 进行密度预测
            prediction = None
            if model is not None:
                prediction = predict_density(model, crop_path, transform, device)
                if prediction is not None:
                    print(f"图片 {crop_filename} 预测密度: {prediction:.2f}")

            # 更新processed_crops中的预测结果
            processed_crops[-1]['prediction'] = prediction

            # 在标号图上绘制边框和编号
            cv2.rectangle(labeled_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # 计算文字位置（居中）
            text = str(crop_number)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = crop_size_pixels / 200.0  # 根据区域大小调整字体大小
            thickness = max(1, int(crop_size_pixels / 100))

            # 获取文字大小
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

            # 计算文字居中位置
            text_x = x1 + (crop_size_pixels - text_width) // 2
            text_y = y1 + (crop_size_pixels + text_height) // 2

            # 绘制白色背景
            cv2.rectangle(labeled_image,
                          (text_x - 5, text_y - text_height - 5),
                          (text_x + text_width + 5, text_y + 5),
                          (255, 255, 255), -1)

            # 绘制黑色文字
            cv2.putText(labeled_image, text, (text_x, text_y),
                        font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            cropped_images.append({
                'number': crop_number,
                'position': (row, col),
                'coordinates': (x1, y1, x2, y2),
                'filename': crop_filename,
                'prediction': prediction
            })

    # 保存标号图
    labeled_image_path = os.path.join(output_folder, "labeled_image.jpg")
    cv2.imwrite(labeled_image_path, labeled_image)

    # 生成由裁切图片重组的完整图像
    print("正在生成重组完整图像...")
    reassembled_image = create_reassembled_image(processed_crops, rows, cols, crop_size_pixels)
    reassembled_path = os.path.join(output_folder, "reassembled_image.jpg")
    cv2.imwrite(reassembled_path, reassembled_image)

    # 生成信息文件
    info_file = os.path.join(output_folder, "crop_info.txt")
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"图片裁切信息\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"原图路径: {image_path}\n")
        f.write(f"原图尺寸: {width} x {height} 像素\n")
        if width < height:
            f.write(f"实际尺寸: {actual_width_cm} x {actual_long_side_cm:.2f} 厘米 (宽x高)\n")
        else:
            f.write(f"实际尺寸: {actual_long_side_cm:.2f} x {actual_width_cm} 厘米 (宽x高)\n")
        f.write(f"像素密度: {pixels_per_cm:.2f} 像素/厘米\n")
        f.write(f"裁切区域大小: {crop_size_cm} x {crop_size_cm} 厘米 ({crop_size_pixels} x {crop_size_pixels} 像素)\n")
        f.write(f"总裁切数量: {len(cropped_images)} 个\n")
        f.write(f"排列方式: {rows} 行 x {cols} 列\n")
        f.write(f"裁切起始位置: {'从图片左边开始' if start_from_left else '从图片右边开始'}\n")
        f.write(f"剩余像素位置: {'右边' if start_from_left else '左边'}\n")
        f.write(f"编号顺序: 从右上开始向左\n")
        f.write(f"内边框处理: {'添加' if add_border else '不添加'}{f' (宽度{border_percentage*100:.1f}%)' if add_border else ''}\n\n")

        f.write("裁切详情:\n")
        f.write("-" * 50 + "\n")
        for crop_info in cropped_images:
            f.write(f"编号 {crop_info['number']:3d}: {crop_info['filename']} ")
            f.write(f"位置({crop_info['position'][0]+1}, {crop_info['position'][1]+1}) ")
            f.write(f"坐标({crop_info['coordinates'][0]}, {crop_info['coordinates'][1]}, ")
            f.write(f"{crop_info['coordinates'][2]}, {crop_info['coordinates'][3]}) ")
            if crop_info['prediction'] is not None:
                f.write(f"预测密度: {crop_info['prediction']:.2f}")
            f.write("\n")

    print(f"\n裁切完成!")
    print(f"总共生成 {len(cropped_images)} 个裁切图片")
    print(f"标号图保存为: {labeled_image_path}")
    print(f"重组图保存为: {reassembled_path}")
    print(f"详细信息保存为: {info_file}")
    print(f"所有文件保存在: {output_folder}")

def create_reassembled_image(processed_crops, rows, cols, crop_size_pixels):
    """
    根据裁切的图片重新组装成完整图像，并添加编号和预测结果
    """
    # 创建空白画布
    canvas_height = rows * crop_size_pixels
    canvas_width = cols * crop_size_pixels
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # 按位置放置每个裁切图片
    for crop_data in processed_crops:
        crop_img = crop_data['image']
        row = crop_data['row']
        col = crop_data['col']
        number = crop_data['number']
        prediction = crop_data.get('prediction', None)

        # 计算在画布上的位置
        y1 = row * crop_size_pixels
        x1 = col * crop_size_pixels
        y2 = y1 + crop_img.shape[0]
        x2 = x1 + crop_img.shape[1]

        # 确保不超出画布边界
        y2 = min(y2, canvas_height)
        x2 = min(x2, canvas_width)
        crop_h = y2 - y1
        crop_w = x2 - x1

        # 将裁切图片放置到画布上
        canvas[y1:y2, x1:x2] = crop_img[:crop_h, :crop_w]

        # 添加文字标注（编号和预测结果）
        # 转换为PIL图像以便添加文字
        canvas_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(canvas_pil)

        try:
            font_size = max(16, crop_size_pixels // 15)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # 准备文字内容
        text_lines = [f"#{number}"]
        if prediction is not None:
            text_lines.append(f"density: {prediction:.1f}")

        # 计算文字位置（左上角）
        text_x = x1 + 5
        text_y = y1 + 5

        # 绘制文字背景和文字
        for i, text in enumerate(text_lines):
            current_y = text_y + i * (font_size + 2)

            # 获取文字边界框
            bbox = draw.textbbox((text_x, current_y), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # 绘制半透明背景
            draw.rectangle([text_x-2, current_y-2, text_x+text_width+2, current_y+text_height+2],
                           fill=(0, 0, 0, 180))

            # 绘制文字
            draw.text((text_x, current_y), text, fill=(255, 255, 255), font=font)

        # 转换回OpenCV格式
        canvas = cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)

    return canvas

# 使用示例
if __name__ == "__main__":
    # 请修改以下路径
    input_image_path = "/home/lsy/gbx_cropping_ws/src/runner/scripts/four_test/image.jpg"  # 输入图片路径
    output_directory = "/home/lsy/gbx_cropping_ws/src/runner/scripts/four_test/cropping"   # 输出文件夹

    # 执行裁切
    image_cropping(
        image_path=input_image_path,
        output_folder=output_directory,
        actual_width_cm=10,         # 图片实际宽度10厘米
        crop_size_cm=5,             # 每个区域5x5厘米
        start_from_left=False,      # True: 从左开始, False: 从右开始
        add_border=True,            # 是否添加内边框
        border_percentage=0.1,     # 内边框宽度百分比 (5%)
        model_path="/home/lsy/gbx_cropping_ws/src/paper/output/resnet50_change_gray_mean_std/best_model.pth",            # 模型路径，如果需要预测请设置
        model_name='resnet50',      # 模型名称
        dropout_rate=0.5,           # Dropout率
        freeze_backbone=False,      # 是否冻结backbone
        image_size=224              # 模型输入图片尺寸
    )

    # print("\n使用说明:")
    # print("1. 修改 input_image_path 为你的图片路径")
    # print("2. 修改 output_directory 为输出文件夹路径")
    # print("3. 设置 start_from_left 参数:")
    # print("   - False: 从图片右边开始裁切，剩余像素留在左边")
    # print("   - True: 从图片左边开始裁切，剩余像素留在右边")
    # print("4. 设置 add_border 参数:")
    # print("   - True: 为每个裁切图片添加黑色内边框")
    # print("   - False: 不添加边框")
    # print("5. 设置 border_percentage 参数: 内边框宽度百分比 (0.05 = 5%)")
    # print("6. 如需密度预测，请设置以下参数:")
    # print("   - model_path: 训练好的模型文件路径 (.pth)")
    # print("   - model_name: 模型架构名称 (resnet50, densenet121等)")
    # print("   - dropout_rate: Dropout率")
    # print("   - freeze_backbone: 是否冻结backbone权重")
    # print("   - image_size: 模型输入图片尺寸")
    # print("7. 运行脚本即可完成裁切并生成带预测结果的重组图像")