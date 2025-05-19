import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_light_panel(image_path, output_path=None):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用阈值分割找出亮区域
    # 使用OTSU算法自动确定阈值
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 找出所有轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果没有找到轮廓，尝试反转阈值再找一次
    if not contours:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("未能找到任何轮廓")
        return None

    # 找出最大的轮廓（假设这是发光板）
    max_contour = max(contours, key=cv2.contourArea)

    # 获取最小外接矩形
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 获取矩形的宽和高
    width = int(rect[1][0])
    height = int(rect[1][1])

    # 确保宽高比与预期的发光板接近
    aspect_ratio = max(width, height) / min(width, height)
    expected_ratio = 20 / 20  # 预期宽高比为1

    if abs(aspect_ratio - expected_ratio) > 0.3:  # 允许一定误差
        print(f"警告: 检测到的形状宽高比与预期不符 (检测到: {aspect_ratio:.2f}, 预期: {expected_ratio:.2f})")

    # 获取透视变换的源点和目标点
    src_pts = box.astype("float32")

    # 确保点的顺序是：左上，右上，右下，左下
    # 按照y坐标排序
    src_pts = sorted(src_pts, key=lambda pt: pt[1])
    # 前两个点按照x坐标排序（确定左上和右上）
    top_pts = sorted(src_pts[:2], key=lambda pt: pt[0])
    # 后两个点按照x坐标排序（确定左下和右下）
    bottom_pts = sorted(src_pts[2:], key=lambda pt: pt[0])
    # 组合排序后的点
    src_pts = np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype="float32")

    # 目标尺寸（正方形）
    dst_size = max(width, height)
    dst_pts = np.array([
        [0, 0],
        [dst_size - 1, 0],
        [dst_size - 1, dst_size - 1],
        [0, dst_size - 1]
    ], dtype="float32")

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 应用透视变换
    warped = cv2.warpPerspective(image, M, (dst_size, dst_size))

    # 保存结果
    if output_path:
        cv2.imwrite(output_path, warped)
        print(f"已保存裁剪后的图像到: {output_path}")

    # 可视化处理过程
    plt.figure(figsize=(15, 10))

    plt.subplot(221)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.subplot(222)
    plt.title("Threshold")
    plt.imshow(thresh, cmap='gray')

    plt.subplot(223)
    contour_img = image.copy()
    cv2.drawContours(contour_img, [box], 0, (0, 255, 0), 2)
    plt.title("Detected Light Panel")
    plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))

    plt.subplot(224)
    plt.title("Cropped Light Panel")
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()

    return warped

# 使用示例
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python script.py <输入图像路径> [输出图像路径]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "cropped_light_panel.jpg"

    extract_light_panel(input_path, output_path)