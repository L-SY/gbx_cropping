import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_and_split_square(image_path, output_folder='output'):
    """
    读取图片，提取正方形区域，并将其分割成四个等大的小正方形

    参数:
        image_path: 图片路径
        output_folder: 输出文件夹
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    # 转换为灰度图，便于处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 按面积排序，找到最大的轮廓（假设是我们的正方形）
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 寻找近似于正方形的轮廓
    square_contour = None
    for contour in contours:
        # 估算轮廓的多边形
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # 如果多边形有4个点，我们认为这可能是我们的正方形
        if len(approx) == 4:
            square_contour = approx
            break

    if square_contour is None:
        print("未能找到合适的正方形轮廓")
        # 可视化找到的所有轮廓，帮助调试
        debug_img = img.copy()
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.title("所有检测到的轮廓")
        plt.show()
        return

    # 绘制找到的正方形轮廓
    contour_img = img.copy()
    cv2.drawContours(contour_img, [square_contour], -1, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    plt.title("检测到的正方形")
    plt.show()

    # 重新排序轮廓点，以便进行透视变换
    points = square_contour.reshape(4, 2)
    rect = order_points(points)

    # 计算新图像的尺寸（10cm对应的像素）
    # 为了简单起见，我们设定每厘米对应50像素
    square_size = 500  # 10cm * 50 pixels/cm = 500 pixels

    # 变换后的目标点
    dst = np.array([
        [0, 0],  # 左上
        [square_size - 1, 0],  # 右上
        [square_size - 1, square_size - 1],  # 右下
        [0, square_size - 1]  # 左下
    ], dtype=np.float32)

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)

    # 应用透视变换
    warped = cv2.warpPerspective(img, M, (square_size, square_size))

    # 显示校正后的正方形
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title("校正后的正方形")
    plt.show()

    # 将正方形分割成四个小正方形
    half_size = square_size // 2

    # 定义四个小正方形区域
    top_left = warped[0:half_size, 0:half_size]
    top_right = warped[0:half_size, half_size:square_size]
    bottom_left = warped[half_size:square_size, 0:half_size]
    bottom_right = warped[half_size:square_size, half_size:square_size]

    # 创建输出文件夹
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 保存四个小正方形
    cv2.imwrite(f"{output_folder}/top_left.jpg", top_left)
    cv2.imwrite(f"{output_folder}/top_right.jpg", top_right)
    cv2.imwrite(f"{output_folder}/bottom_left.jpg", bottom_left)
    cv2.imwrite(f"{output_folder}/bottom_right.jpg", bottom_right)

    # 显示四个小正方形
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(cv2.cvtColor(top_left, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("左上")
    axs[0, 1].imshow(cv2.cvtColor(top_right, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title("右上")
    axs[1, 0].imshow(cv2.cvtColor(bottom_left, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title("左下")
    axs[1, 1].imshow(cv2.cvtColor(bottom_right, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title("右下")
    plt.tight_layout()
    plt.show()

    print(f"处理完成。四个小正方形已保存到 {output_folder} 文件夹。")
    return [top_left, top_right, bottom_left, bottom_right]

def order_points(pts):
    """
    对四边形的四个角点进行排序，使其符合[左上，右上，右下，左下]的顺序
    这对于进行透视变换是必要的
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # 左上角点坐标和最小，右下角点坐标和最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 右上角点差最小，左下角点差最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

# 使用函数
extract_and_split_square("/image_tools/image_train/image_preprocessing/image.png")