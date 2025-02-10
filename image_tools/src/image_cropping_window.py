import cv2
import os
import numpy as np
import shutil

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

def sliding_window_square_search(img, min_size_ratio=0.2, max_size_ratio=0.8, step_ratio=0.1):
    """
    使用滑动窗口搜索正方形区域
    min_size_ratio: 最小窗口尺寸占图片最小边长的比例
    max_size_ratio: 最大窗口尺寸占图片最小边长的比例
    step_ratio: 每次移动的步长占窗口大小的比例
    """
    height, width = img.shape[:2]
    min_dim = min(height, width)

    # 计算最小和最大窗口尺寸
    min_window = int(min_dim * min_size_ratio)
    max_window = int(min_dim * max_size_ratio)

    best_score = -float('inf')
    best_box = None

    # 遍历不同的窗口大小
    for window_size in range(min_window, max_window, int(min_dim * 0.1)):
        step_size = int(window_size * step_ratio)

        # 滑动窗口
        for y in range(0, height - window_size, step_size):
            for x in range(0, width - window_size, step_size):
                # 提取当前窗口
                window = img[y:y + window_size, x:x + window_size]

                # 计算这个窗口的评分
                score = evaluate_square(window)

                if score > best_score:
                    best_score = score
                    best_box = (x, y, window_size, window_size)

    return best_box, best_score

def evaluate_square(window):
    """
    评估窗口中可能包含正方形物体的可能性
    """
    try:
        # 转换为灰度图
        gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)

        # 边缘检测
        edges = auto_canny(gray)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return -float('inf')

        # 计算评分标准
        score = 0
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            if area < 100:  # 忽略太小的轮廓
                continue

            # 计算周长
            perimeter = cv2.arcLength(contour, True)

            # 计算形状复杂度
            complexity = perimeter * perimeter / (4 * np.pi * area) if area > 0 else float('inf')

            # 拟合矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            rect_area = cv2.contourArea(box)

            # 计算矩形度（实际面积与最小外接矩形面积之比）
            rectangularity = area / rect_area if rect_area > 0 else 0

            # 计算宽高比
            w, h = rect[1]
            aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0

            # 计算轮廓与图像边界的距离
            x, y, w, h = cv2.boundingRect(contour)
            border_distance = min(x, y, window.shape[1]-x-w, window.shape[0]-y-h)
            border_score = border_distance / min(window.shape[:2])

            # 综合评分
            current_score = (
                    area * 0.3 +  # 面积权重
                    rectangularity * 0.3 +  # 矩形度权重
                    aspect_ratio * 0.2 +  # 宽高比权重
                    (1.0 / complexity) * 0.1 +  # 形状复杂度权重
                    border_score * 0.1  # 边界距离权重
            )

            score = max(score, current_score)

        return score

    except Exception as e:
        print(f"评估过程中出现错误: {str(e)}")
        return -float('inf')

def find_square_new(img_path, debug_folder=None):
    """
    使用新的方法查找并裁剪正方形区域
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, None

    debug_img = img.copy()
    original_img = img.copy()

    # 使用滑动窗口搜索正方形
    best_box, score = sliding_window_square_search(img)

    if best_box is None or score <= 0:
        return None, debug_img

    x, y, w, h = best_box

    # 在调试图像上画出检测到的区域
    cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 添加边距进行裁剪
    margin = int(min(w, h) * 0.05)  # 边距为边长的5%
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img.shape[1] - x, w + 2*margin)
    h = min(img.shape[0] - y, h + 2*margin)

    cropped = original_img[y:y+h, x:x+w]

    # 保存调试图像
    if debug_folder:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        debug_path = os.path.join(debug_folder, f"{base_name}_detected.jpg")
        cv2.imwrite(debug_path, debug_img)

    return cropped, debug_img

def process_folder(input_folder, output_folder, debug_folder, undetected_folder):
    """
    处理整个文件夹中的图片
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(debug_folder, exist_ok=True)
    os.makedirs(undetected_folder, exist_ok=True)

    total_images = 0
    detected_images = 0

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            total_images += 1
            img_path = os.path.join(input_folder, filename)

            print(f"处理图片: {filename}")
            cropped, debug_img = find_square_new(img_path, debug_folder)

            if cropped is not None:
                detected_images += 1
                output_path = os.path.join(output_folder, f"cropped_{filename}")
                cv2.imwrite(output_path, cropped)
                print(f"成功保存裁剪图片: {output_path}")
            else:
                print(f"未检测到正方形框: {filename}")
                undetected_path = os.path.join(undetected_folder, filename)
                shutil.copy2(img_path, undetected_path)
                print(f"已将未检测图片复制到: {undetected_path}")

    detection_rate = (detected_images / total_images) * 100 if total_images > 0 else 0
    print(f"\n检测总结:")
    print(f"总图片数: {total_images}")
    print(f"成功检测数: {detected_images}")
    print(f"检测率: {detection_rate:.2f}%")

def main():
    input_folder = "/home/lsy/gbx_cropping_ws/src/image_tools/raw_images"
    output_folder = "/home/lsy/gbx_cropping_ws/src/image_tools/cropping"
    debug_folder = "/home/lsy/gbx_cropping_ws/src/image_tools/debug"
    undetected_folder = "/home/lsy/gbx_cropping_ws/src/image_tools/undetected"

    process_folder(input_folder, output_folder, debug_folder, undetected_folder)

if __name__ == "__main__":
    main()