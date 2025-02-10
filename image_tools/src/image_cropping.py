import cv2
import os
import numpy as np
import shutil

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

def find_square(img_path, debug_folder=None):
    img = cv2.imread(img_path)
    if img is None:
        return None, None

    debug_img = img.copy()
    original_img = img.copy()

    height, width = img.shape[:2]
    min_area = (width * height) * 0.05  # 最小面积为图像面积的5%
    max_area = (width * height) * 0.95  # 最大面积为图像面积的95%

    # 多个预处理方法
    preprocessing_methods = [
        # 方法1: 基础预处理
        lambda img: cv2.cvtColor(cv2.GaussianBlur(img, (7, 7), 0), cv2.COLOR_BGR2GRAY),
        # 方法2: 增强对比度
        lambda img: cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
        # 方法3: 中值滤波
        lambda img: cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5),
    ]

    # 多个阈值方法
    threshold_methods = [
        # 方法1: Otsu
        lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        # 方法2: 自适应阈值
        lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2),
        # 方法3: 简单阈值
        lambda img: cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)[1],
    ]

    best_contour = None
    max_score = 0

    for preprocess in preprocessing_methods:
        gray = preprocess(img)

        for threshold in threshold_methods:
            try:
                thresh = threshold(gray)

                # 形态学操作
                kernels = [
                    np.ones((3,3), np.uint8),
                    np.ones((5,5), np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                ]

                for kernel in kernels:
                    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
                    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

                    # 尝试不同的边缘检测参数
                    edges_list = [
                        auto_canny(morph),
                        cv2.Canny(morph, 30, 150),
                        cv2.Canny(morph, 50, 200)
                    ]

                    for edges in edges_list:
                        # 查找轮廓
                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)

                        for contour in contours:
                            area = cv2.contourArea(contour)

                            # 面积过滤
                            if area < min_area or area > max_area:
                                continue

                            # 尝试不同的轮廓近似参数
                            for eps_factor in [0.01, 0.02, 0.03, 0.04, 0.05]:
                                peri = cv2.arcLength(contour, True)
                                approx = cv2.approxPolyDP(contour, eps_factor * peri, True)

                                # 接受4-6个顶点的轮廓
                                if 4 <= len(approx) <= 6:
                                    x, y, w, h = cv2.boundingRect(approx)
                                    aspect_ratio = float(w)/h

                                    # 更宽松的宽高比要求
                                    if 0.5 <= aspect_ratio <= 2.0:
                                        # 计算得分
                                        rect_area = w * h
                                        extent = float(area) / rect_area
                                        # 根据多个因素计算综合得分
                                        score = (area * extent *
                                                 (1 - abs(1 - aspect_ratio)) *
                                                 (1 + 1.0/len(approx)))

                                        if score > max_score:
                                            max_score = score
                                            best_contour = approx
            except Exception as e:
                continue

    if best_contour is not None:
        # 获取边界矩形
        x, y, w, h = cv2.boundingRect(best_contour)

        # 扩大裁剪区域，确保不会裁掉边缘
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(width - x, w + 2*margin)
        h = min(height - y, h + 2*margin)

        # 在调试图像上画出检测到的轮廓
        cv2.drawContours(debug_img, [best_contour], -1, (0, 255, 0), 2)
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 裁剪图像
        cropped = original_img[y:y+h, x:x+w]

        # 保存调试图像
        if debug_folder:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            debug_path = os.path.join(debug_folder, f"{base_name}_detected.jpg")
            cv2.imwrite(debug_path, debug_img)

        return cropped, debug_img

    return None, debug_img

def process_folder(input_folder, output_folder, debug_folder, undetected_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(debug_folder, exist_ok=True)
    os.makedirs(undetected_folder, exist_ok=True)

    total_images = 0
    detected_images = 0

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            total_images += 1
            img_path = os.path.join(input_folder, filename)

            print(f"处理图片: {filename}")
            cropped, debug_img = find_square(img_path, debug_folder)

            if cropped is not None:
                detected_images += 1
                output_path = os.path.join(output_folder, f"cropped_{filename}")
                cv2.imwrite(output_path, cropped)
            else:
                print(f"未检测到正方形框: {filename}")
                # 将未检测到的图片复制到undetected文件夹
                undetected_path = os.path.join(undetected_folder, filename)
                shutil.copy2(img_path, undetected_path)

    detection_rate = (detected_images / total_images) * 100 if total_images > 0 else 0
    print(f"\n检测总结:")
    print(f"总图片数: {total_images}")
    print(f"成功检测数: {detected_images}")
    print(f"检测率: {detection_rate:.2f}%")

if __name__ == "__main__":
    input_folder = "/home/lsy/gbx_cropping_ws/src/image_tools/raw_images"
    output_folder = "/home/lsy/gbx_cropping_ws/src/image_tools/cropping"
    debug_folder = "/home/lsy/gbx_cropping_ws/src/image_tools/debug"
    undetected_folder = "/home/lsy/gbx_cropping_ws/src/image_tools/undetected"

    process_folder(input_folder, output_folder, debug_folder, undetected_folder)