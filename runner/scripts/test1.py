import cv2
import numpy as np
import matplotlib.pyplot as plt

def sort_points(pts):
    """
    对四个点进行排序，顺序为左上、右上、右下、左下
    """
    pts = np.array(pts)
    if pts.shape != (4,2):
        raise ValueError("需要四个点进行排序")
    # 计算每个点的和与差
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    # 左上点有最小的和，右下点有最大的和
    left_top = pts[np.argmin(s)]
    right_bottom = pts[np.argmax(s)]

    # 右上点有最小的差，左下点有最大的差
    right_top = pts[np.argmin(diff)]
    left_bottom = pts[np.argmax(diff)]

    return [left_top, right_top, right_bottom, left_bottom]

def warp_perspective(image, pts, width=500, height=500):
    """
    对四个点进行透视变换，校正为标准的矩形
    """
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    pts = np.array(pts, dtype="float32")

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(pts, dst)

    # 执行透视变换
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped

def detect_and_crop(image_path):
    """
    检测图像中的四个黑色圆形，裁切图像，仅保留四边形内部部分
    返回裁切后的图像和四个圆心的坐标
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像文件: {image_path}")
        return None, None

    # 复制原图用于绘制结果
    output_image = image.copy()

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊以减少噪声
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 使用自适应阈值将黑色圆形点提取出来
    # 由于是黑色点，使用反转的阈值
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # 使用形态学操作去除噪声
    kernel = np.ones((2,2), np.uint8)
    thresh_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh_dilate = cv2.morphologyEx(thresh_close, cv2.MORPH_DILATE, kernel, iterations=3)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    detected_circles_image = output_image.copy()

    for cnt in contours:
        # 计算轮廓面积，滤除过小或过大的区域
        area = cv2.contourArea(cnt)
        if area < 3000 or area > 500000:  # 根据图像实际情况调整
            continue

        # 计算轮廓的最小外接圆
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        # 计算轮廓的圆度以滤除非圆形
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.1:  # 圆度阈值，根据需要调整
            continue

        centers.append(center)

        # 绘制圆形边界
        cv2.circle(detected_circles_image, center, radius, (0, 255, 0), 2)
        # 绘制中心点
        cv2.circle(detected_circles_image, center, 5, (0, 0, 255), -1)

    # 确保找到四个圆心
    if len(centers) != 4:
        print(f"检测到的圆心数量为 {len(centers)}，需要恰好四个圆心。")
        return None, centers

    # 排序圆心
    sorted_centers = sort_points(centers)

    # 绘制连接线
    for i in range(4):
        start_point = tuple(sorted_centers[i])
        end_point = tuple(sorted_centers[(i + 1) % 4])
        cv2.line(detected_circles_image, start_point, end_point, (255, 0, 0), 2)

    # 可视化连接后的图像
    plt.figure(figsize=(10, 10))
    plt.title('Detected Circles with Connections')
    plt.imshow(cv2.cvtColor(detected_circles_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # # 透视变换裁切
    warped_image = image
    # warped_image = warp_perspective(image, sorted_centers, width=500, height=500)

    # # 可视化透视变换后的图像
    # plt.figure(figsize=(10, 10))
    # plt.title('Warped (Perspective Transformed) Image')
    # plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()

    # 保存裁切后的图像
    cropped_output_path = image_path.replace('.', '_cropped.')
    cv2.imwrite(cropped_output_path, warped_image)
    print(f"裁切后的图像已保存至: {cropped_output_path}")

    # 输出所有圆心坐标
    print("检测到的圆心坐标（以图像左上角为原点，单位为像素）：")
    for idx, center in enumerate(sorted_centers, start=1):
        print(f"圆心 {idx}: (x={center[0]}, y={center[1]})")

    return warped_image, sorted_centers

def stitch_images(image1, image2):
    """
    将两张裁切后的图像拼接成一张完整的图像
    """
    # 使用OpenCV的Stitcher进行拼接
    # 注意：Stitcher在OpenCV 4.x中使用cv2.Stitcher_create()
    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch([image1, image2])

    if status == cv2.Stitcher_OK:
        print("图像拼接成功！")
        # 可视化拼接后的图像
        plt.figure(figsize=(15, 15))
        plt.title('Stitched Image')
        plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        return stitched
    else:
        print(f"图像拼接失败，错误码: {status}")
        return None

def main():
    # 替换为你的两张图像路径
    image_paths = ['gbx_cropping_a.jpg', 'gbx_cropping_b.jpg']
    cropped_images = []

    for path in image_paths:
        cropped_image, centers = detect_and_crop(path)
        if cropped_image is not None:
            cropped_images.append(cropped_image)
        else:
            print(f"图像 {path} 未成功裁切，跳过拼接。")

    if len(cropped_images) < 2:
        print("需要两张成功裁切的图像才能进行拼接。")
        return

    # 拼接两张裁切后的图像
    stitched_image = stitch_images(cropped_images[0], cropped_images[1])

    if stitched_image is not None:
        # 保存拼接后的图像
        stitched_output_path = 'stitched_image.jpg'
        cv2.imwrite(stitched_output_path, stitched_image)
        print(f"拼接后的图像已保存至: {stitched_output_path}")
    else:
        print("没有生成拼接后的图像。")

if __name__ == "__main__":
    main()