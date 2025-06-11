import cv2
import numpy as np
import matplotlib.pyplot as plt

def check_uniformity(image_path, show_plot=True, std_threshold_ratio=0.05):
    # 1. 读取图像并转为灰度
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("图像读取失败，请检查路径")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 高斯模糊去除噪声（可调）
    gray_blur = cv2.GaussianBlur(gray, (31, 31), 0)

    # 3. 计算统计信息
    mean_val = np.mean(gray_blur)
    std_val = np.std(gray_blur)
    ratio = std_val / mean_val

    # 4. 输出结论
    print(f"亮度均值: {mean_val:.2f}")
    print(f"亮度标准差: {std_val:.2f}")
    print(f"标准差/均值比例: {ratio:.2%}")

    if ratio < std_threshold_ratio:
        print("✅ 发光板光照较为均匀")
    else:
        print("⚠️ 发光板光照存在明显不均匀")

    # 5. 显示热力图（可视化亮度分布）
    if show_plot:
        plt.figure(figsize=(8, 6))
        plt.imshow(gray_blur, cmap='hot')
        plt.title('Brightness heatmap (blurred)')
        plt.colorbar(label='brightness value')
        plt.tight_layout()
        plt.show()

# 示例用法
check_uniformity("/home/lsy/gbx_cropping_ws/src/camera/hk_camera/scripts/calibration_images/luminous_board.png")
