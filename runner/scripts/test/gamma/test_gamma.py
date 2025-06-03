import cv2
import numpy as np
import argparse

def darken_shadows(img: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """
    对图像进行 Gamma 校正，使暗部更暗。

    参数：
        img:      输入图像（BGR 格式，0–255）。
        gamma:    Gamma 值，默认为 2.0 (>1 会使暗部更暗)。

    返回：
        处理后的图像。
    """
    # 构建查找表：LUT[i] = 255 * (i/255) ** gamma
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in range(256)], dtype="uint8")
    # 应用查找表
    return cv2.LUT(img, table)

def main():
    parser = argparse.ArgumentParser(description="Darken shadows in an image via gamma correction.")
    parser.add_argument("-i", "--input",  required=True,
                        help="输入图像路径")
    parser.add_argument("-o", "--output", required=True,
                        help="输出图像路径")
    parser.add_argument("-g", "--gamma",  type=float, default=2.0,
                        help="Gamma 值 (>1 暗部更暗，<1 暗部更亮)，默认为 2.0")
    args = parser.parse_args()

    # 读取图像
    img = cv2.imread(args.input)
    if img is None:
        print(f"无法读取图像：{args.input}")
        return

    # 处理
    result = darken_shadows(img, gamma=args.gamma)

    # 保存结果
    cv2.imwrite(args.output, result)
    print(f"已保存处理后的图像到：{args.output}")

if __name__ == "__main__":
    main()
