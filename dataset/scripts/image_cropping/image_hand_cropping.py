import cv2
import os
import numpy as np
import argparse

class ImageCropper:
    def __init__(self):
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.square_size = 0
        self.image = None
        self.original_image = None
        self.current_rect = None

    def draw_square(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # 创建图像副本
                temp_img = self.original_image.copy()
                # 计算正方形边长
                size = max(abs(x - self.ix), abs(y - self.iy))
                # 确保正方形不会超出图像边界
                size = min(size,
                           self.image.shape[1] - self.ix,
                           self.image.shape[0] - self.iy)
                # 画正方形
                cv2.rectangle(temp_img, (self.ix, self.iy),
                              (self.ix + size, self.iy + size), (0, 255, 0), 2)
                self.image = temp_img
                self.current_rect = (self.ix, self.iy, size)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            # 最终的正方形
            size = max(abs(x - self.ix), abs(y - self.iy))
            size = min(size,
                       self.image.shape[1] - self.ix,
                       self.image.shape[0] - self.iy)
            cv2.rectangle(self.image, (self.ix, self.iy),
                          (self.ix + size, self.iy + size), (0, 255, 0), 2)
            self.current_rect = (self.ix, self.iy, size)

def process_images(input_folder, output_folder=None):
    """处理文件夹中的所有图片"""
    # 如果未指定输出文件夹，则使用输入文件夹的父目录下的"cropping"文件夹
    if output_folder is None:
        parent_dir = os.path.dirname(os.path.abspath(input_folder))
        output_folder = os.path.join(parent_dir, "cropping")

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    print("操作指南:")
    print("  - 单击并拖动鼠标以选择正方形区域")
    print("  - 按 's' 保存裁剪的图像并继续下一张")
    print("  - 按 'r' 重置当前选择")
    print("  - 按 'n' 跳过当前图像")
    print("  - 按 'q' 退出程序")

    # 创建裁剪器实例
    cropper = ImageCropper()

    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"错误: 在 {input_folder} 中没有找到图像文件!")
        return

    print(f"找到 {len(image_files)} 个图像文件")

    # 设置窗口
    window_name = 'Image Cropper'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, cropper.draw_square)

    for idx, filename in enumerate(image_files):
        # 读取图片
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {filename}")
            continue

        # 初始化裁剪器
        cropper.image = image.copy()
        cropper.original_image = image.copy()
        cropper.current_rect = None

        # 显示进度
        print(f"\n处理图片 {idx+1}/{len(image_files)}: {filename}")

        while True:
            # 显示图片
            cv2.imshow(window_name, cropper.image)

            # 等待键盘输入
            key = cv2.waitKey(1) & 0xFF

            # 's' 保存当前裁剪
            if key == ord('s'):
                if cropper.current_rect is not None:
                    x, y, size = cropper.current_rect
                    # 保存裁剪后的图片
                    output_path = os.path.join(output_folder, f'cropped_{filename}')
                    cropped = image[y:y+size, x:x+size]
                    cv2.imwrite(output_path, cropped)
                    print(f"已保存: {output_path}")
                    break
                else:
                    print("请先选择一个区域再保存")

            # 'r' 重置当前图片
            elif key == ord('r'):
                cropper.image = image.copy()
                cropper.current_rect = None
                print("已重置选择")

            # 'q' 退出程序
            elif key == ord('q'):
                print("程序退出")
                cv2.destroyAllWindows()
                return

            # 'n' 跳过当前图片
            elif key == ord('n'):
                print(f"跳过图片: {filename}")
                break

    print("\n所有图片处理完成!")
    cv2.destroyAllWindows()

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='手动裁剪图像中的正方形区域')
    parser.add_argument('--input', type=str, required=True, help='输入图像文件夹路径')
    parser.add_argument('--output', type=str, help='输出文件夹路径 (默认: 与输入文件夹同级的"cropping"文件夹)')

    # 解析命令行参数
    args = parser.parse_args()

    # 处理文件夹
    process_images(args.input, args.output)

if __name__ == "__main__":
    main()