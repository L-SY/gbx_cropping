import cv2
import os
import numpy as np

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

def process_images(input_folder, output_folder):
    """处理文件夹中的所有图片"""
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 创建裁剪器实例
    cropper = ImageCropper()

    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

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

            # 'r' 重置当前图片
            elif key == ord('r'):
                cropper.image = image.copy()
                cropper.current_rect = None

            # 'q' 退出程序
            elif key == ord('q'):
                print("程序退出")
                cv2.destroyAllWindows()
                return

            # 'n' 跳过当前图片
            elif key == ord('n'):
                print(f"跳过图片: {filename}")
                break

    cv2.destroyAllWindows()

def main():
    input_folder = "/home/lsy/gbx_cropping_ws/src/image_tools/second_batch_of_samples/undetected"
    output_folder = "/home/lsy/gbx_cropping_ws/src/image_tools/second_batch_of_samples/cropping"

    process_images(input_folder, output_folder)

if __name__ == "__main__":
    main()