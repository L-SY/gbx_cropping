import cv2
import numpy as np
import os
import argparse

def stitch_images_from_folder(folder_path, output_path, direction='horizontal'):
    # 获取图像文件
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith(valid_exts)]
    image_paths = [os.path.join(folder_path, f) for f in image_files]

    if not image_paths:
        print("❌ 文件夹中未找到图像文件")
        return

    # 加载所有图像
    images = [cv2.imread(p) for p in image_paths]
    if any(img is None for img in images):
        print("❌ 存在无法读取的图像文件")
        return

    # 对齐尺寸后拼接
    if direction == 'horizontal':
        min_height = min(img.shape[0] for img in images)
        resized = [
            cv2.resize(img, (int(img.shape[1] * min_height / img.shape[0]), min_height))
            for img in images
        ]
        stitched = cv2.hconcat(resized)
    elif direction == 'vertical':
        min_width = min(img.shape[1] for img in images)
        resized = [
            cv2.resize(img, (min_width, int(img.shape[0] * min_width / img.shape[1])))
            for img in images
        ]
        stitched = cv2.vconcat(resized)
    else:
        print("❌ 参数 direction 只能为 'horizontal' 或 'vertical'")
        return

    cv2.imwrite(output_path, stitched)
    print(f"✅ 拼接完成，保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch cropped images in a folder")
    parser.add_argument('--folder', type=str, required=True, help='Path to folder containing images')
    parser.add_argument('--output', type=str, default='stitched_result.jpg', help='Path to save stitched image')
    parser.add_argument('--direction', type=str, default='horizontal', choices=['horizontal', 'vertical'],
                        help='Stitching direction: horizontal or vertical')

    args = parser.parse_args()
    stitch_images_from_folder(args.folder, args.output, args.direction)
