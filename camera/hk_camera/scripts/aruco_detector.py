import cv2
import numpy as np
import argparse
from pdf2image import convert_from_path
import os

def load_pdf_as_image(pdf_path):
    # 将 PDF 转为图像（只处理第一页）
    pages = convert_from_path(pdf_path, dpi=300)
    image = np.array(pages[0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def detect_charuco_tags(image, marker_dict=cv2.aruco.DICT_4X4_50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(marker_dict)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        print(f"检测到 {len(ids)} 个标签：", ids.flatten())
    else:
        print("未检测到任何标签")

    vis = image.copy()
    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    return vis

def save_result_image(image, pdf_path):
    # 输出路径 = 输入 PDF 路径 + "_detected.png"
    base_path = os.path.splitext(pdf_path)[0]
    save_path = base_path + "_detected.png"
    cv2.imwrite(save_path, image)
    print(f"检测结果已保存至: {save_path}")

def main(pdf_path):
    print(f"读取 PDF 文件: {pdf_path}")
    image = load_pdf_as_image(pdf_path)
    vis_img = detect_charuco_tags(image)
    save_result_image(vis_img, pdf_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="PDF 文件路径，例如 charuco_board.pdf")
    args = parser.parse_args()
    main(args.pdf_path)
