import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import argparse

def detect_circles_contour(gray, image, debug=False):
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000 or area > 1000000:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.3:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        centers.append((int(x), int(y)))
        cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        cv2.circle(image, (int(x), int(y)), 4, (0, 0, 255), -1)

    return centers

def filter_nearby_circles(circles, min_dist=30):
    filtered = []
    for c in circles:
        if all(np.linalg.norm(np.array(c[:2]) - np.array(f[:2])) > min_dist for f in filtered):
            filtered.append(c)
    return filtered

def detect_circles_hough(gray, image):
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=40
    )
    centers = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        circles = filter_nearby_circles(circles, min_dist=30)
        for (x, y, r) in circles:
            centers.append((x, y))
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)
    return centers

def process_image(image_path, use_hough=False, combine=False, debug=False):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image: {image_path}")
        return

    base_name = os.path.splitext(image_path)[0]
    base_dir = os.path.dirname(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    draw_image = image.copy()

    if combine:
        centers_c = detect_circles_contour(gray, draw_image, debug=debug)
        centers_h = detect_circles_hough(gray, draw_image) if use_hough else []
        centers_all = list({(x, y) for (x, y) in centers_c + centers_h})
    elif use_hough:
        centers_h = detect_circles_hough(gray, draw_image) if use_hough else []
        centers_all = centers_h
    else:
        centers_c = detect_circles_contour(gray, draw_image, debug=debug)
        centers_all = centers_c

    print(f"Number of detected circles: {len(centers_all)}")
    for i, (x, y) in enumerate(centers_all):
        print(f"  Circle {i+1}: x={x}, y={y}")

    # Create output folders
    cropped_dir = os.path.join(base_dir, "cropped")
    marked_dir = os.path.join(base_dir, "marked")
    os.makedirs(cropped_dir, exist_ok=True)
    os.makedirs(marked_dir, exist_ok=True)

    if len(centers_all) >= 3:
        pts = np.array(centers_all, dtype=np.float32)
        # 按 y 坐标排序，划分上下行
        pts_sorted = pts[np.argsort(pts[:, 1])]
        mid_y = np.median(pts[:, 1])
        top_row = pts_sorted[pts_sorted[:, 1] <= mid_y]
        bottom_row = pts_sorted[pts_sorted[:, 1] > mid_y]

        if len(top_row) < 2 or len(bottom_row) < 2:
            print("⚠️ Not enough points in top or bottom row to define a rectangle.")
            return

        # 左右边界取所有点的最左和最右，保证贴近边缘
        min_x = np.min(pts[:, 0])
        max_x = np.max(pts[:, 0])
        top_y = np.min(top_row[:, 1])
        bottom_y = np.max(bottom_row[:, 1])

        margin = 5  # 可调，设为 0 表示最贴合

        rect_pts = np.array([
            [min_x - margin, top_y - margin],      # top-left
            [max_x + margin, top_y - margin],      # top-right
            [max_x + margin, bottom_y + margin],   # bottom-right
            [min_x - margin, bottom_y + margin]    # bottom-left
        ], dtype=np.float32)

        width = int(np.linalg.norm(rect_pts[1] - rect_pts[0]))
        height = int(np.linalg.norm(rect_pts[3] - rect_pts[0]))
        dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
        cropped = cv2.warpPerspective(image, M, (width, height))
        cropped_path = os.path.join(cropped_dir, os.path.basename(base_name) + "_cropped.jpg")
        cv2.imwrite(cropped_path, cropped)
        print(f"✅ Cropped image saved to: {cropped_path}")


    marked_path = os.path.join(marked_dir, os.path.basename(base_name) + "_marked.jpg")
    cv2.imwrite(marked_path, draw_image)
    print(f"✅ Marked image saved to: {marked_path}")

    if debug:
        plt.imshow(cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB))
        plt.title("Detected Circles Visualization")
        plt.axis("off")
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to image or folder")
    parser.add_argument("--use-hough", action="store_true", help="Enable Hough circle detection")
    parser.add_argument("--combine", action="store_true", help="Combine results from both methods")
    parser.add_argument("--debug", action="store_true", help="Show debug visualization")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        process_image(args.input, use_hough=args.use_hough, combine=args.combine, debug=args.debug)
    elif os.path.isdir(args.input):
        image_paths = sorted(glob(os.path.join(args.input, '*.[jp][pn]g')))
        for path in image_paths:
            process_image(path, use_hough=args.use_hough, combine=args.combine, debug=args.debug)
    else:
        print("❌ Invalid input path")

if __name__ == "__main__":
    main()