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

    centers_c = detect_circles_contour(gray, draw_image, debug=debug)
    centers_h = detect_circles_hough(gray, draw_image) if use_hough else []

    if combine:
        centers_all = list({(x, y) for (x, y) in centers_c + centers_h})
    elif use_hough:
        centers_all = centers_h
    else:
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
        pts = np.array(centers_all, dtype=np.int32)
        # Compute bounding rectangle aligned along the longest axis (assume alignment along line)
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        # Expand to rectangular ROI covering all points in a line
        padding = 20  # Optional padding
        if (x_max - x_min) > (y_max - y_min):
            y_center = int(np.mean(y_coords))
            height = max(50, int((y_max - y_min) * 1.5))
            y_crop_min = max(0, y_center - height // 2 - padding)
            y_crop_max = min(image.shape[0], y_center + height // 2 + padding)
            cropped = image[y_crop_min:y_crop_max, x_min - padding:x_max + padding]
        else:
            x_center = int(np.mean(x_coords))
            width = max(50, int((x_max - x_min) * 1.5))
            x_crop_min = max(0, x_center - width // 2 - padding)
            x_crop_max = min(image.shape[1], x_center + width // 2 + padding)
            cropped = image[y_min - padding:y_max + padding, x_crop_min:x_crop_max]

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
