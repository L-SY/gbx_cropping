import cv2
import numpy as np
import argparse
import os

# Functions provided by the user
def filter_nearby_circles(circles, min_dist=30):
    filtered = []
    for circle in circles:
        x, y, r = circle
        if all(np.linalg.norm(np.array((x, y)) - np.array((cx, cy))) >= min_dist for cx, cy, _ in filtered):
            filtered.append((x, y, r))
    return filtered

def detect_circles_hough(gray, image=None, debug=False):
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=18,
        minRadius=10,
        maxRadius=40
    )
    centers = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        circles = filter_nearby_circles(circles, min_dist=30)
        for (x, y, r) in circles:
            centers.append((x, y, r))
            if debug and image is not None:
                cv2.circle(image, (x, y), r, (0, 255, 0), 2)
                cv2.circle(image, (x, y), 2, (0, 0, 255), 3)
    return centers

# Helper to separate circles into two rows
def split_rows(circles, threshold=50):
    circles.sort(key=lambda c: c[1])
    top_row = [c for c in circles if c[1] <= circles[0][1] + threshold]
    bottom_row = [c for c in circles if c[1] > circles[0][1] + threshold]
    top_row.sort(key=lambda c: c[0])
    bottom_row.sort(key=lambda c: c[0])
    return top_row, bottom_row

# Find the best shift based on minimum error
def find_best_shift(base_row, next_row):
    min_error = float('inf')
    best_dx = 0

    for i in range(len(base_row) - 2):
        for j in range(min(3, len(next_row))):
            dx = int(base_row[i][0]) - int(next_row[j][0])
            shifted_next_x = [int(x) + dx for x, y, r in next_row]
            errors = [abs(shifted_next_x[k] - int(base_row[i + k][0])) for k in range(min(len(base_row)-i, len(next_row))) if i+k < len(base_row)]
            error = np.mean(errors)

            if error < min_error:
                min_error = error
                best_dx = dx

    return best_dx

# Stitch two images based on best shift
def stitch_images(img1, img2, dx):
    if dx > 0:
        stitched_width = img1.shape[1] + dx
        stitched_image = np.zeros((img1.shape[0], stitched_width, 3), dtype=np.uint8)
        stitched_image[:, :img1.shape[1]] = img1
        stitched_image[:, dx:dx+img2.shape[1]] = img2
    else:
        stitched_width = img2.shape[1] - dx
        stitched_image = np.zeros((img1.shape[0], stitched_width, 3), dtype=np.uint8)
        stitched_image[:, :img2.shape[1]] = img2
        stitched_image[:, -dx:-dx+img1.shape[1]] = img1

    return stitched_image

# Main stitching function for a folder
def stitch_folder(folder_path, debug=False):
    images = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

    stitched_img = cv2.imread(images[0])
    for next_img_path in images[1:]:
        next_img = cv2.imread(next_img_path)

        gray1 = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

        circles1 = detect_circles_hough(gray1, stitched_img.copy(), debug)
        circles2 = detect_circles_hough(gray2, next_img.copy(), debug)

        top1, bottom1 = split_rows(circles1)
        top2, bottom2 = split_rows(circles2)

        dx_top = find_best_shift(top1, top2)
        dx_bottom = find_best_shift(bottom1, bottom2)

        dx_final = int((dx_top + dx_bottom) / 2)

        stitched_img = stitch_images(stitched_img, next_img, dx_final)

        if debug:
            print(f"Top shift: {dx_top}, Bottom shift: {dx_bottom}, Final shift: {dx_final}")

    return stitched_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image stitching based on circle matching.')
    parser.add_argument('--folder', required=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    result_img = stitch_folder(args.folder, args.debug)
    cv2.imwrite('stitched_result.jpg', result_img)
    if args.debug:
        cv2.imshow('Stitched Image', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()