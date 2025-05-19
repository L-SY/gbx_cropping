import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def undistort_image(image):
    """Undistort image using camera calibration parameters"""

    # Camera matrix
    camera_matrix = np.array([
        [2343.181585, 0.000000, 1221.765641],
        [0.000000, 2341.245683, 1040.731733],
        [0.000000, 0.000000, 1.000000]
    ])

    # Distortion coefficients
    dist_coeffs = np.array([-0.080789, 0.084471, 0.000261, 0.000737, 0.000000])

    # Rectification matrix (identity in this case)
    # Not needed for undistortion

    # Projection matrix (not needed for undistortion)
    # Not used here

    # Undistort the image
    h, w = image.shape[:2]

    # Calculate optimal new camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    # Undistort the image
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, newcameramtx)

    # Crop the image (optional)
    x, y, w, h = roi
    if x > 0 and y > 0 and w > 0 and h > 0:
        undistorted = undistorted[y:y+h, x:x+w]

    return undistorted

def extract_light_panel(image):
    """Extract the 20x20cm light panel from the image"""

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use threshold segmentation to find bright areas
    # Use OTSU algorithm to automatically determine threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find all contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, try inverting the threshold and search again
    if not contours:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found")
        return None, None

    # Find the largest contour (assumed to be the light panel)
    max_contour = max(contours, key=cv2.contourArea)

    # Get the minimum area rectangle
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Get rectangle width and height
    width = int(rect[1][0])
    height = int(rect[1][1])

    # Ensure aspect ratio is close to expected light panel
    aspect_ratio = max(width, height) / min(width, height)
    expected_ratio = 20 / 20  # Expected aspect ratio is 1

    if abs(aspect_ratio - expected_ratio) > 0.3:  # Allow some error
        print(f"Warning: Detected shape aspect ratio differs from expected (Detected: {aspect_ratio:.2f}, Expected: {expected_ratio:.2f})")

    # Get perspective transform source and destination points
    src_pts = box.astype("float32")

    # Ensure points order is: top-left, top-right, bottom-right, bottom-left
    # Sort by y-coordinate
    src_pts = sorted(src_pts, key=lambda pt: pt[1])
    # First two points sorted by x-coordinate (determine top-left and top-right)
    top_pts = sorted(src_pts[:2], key=lambda pt: pt[0])
    # Last two points sorted by x-coordinate (determine bottom-left and bottom-right)
    bottom_pts = sorted(src_pts[2:], key=lambda pt: pt[0])
    # Combine sorted points
    src_pts = np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype="float32")

    # Target size (square)
    dst_size = max(width, height)
    dst_pts = np.array([
        [0, 0],
        [dst_size - 1, 0],
        [dst_size - 1, dst_size - 1],
        [0, dst_size - 1]
    ], dtype="float32")

    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply perspective transform
    warped = cv2.warpPerspective(image, M, (dst_size, dst_size))

    # Create visualization figure
    plt.figure(figsize=(15, 10))

    plt.subplot(221)
    plt.title("Undistorted Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(222)
    plt.title("Threshold")
    plt.imshow(thresh, cmap='gray')
    plt.axis('off')

    plt.subplot(223)
    contour_img = image.copy()
    cv2.drawContours(contour_img, [box], 0, (0, 255, 0), 2)
    plt.title("Detected Light Panel")
    plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(224)
    plt.title("Cropped Light Panel")
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()

    return warped, plt.gcf()  # Return the cropped image and figure

def extract_foam_board(image):
    """Extract the 20x10cm foam board from the light panel image"""

    height, width = image.shape[:2]

    # Calculate foam board dimensions (same width, half height, centered)
    foam_width = width
    foam_height = height // 2

    # Calculate top position to center the crop in the image
    top = (height - foam_height) // 2

    # Extract foam board region
    foam_board = image[top:top+foam_height, 0:foam_width]

    # Create visualization figure
    plt.figure(figsize=(15, 5))

    # Original light panel
    plt.subplot(131)
    plt.title("Light Panel")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Draw rectangle on original image to show cropping area
    rect_img = image.copy()
    cv2.rectangle(rect_img, (0, top), (foam_width, top + foam_height), (0, 255, 0), 3)

    plt.subplot(132)
    plt.title("Foam Board Area")
    plt.imshow(cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Extracted foam board
    plt.subplot(133)
    plt.title("Extracted Foam Board")
    plt.imshow(cv2.cvtColor(foam_board, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()

    return foam_board, plt.gcf()  # Return the cropped image and figure

def process_image(input_path):
    """Process the image: undistort, extract light panel, and then foam board"""

    # Get base directories and filenames
    dir_path = os.path.dirname(os.path.abspath(input_path))
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Define output paths
    undistorted_path = os.path.join(dir_path, f"{base_name}_undistorted.jpg")
    light_panel_path = os.path.join(dir_path, f"{base_name}_light_panel.jpg")
    light_panel_viz_path = os.path.join(dir_path, f"{base_name}_light_panel_process.png")
    foam_board_path = os.path.join(dir_path, f"{base_name}_foam_board.jpg")
    foam_board_viz_path = os.path.join(dir_path, f"{base_name}_foam_board_process.png")

    # Step 0: Read image
    print(f"Processing image: {input_path}")
    original_image = cv2.imread(input_path)
    if original_image is None:
        print(f"Cannot read image: {input_path}")
        return False

    # Step 1: Undistort the image
    print("Step 1: Undistorting image...")
    undistorted_image = undistort_image(original_image)

    # Save undistorted image
    cv2.imwrite(undistorted_path, undistorted_image)
    print(f"Undistorted image saved to: {undistorted_path}")

    # Create comparison visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(122)
    plt.title("Undistorted Image")
    plt.imshow(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    undistort_viz_path = os.path.join(dir_path, f"{base_name}_undistort_comparison.png")
    plt.savefig(undistort_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Undistortion comparison saved to: {undistort_viz_path}")

    # Step 2: Extract light panel from undistorted image
    print("Step 2: Extracting light panel...")
    light_panel, light_panel_fig = extract_light_panel(undistorted_image)
    if light_panel is None:
        print("Failed to extract light panel. Process aborted.")
        return False

    # Save light panel results
    cv2.imwrite(light_panel_path, light_panel)
    light_panel_fig.savefig(light_panel_viz_path, dpi=300, bbox_inches='tight')
    plt.close(light_panel_fig)
    print(f"Light panel saved to: {light_panel_path}")
    print(f"Light panel process visualization saved to: {light_panel_viz_path}")

    # Step 3: Extract foam board from light panel
    print("Step 3: Extracting foam board...")
    foam_board, foam_board_fig = extract_foam_board(light_panel)

    # Save foam board results
    cv2.imwrite(foam_board_path, foam_board)
    foam_board_fig.savefig(foam_board_viz_path, dpi=300, bbox_inches='tight')
    plt.close(foam_board_fig)
    print(f"Foam board saved to: {foam_board_path}")
    print(f"Foam board process visualization saved to: {foam_board_viz_path}")

    print("Processing complete!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_panels.py <input_image_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    process_image(input_path)