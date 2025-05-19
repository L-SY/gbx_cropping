import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def extract_foam_board_direct(image_path, output_path=None, process_viz_path=None):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image: {image_path}")
        return None

    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate foam board dimensions (same width, half height, centered)
    foam_width = width
    foam_height = height // 2

    # Calculate top position to center the crop in the image
    top = (height - foam_height) // 2

    # Extract foam board region
    foam_board = image[top:top+foam_height, 0:foam_width]

    # Save result
    if output_path:
        cv2.imwrite(output_path, foam_board)
        print(f"Extracted foam board saved to: {output_path}")

    # Visualize the process with improved presentation
    plt.figure(figsize=(15, 10))

    # Original image
    plt.subplot(131)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axis for cleaner look

    # Draw rectangle on original image to show cropping area
    rect_img = image.copy()
    cv2.rectangle(rect_img, (0, top), (foam_width, top + foam_height), (0, 255, 0), 3)

    plt.subplot(132)
    plt.title("Crop Area")
    plt.imshow(cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axis for cleaner look

    # Extracted foam board
    plt.subplot(133)
    plt.title("Extracted Foam Board")
    plt.imshow(cv2.cvtColor(foam_board, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axis for cleaner look

    # Adjust layout
    plt.tight_layout()

    # Save visualization image if path is provided
    if process_viz_path:
        plt.savefig(process_viz_path, dpi=300, bbox_inches='tight')
        print(f"Process visualization saved to: {process_viz_path}")

    plt.show()

    return foam_board

# Usage example
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_foam_board_direct.py <input_image_path> [output_image_path] [process_viz_path]")
        sys.exit(1)

    input_path = sys.argv[1]

    # Default output filenames based on input filename
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = sys.argv[2] if len(sys.argv) > 2 else f"{base_name}_foam_board.jpg"
    process_viz_path = sys.argv[3] if len(sys.argv) > 3 else f"{base_name}_foam_process.png"

    extract_foam_board_direct(input_path, output_path, process_viz_path)