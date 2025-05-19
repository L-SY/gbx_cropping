#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import os
import sys

class PanelDetector:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('panel_detector', anonymous=True)

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Camera calibration parameters
        self.camera_matrix = np.array([
            [2343.181585, 0.000000, 1221.765641],
            [0.000000, 2341.245683, 1040.731733],
            [0.000000, 0.000000, 1.000000]
        ])

        self.dist_coeffs = np.array([-0.080789, 0.084471, 0.000261, 0.000737, 0.000000])

        # Create publishers
        self.result_pub = rospy.Publisher('/panel_detector/result/compressed', CompressedImage, queue_size=1)
        self.light_panel_pub = rospy.Publisher('/panel_detector/light_panel/compressed', CompressedImage, queue_size=1)
        self.foam_board_pub = rospy.Publisher('/panel_detector/foam_board/compressed', CompressedImage, queue_size=1)

        # Set up subscriber with callback
        self.image_sub = rospy.Subscriber("/hk_camera/image_raw/compressed", CompressedImage, self.image_callback)

        rospy.loginfo("Panel detector initialized and waiting for images.")

    def image_callback(self, compressed_msg):
        """Process incoming compressed image messages"""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Process the image
            self.process_image(cv_image, compressed_msg.header.stamp)

        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

    def process_image(self, image, timestamp):
        """Process image: undistort, detect panels, publish results"""

        # Step 1: Undistort the image
        undistorted = self.undistort_image(image)

        # Step 2: Try to detect light panel
        light_panel_result = self.detect_light_panel(undistorted)
        if light_panel_result is None:
            rospy.logwarn("Failed to detect light panel")
            # Publish the undistorted image with a warning text
            result_img = undistorted.copy()
            cv2.putText(result_img, "Light panel not detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.publish_compressed_image(result_img, self.result_pub, timestamp)
            return

        light_panel, light_panel_box, M = light_panel_result

        # Step 3: Extract foam board region
        foam_board, foam_board_box = self.extract_foam_board(light_panel)

        # Step 4: Transform foam board box back to original image coordinates
        foam_board_box_orig = self.transform_foam_box_to_original(foam_board_box, light_panel_box, M)

        # Create result visualization
        result_img = undistorted.copy()

        # Draw light panel rectangle on original image (green)
        cv2.drawContours(result_img, [light_panel_box], 0, (0, 255, 0), 2)

        # Draw foam board rectangle on original image (red)
        cv2.drawContours(result_img, [foam_board_box_orig], 0, (0, 0, 255), 2)

        # Add labels
        cv2.putText(result_img, "Light Panel", (light_panel_box[0][0], light_panel_box[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_img, "Foam Board", (foam_board_box_orig[0][0], foam_board_box_orig[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Publish results
        self.publish_compressed_image(result_img, self.result_pub, timestamp)
        self.publish_compressed_image(light_panel, self.light_panel_pub, timestamp)
        self.publish_compressed_image(foam_board, self.foam_board_pub, timestamp)

    def transform_foam_box_to_original(self, foam_box, light_panel_box, M):
        """Transform foam board box coordinates from light panel space to original image space"""
        # Create a simplified representation of foam board in light panel coordinates
        # Since the foam board is centered and half the height of the light panel
        height, width = foam_box[2][1] - foam_box[0][1], foam_box[1][0] - foam_box[0][0]

        # Calculate inverse perspective transform matrix
        M_inv = np.linalg.inv(M)

        # Transform each corner of the foam box back to the original image
        foam_box_orig = []
        for point in foam_box:
            # Convert to homogeneous coordinates
            p = np.array([point[0], point[1], 1])

            # Apply inverse transform
            p_orig = M_inv.dot(p)

            # Convert back from homogeneous coordinates
            p_orig = (p_orig[:2] / p_orig[2]).astype(int)

            foam_box_orig.append(p_orig)

        return np.array(foam_box_orig, dtype=np.int32)

    def undistort_image(self, image):
        """Undistort image using camera calibration parameters"""
        h, w = image.shape[:2]

        # Calculate optimal new camera matrix
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )

        # Undistort the image
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, newcameramtx)

        # Crop the image (optional)
        x, y, w, h = roi
        if x > 0 and y > 0 and w > 0 and h > 0:
            undistorted = undistorted[y:y+h, x:x+w]

        return undistorted

    def detect_light_panel(self, image):
        """Detect and extract the 20x20cm light panel from the image"""

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
            rospy.logwarn("No contours found in the image")
            return None

        # Find the largest contour (assumed to be the light panel)
        max_contour = max(contours, key=cv2.contourArea)

        # Skip very small contours
        if cv2.contourArea(max_contour) < 1000:  # Minimum area threshold
            rospy.logwarn("Largest contour too small, likely not the light panel")
            return None

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
            rospy.logwarn(f"Warning: Detected shape aspect ratio differs from expected (Detected: {aspect_ratio:.2f}, Expected: {expected_ratio:.2f})")

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

        return warped, box, M

    def extract_foam_board(self, image):
        """Extract the 20x10cm foam board from the light panel image"""

        height, width = image.shape[:2]

        # Calculate foam board dimensions (same width, half height, centered)
        foam_width = width
        foam_height = height // 2

        # Calculate top position to center the crop in the image
        top = (height - foam_height) // 2

        # Extract foam board region
        foam_board = image[top:top+foam_height, 0:foam_width]

        # Create a box representing the foam board within the light panel
        foam_box = np.array([
            [0, top],
            [foam_width-1, top],
            [foam_width-1, top + foam_height-1],
            [0, top + foam_height-1]
        ], dtype=np.int32)

        return foam_board, foam_box

    def publish_compressed_image(self, cv_image, publisher, timestamp):
        """Convert OpenCV image to CompressedImage and publish it"""
        msg = CompressedImage()
        msg.header.stamp = timestamp
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tostring()
        publisher.publish(msg)

def main():
    try:
        detector = PanelDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()