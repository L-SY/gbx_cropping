#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import threading
import queue

class PanelDetector:
    def __init__(self):
        rospy.init_node('panel_detector', anonymous=True)
        self.bridge = CvBridge()

        self.camera_matrix = np.array([
            [2343.181585, 0.000000, 1221.765641],
            [0.000000, 2341.245683, 1040.731733],
            [0.000000, 0.000000, 1.000000]
        ])
        self.dist_coeffs = np.array([-0.080789, 0.084471, 0.000261, 0.000737, 0.000000])

        self.result_pub = rospy.Publisher('/panel_detector/result/compressed', CompressedImage, queue_size=1)
        self.light_panel_pub = rospy.Publisher('/panel_detector/light_panel/compressed', CompressedImage, queue_size=1)
        self.foam_board_pub = rospy.Publisher('/panel_detector/foam_board/compressed', CompressedImage, queue_size=1)

        self.image_sub = rospy.Subscriber("/hk_camera/image_raw", CompressedImage, self.image_callback)

        self.image_queue = queue.Queue(maxsize=5)
        self.processor_thread = threading.Thread(target=self.image_processor_loop)
        self.processor_thread.daemon = True
        self.processor_thread.start()

        self.last_pub_time = rospy.Time.now()
        rospy.loginfo("Panel detector initialized.")

    def image_callback(self, compressed_msg):
        if not self.image_queue.full():
            self.image_queue.put((compressed_msg.data, compressed_msg.header.stamp))

    def image_processor_loop(self):
        while not rospy.is_shutdown():
            try:
                data, timestamp = self.image_queue.get(timeout=1.0)
                np_arr = np.frombuffer(data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                self.process_image(cv_image, timestamp)
            except queue.Empty:
                continue
            except Exception as e:
                rospy.logerr(f"Processing error: {e}")

    def process_image(self, image, timestamp):
        undistorted = self.undistort_image(image)
        light_panel_result = self.detect_light_panel(undistorted)
        if light_panel_result is None:
            result_img = undistorted.copy()
            cv2.putText(result_img, "Light panel not detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.publish_compressed_image(result_img, self.result_pub, timestamp)
            return

        light_panel, light_panel_box, M = light_panel_result
        foam_board, foam_board_box = self.extract_foam_board(light_panel)
        foam_board_box_orig = self.transform_foam_box_to_original(foam_board_box, light_panel_box, M)

        result_img = undistorted.copy()
        cv2.drawContours(result_img, [light_panel_box], 0, (0, 255, 0), 2)
        cv2.drawContours(result_img, [foam_board_box_orig], 0, (0, 0, 255), 2)
        cv2.putText(result_img, "Light Panel", (light_panel_box[0][0], light_panel_box[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_img, "Foam Board", (foam_board_box_orig[0][0], foam_board_box_orig[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.publish_compressed_image(result_img, self.result_pub, timestamp)
        self.publish_compressed_image(light_panel, self.light_panel_pub, timestamp)
        self.publish_compressed_image(foam_board, self.foam_board_pub, timestamp)

    def undistort_image(self, image):
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, newcameramtx)
        x, y, w, h = roi
        if x > 0 and y > 0 and w > 0 and h > 0:
            undistorted = undistorted[y:y + h, x:x + w]
        return undistorted

    def detect_light_panel(self, image):
        scale = 0.5
        small = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) < 300:
            return None

        max_contour = (max_contour / scale).astype(np.int32)
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        width = int(rect[1][0])
        height = int(rect[1][1])
        aspect_ratio = max(width, height) / (min(width, height) + 1e-5)

        if abs(aspect_ratio - 1.0) > 0.3:
            return None

        src_pts = box.astype("float32")
        src_pts = sorted(src_pts, key=lambda pt: pt[1])
        top_pts = sorted(src_pts[:2], key=lambda pt: pt[0])
        bottom_pts = sorted(src_pts[2:], key=lambda pt: pt[0])
        src_pts = np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype="float32")

        dst_size = max(width, height)
        dst_pts = np.array([
            [0, 0],
            [dst_size - 1, 0],
            [dst_size - 1, dst_size - 1],
            [0, dst_size - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (dst_size, dst_size))
        return warped, box, M

    def extract_foam_board(self, image):
        height, width = image.shape[:2]
        foam_height = height // 2
        top = (height - foam_height) // 2
        foam_board = image[top:top + foam_height, 0:width]
        foam_box = np.array([
            [0, top],
            [width - 1, top],
            [width - 1, top + foam_height - 1],
            [0, top + foam_height - 1]
        ], dtype=np.int32)
        return foam_board, foam_box

    def transform_foam_box_to_original(self, foam_box, light_panel_box, M):
        M_inv = np.linalg.inv(M)
        foam_box_orig = []
        for point in foam_box:
            p = np.array([point[0], point[1], 1])
            p_orig = M_inv.dot(p)
            p_orig = (p_orig[:2] / p_orig[2]).astype(int)
            foam_box_orig.append(p_orig)
        return np.array(foam_box_orig, dtype=np.int32)

    def publish_compressed_image(self, cv_image, publisher, timestamp):
        msg = CompressedImage()
        msg.header.stamp = timestamp
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tobytes()
        publisher.publish(msg)