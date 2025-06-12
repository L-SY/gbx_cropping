#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ArucoTagDetectorNode:
    def __init__(self):
        rospy.init_node('aruco_tag_detector_node', anonymous=True)
        self.bridge = CvBridge()

        # 获取参数
        self.camera_topic = rospy.get_param('~camera_topic', "/debug_image")
        self.aruco_dict_name = rospy.get_param('~aruco_dict', "DICT_4X4_50")
        self.force_invert = rospy.get_param('~invert_image', False)

        rospy.loginfo(f"🔍 Subscribing to: {self.camera_topic}, using dict: {self.aruco_dict_name}")

        # 加载 ArUco 字典
        self.aruco_dict = self._get_aruco_dict(self.aruco_dict_name)
        self.detector_params = cv2.aruco.DetectorParameters_create()

        # 订阅原始 Image 消息
        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback, queue_size=1)

        # 发布检测图像
        self.image_pub = rospy.Publisher("/aruco_detected/image", Image, queue_size=1)
        self.preprocess_pub = rospy.Publisher("/aruco_preprocessing/image", Image, queue_size=1)

        # 状态统计
        self.total_frames = 0
        self.successful_detections = 0

        rospy.loginfo("✅ ArUco detector node ready.")
        rospy.spin()

    def _get_aruco_dict(self, name):
        if not hasattr(cv2.aruco, name):
            rospy.logerr(f"❌ Invalid ArUco dictionary name: {name}")
            raise ValueError("Invalid dictionary name")
        return cv2.aruco.Dictionary_get(getattr(cv2.aruco, name))

    def image_callback(self, msg: Image):
        self.total_frames += 1
        try:
            # 将 ROS Image 转为 OpenCV BGR
            image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            image_orig = image_bgr.copy()

            # 预处理
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 自动判断反转
            mean_intensity = np.mean(gray)
            auto_invert = mean_intensity > 240
            if self.force_invert or auto_invert:
                binary = cv2.bitwise_not(binary)
                rospy.loginfo_throttle(5, f"Inverting image (mean: {mean_intensity:.2f})")

            # 发布预处理图像
            self.preprocess_pub.publish(self.bridge.cv2_to_imgmsg(binary, encoding="mono8"))

            # ArUco 检测
            corners, ids, _ = cv2.aruco.detectMarkers(binary, self.aruco_dict, parameters=self.detector_params)

            if ids is not None:
                self.successful_detections += 1
                rospy.loginfo(f"✅ Detected {len(ids)} ArUco tag(s)")
                cv2.aruco.drawDetectedMarkers(image_orig, corners, ids)
                # 标注 tag ID
                for i, corner in enumerate(corners):
                    center = np.mean(corner[0], axis=0).astype(int)
                    cv2.putText(image_orig, f"ID: {ids[i][0]}", tuple(center),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                rospy.loginfo_throttle(5, "No ArUco tags detected.")

            # 统计和发布
            detection_rate = (self.successful_detections / self.total_frames) * 100 if self.total_frames > 0 else 0
            cv2.putText(image_orig, f"Tags: {0 if ids is None else len(ids)} Rate: {detection_rate:.1f}%",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            self.image_pub.publish(self.bridge.cv2_to_imgmsg(image_orig, encoding="bgr8"))

        except Exception as e:
            rospy.logerr(f"⚠️ Detection failed: {e}")


if __name__ == '__main__':
    try:
        ArucoTagDetectorNode()
    except rospy.ROSInterruptException:
        pass
