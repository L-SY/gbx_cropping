#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import apriltag

from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

class AprilTagDetectorNode:
    def __init__(self):
        rospy.init_node('apriltag_detector_node', anonymous=True)

        # Bridge for format conversion
        self.bridge = CvBridge()

        # 配置检测器选项，提高检测鲁棒性但保持简单
        try:
            # 尝试使用精简参数集
            options = apriltag.DetectorOptions(
                families="tag25h9",
                border=1,           # 增加边界检测（默认为1）
                nthreads=4,         # 使用多线程
                quad_decimate=1.5,  # 降采样因子，1.5在速度和精度之间更好的平衡
                refine_edges=True   # 优化边缘
            )
        except TypeError:
            # 降级到最小参数集
            rospy.logwarn("Using simplified detector options due to library version limitations")
            options = apriltag.DetectorOptions(families="tag25h9")

        self.detector = apriltag.Detector(options)

        # 获取相机话题名称参数，默认为"/hk_camera/image_raw/compressed"
        camera_topic = rospy.get_param('~camera_topic', "/hk_camera/image_raw/compressed")
        tag_family = rospy.get_param('~tag_family', "tag25h9")
        rospy.loginfo(f"Subscribing to camera topic: {camera_topic}, using tag family: {tag_family}")

        # 订阅压缩图像话题
        self.image_sub = rospy.Subscriber(camera_topic, CompressedImage, self.image_callback, queue_size=1)

        # 发布检测图像
        self.image_pub = rospy.Publisher("/apriltag_detected/image", Image, queue_size=1)

        # 发布预处理后的图像
        self.preprocess_pub = rospy.Publisher("/apriltag_preprocessing/image", Image, queue_size=1)

        # 强反转（用于严重过曝的图像）
        self.invert_image = rospy.get_param('~invert_image', False)

        # 阈值
        self.threshold_value = rospy.get_param('~threshold', 100)

        # 处理统计
        self.total_frames = 0
        self.successful_detections = 0

        rospy.loginfo("✅ AprilTag detector node initialized with simplified processing.")
        rospy.spin()

    def image_callback(self, msg):
        self.total_frames += 1
        try:
            # 解压缩图像
            np_arr = np.frombuffer(msg.data, np.uint8)
            image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            image_orig = image_bgr.copy()

            # 简单高效的预处理
            # 1. 转换为灰度图像
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

            # 2. 直方图均衡化 - 对曝光不均很有效
            equalized = cv2.equalizeHist(gray)

            # 3. 高斯模糊减少噪声
            blurred = cv2.GaussianBlur(equalized, (5,5), 0)

            # 4. 应用自适应阈值 - 对光照不均更有效
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2)

            # 可选：图像反转 - 对严重过曝的图像有用
            if self.invert_image:
                binary = cv2.bitwise_not(binary)

            # 发布预处理后的图像
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(binary, encoding="mono8")
                self.preprocess_pub.publish(debug_msg)
            except Exception as e:
                rospy.logwarn(f"Failed to publish preprocessed image: {e}")

            # 执行AprilTag检测
            detections = self.detector.detect(binary)

            if detections:
                self.successful_detections += 1
                rospy.loginfo(f"Detected {len(detections)} tags")

            # 在原图上绘制检测结果
            for det in detections:
                corners = np.int32(det.corners)
                tag_id = det.tag_id

                # 绘制边框
                for i in range(4):
                    pt1 = tuple(corners[i])
                    pt2 = tuple(corners[(i + 1) % 4])
                    cv2.line(image_orig, pt1, pt2, (0, 255, 0), 2)

                # 标注 ID
                center = tuple(np.int32(det.center))
                cv2.putText(image_orig, f"ID: {tag_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 添加检测数量信息
            detection_rate = (self.successful_detections / self.total_frames) * 100 if self.total_frames > 0 else 0
            cv2.putText(image_orig, f"Tags: {len(detections)} Rate: {detection_rate:.1f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 发布结果图像
            img_msg = self.bridge.cv2_to_imgmsg(image_orig, encoding="bgr8")
            self.image_pub.publish(img_msg)

        except Exception as e:
            rospy.logerr(f"⚠️ Image processing failed: {e}")

if __name__ == '__main__':
    try:
        AprilTagDetectorNode()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    try:
        AprilTagDetectorNode()
    except rospy.ROSInterruptException:
        pass