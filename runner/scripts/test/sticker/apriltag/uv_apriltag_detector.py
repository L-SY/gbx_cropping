#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import apriltag

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class AprilTagDetectorNode:
    def __init__(self):
        rospy.init_node('apriltag_detector_node', anonymous=True)
        self.bridge = CvBridge()

        # 获取参数
        # 默认订阅 /stitched_image，如果需要可以通过 ROS 参数重设
        self.camera_topic = rospy.get_param('~camera_topic', "/stitched_image")
        self.tag_family = rospy.get_param('~tag_family', "tag16h5")
        self.force_invert = rospy.get_param('~invert_image', False)

        rospy.loginfo(f"Subscribing to: {self.camera_topic}, using tag family: {self.tag_family}")

        # 初始化 apriltag 检测器
        try:
            options = apriltag.DetectorOptions(
                families=self.tag_family,
                border=1,
                nthreads=4,
                quad_decimate=1.5,
                refine_edges=True
            )
        except TypeError:
            rospy.logwarn("Apriltag version 不兼容部分参数，退回默认选项")
            options = apriltag.DetectorOptions(families=self.tag_family)

        self.detector = apriltag.Detector(options)

        # 订阅原始 Image 而非 CompressedImage
        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self.image_callback, queue_size=1)

        # 发布结果和预处理图
        self.image_pub = rospy.Publisher("/apriltag_detected/image", Image, queue_size=1)
        self.preprocess_pub = rospy.Publisher("/apriltag_preprocessing/image", Image, queue_size=1)

        self.total_frames = 0
        self.successful_detections = 0

        rospy.loginfo("✅ AprilTag detector node ready.")
        rospy.spin()

    def image_callback(self, msg: Image):
        self.total_frames += 1
        try:
            # 直接从 Image 消息转换
            image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            image_orig = image_bgr.copy()

            # ===== 图像预处理 =====
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            mean_intensity = float(np.mean(gray))
            if self.force_invert or mean_intensity > 240:
                binary = cv2.bitwise_not(binary)
                rospy.loginfo_throttle(5, f"Inverting image (mean={mean_intensity:.1f})")

            # 发布预处理结果
            prep_msg = self.bridge.cv2_to_imgmsg(binary, encoding="mono8")
            self.preprocess_pub.publish(prep_msg)

            # ===== AprilTag 检测 =====
            detections = self.detector.detect(binary)
            if detections:
                self.successful_detections += 1
                rospy.loginfo(f"✅ Detected {len(detections)} tag(s)")

            # 在原图上绘制结果
            for det in detections:
                corners = np.int32(det.corners)
                for i in range(4):
                    cv2.line(image_orig,
                             tuple(corners[i]), tuple(corners[(i+1)%4]),
                             (0, 255, 0), 2)
                c = tuple(np.int32(det.center))
                cv2.putText(image_orig, f"ID:{det.tag_id}", c,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            rate = (self.successful_detections/self.total_frames*100) if self.total_frames else 0
            cv2.putText(image_orig,
                        f"Tags:{len(detections)} Rate:{rate:.1f}%",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            out_msg = self.bridge.cv2_to_imgmsg(image_orig, encoding="bgr8")
            self.image_pub.publish(out_msg)

        except Exception as e:
            rospy.logerr(f"⚠️ Detection failed: {e}")


if __name__ == '__main__':
    try:
        AprilTagDetectorNode()
    except rospy.ROSInterruptException:
        pass
