#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import apriltag

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def balance_white(img):
    """
    简易灰度世界算法白平衡
    """
    img = img.astype(np.float32)
    mean_b, mean_g, mean_r = cv2.mean(img)[:3]
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    scale_b = mean_gray / (mean_b + 1e-6)
    scale_g = mean_gray / (mean_g + 1e-6)
    scale_r = mean_gray / (mean_r + 1e-6)
    balanced = cv2.merge([
        np.clip(img[:, :, 0] * scale_b, 0, 255),
        np.clip(img[:, :, 1] * scale_g, 0, 255),
        np.clip(img[:, :, 2] * scale_r, 0, 255)
    ]).astype(np.uint8)
    return balanced

class AprilTagDetectorNode:
    def __init__(self):
        rospy.init_node('apriltag_detector_node', anonymous=True)
        self.bridge = CvBridge()

        self.camera_topic = rospy.get_param('~camera_topic', "/stitched_image")
        self.tag_family = rospy.get_param('~tag_family', "tag16h5")
        self.force_invert = rospy.get_param('~invert_image', False)

        rospy.loginfo(f"Subscribing to: {self.camera_topic}, using family: {self.tag_family}")

        # Apriltag Detector
        try:
            opts = apriltag.DetectorOptions(families=self.tag_family,
                                            border=1, nthreads=4,
                                            quad_decimate=1.5,
                                            refine_edges=True)
        except TypeError:
            rospy.logwarn("Fallback to default DetectorOptions")
            opts = apriltag.DetectorOptions(families=self.tag_family)
        self.detector = apriltag.Detector(opts)

        self.image_sub = rospy.Subscriber(self.camera_topic, Image,
                                          self.image_callback, queue_size=1)
        self.preprocess_pub = rospy.Publisher("/apriltag_preprocessing/image",
                                              Image, queue_size=1)
        self.image_pub = rospy.Publisher("/apriltag_detected/image",
                                         Image, queue_size=1)

        self.total = 0
        self.success = 0
        rospy.loginfo("AprilTag Node Ready")
        rospy.spin()

    def image_callback(self, msg: Image):
        self.total += 1
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            orig = bgr.copy()

            # 白平衡
            wb = balance_white(bgr)

            # LAB CLAHE
            lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l2 = clahe.apply(l)
            lab2 = cv2.merge((l2, a, b))
            rgb2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

            # 转 HSV 提取 S、V
            hsv = cv2.cvtColor(rgb2, cv2.COLOR_BGR2HSV)
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]
            # 低饱和度掩码（灰度／黑白区域）
            _, s_mask = cv2.threshold(s, 60, 255, cv2.THRESH_BINARY_INV)

            # V 通道平滑 + 自适应阈值
            v_blur = cv2.GaussianBlur(v, (5,5), 0)
            th_v = cv2.adaptiveThreshold(v_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY_INV,
                                         blockSize=51, C=10)

            # 合并掩码
            binary = cv2.bitwise_and(th_v, s_mask)

            # 反转条件
            mean_v = float(np.mean(v))
            if self.force_invert or mean_v > 240:
                binary = cv2.bitwise_not(binary)
                rospy.loginfo_throttle(5, f"Inverted (mean V={mean_v:.1f})")

            # 发布预处理图
            prep_msg = self.bridge.cv2_to_imgmsg(binary, "mono8")
            self.preprocess_pub.publish(prep_msg)

            # 检测
            dets = self.detector.detect(binary)
            if dets:
                self.success += 1
                rospy.loginfo(f"Detected {len(dets)} tags")

            # 绘图
            for d in dets:
                cns = np.int32(d.corners)
                for i in range(4):
                    cv2.line(orig, tuple(cns[i]), tuple(cns[(i+1)%4]), (0,255,0),2)
                ctr = tuple(np.int32(d.center))
                cv2.putText(orig, f"ID:{d.tag_id}", ctr,
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

            rate = (self.success/self.total*100) if self.total else 0
            cv2.putText(orig, f"Tags:{len(dets)} Rate:{rate:.1f}%",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

            out = self.bridge.cv2_to_imgmsg(orig, "bgr8")
            self.image_pub.publish(out)
        except Exception as e:
            rospy.logerr(f"Detection failed: {e}")

if __name__ == '__main__':
    try:
        AprilTagDetectorNode()
    except rospy.ROSInterruptException:
        pass
