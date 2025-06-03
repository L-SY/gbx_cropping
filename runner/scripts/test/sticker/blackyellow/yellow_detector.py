#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class YellowDetector:
    def __init__(self):
        rospy.init_node('yellow_detector_node', anonymous=True)
        self.bridge = CvBridge()

        # 订阅原始图像话题
        rospy.Subscriber("/hk_camera/image_raw/compressed", CompressedImage, self.image_callback)

        # 发布处理后的图像
        self.image_pub = rospy.Publisher("/yellow_detect/image/compressed", CompressedImage, queue_size=1)

        rospy.loginfo("Yellow Detector Node Started.")
        rospy.spin()

    def image_callback(self, msg):
        # 解码压缩图像
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 转为 HSV 色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 定义黄色的HSV范围
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # 创建掩码
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 查找黄色区域轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # 忽略太小的噪声
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 编码为 JPEG 格式
        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            rospy.logwarn("图像编码失败")
            return

        # 创建 CompressedImage 消息
        compressed_msg = CompressedImage()
        compressed_msg.header.stamp = rospy.Time.now()
        compressed_msg.format = "jpeg"
        compressed_msg.data = encoded_image.tobytes()

        # 发布图像
        self.image_pub.publish(compressed_msg)

if __name__ == '__main__':
    try:
        YellowDetector()
    except rospy.ROSInterruptException:
        pass
