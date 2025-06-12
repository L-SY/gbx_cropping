#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import json
import os

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class JsonPolygonCropNode:
    def __init__(self):
        rospy.init_node("json_polygon_crop_node")

        # —— 参数 ——
        self.image_topic    = rospy.get_param("~image_topic",       "/stitched_image")
        self.json_path      = rospy.get_param("~polygon_json_path", "apriltag_polygon.json")
        self.output_topic   = rospy.get_param("~output_topic",     "/cropped_from_json/image")

        # ROS 工具
        self.bridge   = CvBridge()
        self.pub      = rospy.Publisher(self.output_topic, Image, queue_size=1)

        # 缓存 JSON 文件修改时间及顶点
        self._last_mtime = None
        self._points     = None

        # 订阅图像
        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)

        rospy.loginfo(f"[json_crop] Subscribed to '{self.image_topic}'")
        rospy.loginfo(f"[json_crop] JSON path: {self.json_path}")
        rospy.loginfo(f"[json_crop] Publishing to '{self.output_topic}'")
        rospy.spin()

    def _load_json(self):
        """按需从 JSON 加载多边形顶点缓存"""
        try:
            mtime = os.path.getmtime(self.json_path)
        except OSError:
            rospy.logwarn_throttle(10, f"[json_crop] JSON 文件不存在：{self.json_path}")
            return False

        if self._last_mtime == mtime and self._points is not None:
            return True

        try:
            with open(self.json_path, "r") as f:
                data = json.load(f)
            pts = data.get("points", [])
            if not isinstance(pts, list) or len(pts) < 3:
                raise ValueError("points 列表格式错误或点数不足")
            # 转成 OpenCV 多边形顶点形式
            self._points = np.array(pts, dtype=np.int32).reshape(-1,1,2)
            self._last_mtime = mtime
            rospy.loginfo_throttle(5, f"[json_crop] Loaded {len(self._points)} points")
            return True
        except Exception as e:
            rospy.logwarn_throttle(10, f"[json_crop] 读取 JSON 失败: {e}")
            return False

    def image_callback(self, img_msg: Image):
        # 1) 加载或更新顶点
        if not self._load_json():
            return

        # 2) 解码图像
        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"[json_crop] 解码图像失败: {e}")
            return

        h, w = img.shape[:2]

        # 3) 用多边形顶点生成掩码并裁切
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [self._points], 255)
        cropped = cv2.bitwise_and(img, img, mask=mask)

        # 4) 计算最小外接矩形，去掉黑边
        x, y, w_box, h_box = cv2.boundingRect(self._points)
        if w_box > 0 and h_box > 0:
            cropped = cropped[y:y+h_box, x:x+w_box]
        else:
            rospy.logwarn_throttle(5, "[json_crop] 无效的裁切区域，跳过剪裁黑边")

        # 5) 发布裁切后的小图
        out_msg = self.bridge.cv2_to_imgmsg(cropped, encoding="bgr8")
        out_msg.header = img_msg.header
        self.pub.publish(out_msg)


if __name__ == "__main__":
    try:
        JsonPolygonCropNode()
    except rospy.ROSInterruptException:
        pass
