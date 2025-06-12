#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import apriltag
import json
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PolygonStamped, Point32


class AprilTagDetectorNode:
    def __init__(self):
        rospy.init_node('apriltag_detector_node', anonymous=True)

        # —— 参数 ——
        self.camera_topic = rospy.get_param('~camera_topic', "/stitched_image")
        self.tag_family    = rospy.get_param('~tag_family',    "tag16h5")
        self.force_invert  = rospy.get_param('~invert_image',  False)
        self.save_path     = rospy.get_param('~polygon_save_path', "apriltag_polygon.json")

        rospy.loginfo(f"Subscribing to: {self.camera_topic}, tag family: {self.tag_family}")
        rospy.loginfo(f"Polygon JSON → {self.save_path}")

        # —— 初始化工具 ——
        self.bridge   = CvBridge()
        try:
            options = apriltag.DetectorOptions(
                families   = self.tag_family,
                border     = 1,
                nthreads   = 4,
                quad_decimate = 1.5,
                refine_edges  = True
            )
        except TypeError:
            rospy.logwarn("DetectorOptions 参数不兼容，退回默认")
            options = apriltag.DetectorOptions(families=self.tag_family)
        self.detector = apriltag.Detector(options)

        # —— ROS 话题 ——
        self.image_sub     = rospy.Subscriber(self.camera_topic, Image, self.image_callback, queue_size=1)
        self.preprocess_pub= rospy.Publisher("/apriltag_preprocessing/image", Image, queue_size=1)
        self.image_pub     = rospy.Publisher("/apriltag_detected/image",     Image, queue_size=1)
        self.poly_pub      = rospy.Publisher("/apriltag_polygon", PolygonStamped, queue_size=1)

        self.total_frames = 0
        self.successful_detections = 0

        rospy.loginfo("✅ AprilTag detector node ready.")
        rospy.spin()


    def image_callback(self, msg: Image):
        self.total_frames += 1
        try:
            # 原始图转 OpenCV
            image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            image_orig = image_bgr.copy()

            # ———— 1. 预处理 ————
            gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced= clahe.apply(gray)
            blurred = cv2.GaussianBlur(enhanced, (3,3), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 根据亮度自动反转
            mean_int = float(np.mean(gray))
            if self.force_invert or mean_int > 240:
                binary = cv2.bitwise_not(binary)
                rospy.loginfo_throttle(5, f"Inverting image (mean={mean_int:.1f})")

            # 发布预处理结果
            prep_msg = self.bridge.cv2_to_imgmsg(binary, encoding="mono8")
            self.preprocess_pub.publish(prep_msg)

            # ———— 2. AprilTag 检测 ————
            detections = self.detector.detect(binary)
            if detections:
                self.successful_detections += 1
                rospy.loginfo(f"Detected {len(detections)} tag(s)")

                # 提取所有中心点
                centers = np.array([det.center for det in detections], dtype=np.float32)
                centroid= centers.mean(axis=0)

                # 极角排序，做闭合多边形
                angles   = np.arctan2(centers[:,1] - centroid[1],
                                      centers[:,0] - centroid[0])
                sort_idx = np.argsort(angles)
                sorted_centers = centers[sort_idx]

                # 在图上绘制多边形（蓝色）
                pts = np.int32(sorted_centers).reshape(-1,1,2)
                cv2.polylines(image_orig, [pts], isClosed=True, color=(255,0,0), thickness=2)

                # 保存到 JSON 文件
                poly_dict = {
                    "stamp":  msg.header.stamp.to_sec(),
                    "points": sorted_centers.tolist()
                }
                try:
                    with open(self.save_path, "w") as f:
                        json.dump(poly_dict, f, indent=2)
                    rospy.loginfo_throttle(5, f"Saved polygon → {self.save_path}")
                except Exception as e:
                    rospy.logwarn(f"Failed to save polygon: {e}")

            # ———— 3. 绘制 Tag 角点与 ID ————
            for det in detections:
                corners = np.int32(det.corners)
                for i in range(4):
                    cv2.line(image_orig,
                             tuple(corners[i]), tuple(corners[(i+1)%4]),
                             (0,255,0), 2)
                c = tuple(np.int32(det.center))
                cv2.putText(image_orig, f"ID:{det.tag_id}", c,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # ———— 4. 显示检测率 ————
            rate = (self.successful_detections/self.total_frames*100) if self.total_frames else 0
            cv2.putText(image_orig,
                        f"Tags:{len(detections)} Rate:{rate:.1f}%",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # 发布最终可视化图
            out_msg = self.bridge.cv2_to_imgmsg(image_orig, encoding="bgr8")
            self.image_pub.publish(out_msg)

        except Exception as e:
            rospy.logerr(f"⚠️ Detection error: {e}")


if __name__ == "__main__":
    try:
        AprilTagDetectorNode()
    except rospy.ROSInterruptException:
        pass
