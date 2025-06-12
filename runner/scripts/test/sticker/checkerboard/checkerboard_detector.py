#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

class LinearChessboardDetectorNode:
    def __init__(self):
        rospy.init_node('linear_chessboard_detector_node', anonymous=True)
        self.bridge = CvBridge()

        self.camera_topic = rospy.get_param('~camera_topic', "/hk_camera/image_raw/compressed")
        self.max_corners = rospy.get_param('~max_corners', 20)
        self.pattern_orientation = rospy.get_param('~pattern_orientation', 'horizontal')  # or 'vertical'

        rospy.loginfo(f"订阅图像: {self.camera_topic}，最大角点数: {self.max_corners}")
        self.image_sub = rospy.Subscriber(self.camera_topic, CompressedImage, self.image_callback, queue_size=1)
        self.image_pub = rospy.Publisher("/chessboard_line_detected/image", Image, queue_size=1)

        rospy.spin()

    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 角点检测（不依赖棋盘格结构）
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_corners,
                qualityLevel=0.01,
                minDistance=10
            )

            if corners is not None and len(corners) >= 2:
                corners = np.int0(corners)
                points = [tuple(pt[0]) for pt in corners]

                # 根据方向排序
                if self.pattern_orientation == 'horizontal':
                    points.sort(key=lambda pt: pt[0])  # 按 X 排序
                else:
                    points.sort(key=lambda pt: pt[1])  # 按 Y 排序

                # 连线并画角点
                for i in range(len(points) - 1):
                    cv2.line(image, points[i], points[i + 1], (0, 255, 0), 2)

                for pt in points:
                    cv2.circle(image, pt, 4, (0, 0, 255), -1)

                rospy.loginfo_throttle(1, f"✅ 检测到角点并连线，共 {len(points)} 点")
                cv2.putText(image, f"Corners: {len(points)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(image, f"No sufficient corners", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))

        except Exception as e:
            rospy.logerr(f"图像处理失败: {e}")

if __name__ == "__main__":
    try:
        LinearChessboardDetectorNode()
    except rospy.ROSInterruptException:
        pass
