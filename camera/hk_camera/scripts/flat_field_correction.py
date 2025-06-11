#!/usr/bin/env python
import rospy
import cv2
import numpy as np
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

bridge = CvBridge()
FLAT_DIR = "flat_images"
os.makedirs(FLAT_DIR, exist_ok=True)

def callback(img_left_msg, img_right_msg):
    img_left = bridge.imgmsg_to_cv2(img_left_msg, "bgr8")
    img_right = bridge.imgmsg_to_cv2(img_right_msg, "bgr8")

    # 灰度图（适合光照分析）
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # 归一化为平均值 = 1
    flat_left = gray_left.astype(np.float32)
    flat_right = gray_right.astype(np.float32)
    flat_left /= np.mean(flat_left)
    flat_right /= np.mean(flat_right)

    np.save(os.path.join(FLAT_DIR, "flat_left.npy"), flat_left)
    np.save(os.path.join(FLAT_DIR, "flat_right.npy"), flat_right)

    rospy.loginfo("平场图已保存：flat_left.npy / flat_right.npy")
    rospy.signal_shutdown("完成平场图采集，退出")

def main():
    rospy.init_node('flatfield_capture_node')

    left_sub = Subscriber("/hk_camera_left/image_raw", Image)
    right_sub = Subscriber("/hk_camera_right/image_raw", Image)

    sync = ApproximateTimeSynchronizer([left_sub, right_sub], queue_size=10, slop=0.1)
    sync.registerCallback(callback)

    rospy.loginfo("等待图像采集平场图，请对准均匀白光源...")
    rospy.spin()

if __name__ == "__main__":
    main()
