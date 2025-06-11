#!/usr/bin/env python
import rospy
import cv2
import numpy as np
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

bridge = CvBridge()
H_SAVE_PATH = "H_right_to_left.npy"
IMG_DIR = "calibration_images"
os.makedirs(IMG_DIR, exist_ok=True)

pattern_size = (6, 5)
flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

def callback(img_left_msg, img_right_msg):
    img_left = bridge.imgmsg_to_cv2(img_left_msg, "bgr8")
    img_right = bridge.imgmsg_to_cv2(img_right_msg, "bgr8")

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    ret_l, corners_l = cv2.findChessboardCorners(gray_left, pattern_size, flags)
    ret_r, corners_r = cv2.findChessboardCorners(gray_right, pattern_size, flags)

    debug_left = img_left.copy()
    debug_right = img_right.copy()

    if ret_l:
        cv2.drawChessboardCorners(debug_left, pattern_size, corners_l, ret_l)
    if ret_r:
        cv2.drawChessboardCorners(debug_right, pattern_size, corners_r, ret_r)

    if ret_l and ret_r:
        left_path = os.path.join(IMG_DIR, "left.png")
        right_path = os.path.join(IMG_DIR, "right.png")
        cv2.imwrite(left_path, img_left)
        cv2.imwrite(right_path, img_right)
        rospy.loginfo("成功保存图像对，开始标定")

        H, _ = cv2.findHomography(corners_r, corners_l)
        np.save(H_SAVE_PATH, H)
        rospy.loginfo("标定成功，H 保存为 %s", H_SAVE_PATH)

        rospy.signal_shutdown("标定完成，退出程序")

def main():
    rospy.init_node('auto_calibrate_node')

    left_sub = Subscriber("/hk_camera_left/image_raw", Image)
    right_sub = Subscriber("/hk_camera_right/image_raw", Image)

    sync = ApproximateTimeSynchronizer([left_sub, right_sub], queue_size=10, slop=0.1)
    sync.registerCallback(callback)

    rospy.loginfo("等待左右相机图像并开始自动标定...")
    rospy.spin()

if __name__ == "__main__":
    main()
