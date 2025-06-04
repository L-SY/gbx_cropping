#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()
H = np.load("H_right_to_left.npy")  # Homography from right to left

def callback_left(msg):
    global img_left
    img_left = bridge.imgmsg_to_cv2(msg, "bgr8")

def callback_right(msg):
    global img_right
    img_right = bridge.imgmsg_to_cv2(msg, "bgr8")

def warp_and_stitch(img_left, img_right, H):
    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]

    # 将右图的四个角变换到左图坐标系
    corners_right = np.array([[0, 0], [w_right, 0], [w_right, h_right], [0, h_right]], dtype=np.float32).reshape(-1, 1, 2)
    corners_right_trans = cv2.perspectiveTransform(corners_right, H)

    # 获取变换后左右图总范围
    corners_left = np.array([[0, 0], [w_left, 0], [w_left, h_left], [0, h_left]], dtype=np.float32).reshape(-1, 1, 2)
    all_corners = np.concatenate((corners_left, corners_right_trans), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    translation = [-xmin, -ymin]

    # 计算新图像大小 + 平移矩阵
    size = (xmax - xmin, ymax - ymin)
    H_trans = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]]) @ H

    # warp 右图 + paste 左图
    warped_right = cv2.warpPerspective(img_right, H_trans, size)
    stitched = warped_right.copy()
    stitched[translation[1]:translation[1]+h_left, translation[0]:translation[0]+w_left] = img_left

    return stitched, translation, H_trans

def publisher():
    pub_stitched = rospy.Publisher("/stitched_image", Image, queue_size=1)
    pub_debug = rospy.Publisher("/debug_image", Image, queue_size=1)

    rospy.init_node('image_stitcher', anonymous=True)
    rospy.Subscriber("/hk_camera_left/image_raw", Image, callback_left)
    rospy.Subscriber("/hk_camera_right/image_raw", Image, callback_right)

    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        if img_left is None or img_right is None:
            rate.sleep()
            continue

        stitched, translation, H_trans = warp_and_stitch(img_left, img_right, H)

        # 画 debug 框
        h_left, w_left = img_left.shape[:2]
        h_right, w_right = img_right.shape[:2]

        debug = stitched.copy()
        cv2.rectangle(debug,
                      (translation[0], translation[1]),
                      (translation[0] + w_left, translation[1] + h_left),
                      (0, 255, 0), 3)

        corners_right = np.array([[0, 0], [w_right, 0], [w_right, h_right], [0, h_right]], dtype=np.float32).reshape(-1, 1, 2)
        corners_right_trans = cv2.perspectiveTransform(corners_right, H_trans).astype(np.int32)
        cv2.polylines(debug, [corners_right_trans], isClosed=True, color=(0, 0, 255), thickness=3)


        pub_stitched.publish(bridge.cv2_to_imgmsg(stitched, "bgr8"))
        pub_debug.publish(bridge.cv2_to_imgmsg(debug, "bgr8"))

        rate.sleep()

if __name__ == "__main__":
    img_left, img_right = None, None
    publisher()
