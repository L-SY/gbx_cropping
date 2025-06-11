#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()
H = np.load("H_right_to_left.npy")  # Homography from right to left

img_left, img_right = None, None
use_flat_field = False
flat_left = None
flat_right = None

def correct_flat(image, flat_map, epsilon=1e-6):
    image_f32 = image.astype(np.float32)
    flat_safe = np.where(flat_map < epsilon, epsilon, flat_map)

    if len(image.shape) == 3:
        for c in range(3):
            image_f32[:, :, c] = image_f32[:, :, c] / flat_safe
    else:
        image_f32 = image_f32 / flat_safe

    return np.clip(image_f32, 0, 255).astype(np.uint8)

def callback_left(msg):
    global img_left
    img_left_raw = bridge.imgmsg_to_cv2(msg, "bgr8")
    if use_flat_field:
        img_left = correct_flat(img_left_raw, flat_left)
    else:
        img_left = img_left_raw

def callback_right(msg):
    global img_right
    img_right_raw = bridge.imgmsg_to_cv2(msg, "bgr8")
    if use_flat_field:
        img_right = correct_flat(img_right_raw, flat_right)
    else:
        img_right = img_right_raw

def warp_and_stitch(img_left, img_right, H):
    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]

    corners_right = np.array([[0, 0], [w_right, 0], [w_right, h_right], [0, h_right]], dtype=np.float32).reshape(-1, 1, 2)
    corners_right_trans = cv2.perspectiveTransform(corners_right, H)

    corners_left = np.array([[0, 0], [w_left, 0], [w_left, h_left], [0, h_left]], dtype=np.float32).reshape(-1, 1, 2)
    all_corners = np.concatenate((corners_left, corners_right_trans), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    translation = [-xmin, -ymin]

    size = (xmax - xmin, ymax - ymin)
    H_trans = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]]) @ H

    warped_right = cv2.warpPerspective(img_right, H_trans, size)
    stitched = warped_right.copy()
    stitched[translation[1]:translation[1]+h_left, translation[0]:translation[0]+w_left] = img_left

    return stitched, translation, H_trans

def publisher():
    global use_flat_field, flat_left, flat_right

    rospy.init_node('image_stitcher', anonymous=True)

    use_flat_field = rospy.get_param("~use_flat_field", True)
    if use_flat_field:
        rospy.loginfo("启用平场校正")
        flat_left = np.load("flat_images/flat_left.npy")
        flat_right = np.load("flat_images/flat_right.npy")
    else:
        rospy.logwarn("未启用平场校正")

    rospy.Subscriber("/hk_camera_left/image_rect", Image, callback_left)
    rospy.Subscriber("/hk_camera_right/image_rect", Image, callback_right)

    pub_stitched = rospy.Publisher("/stitched_image", Image, queue_size=1)
    pub_debug = rospy.Publisher("/debug_image", Image, queue_size=1)

    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        if img_left is None or img_right is None:
            rate.sleep()
            continue

        stitched, translation, H_trans = warp_and_stitch(img_left, img_right, H)

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
    publisher()
