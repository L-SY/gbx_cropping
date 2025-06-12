#!/usr/bin/env python
import rospy
import cv2
import numpy as np
import os
import sys
import select
import termios
import tty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

# 全局设置
bridge = CvBridge()
PATTERN_SIZE = (6, 5)
# Flags 组合：
#   CALIB_CB_ADAPTIVE_THRESH  - 自适应阈值处理，增强局部对比
#   CALIB_CB_NORMALIZE_IMAGE  - 归一化灰度，减小光照不均影响
#   CALIB_CB_FAST_CHECK       - 先做快速二值检测，提高速度
# 注意：这些选项可以加速角点检测，但在光照极端或棋盘边缘模糊时可能导致漏检或误检，可根据实际情况增删
FLAGS = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
)

# 保存目录
SAVE_DIR = "double_camera_calibration"
os.makedirs(SAVE_DIR, exist_ok=True)

# 发布器
pub_left_corners = None
pub_right_corners = None
pub_debug = None

# 临时存储最新图像与角点
img_left = None
img_right = None
corners_left = None
corners_right = None
ret_l = False
ret_r = False

# 多张变换矩阵列表及平均矩阵
Hs = []
H_avg = None

# Terminal 设置，用于捕获键盘输入
old_term = termios.tcgetattr(sys.stdin)
def init_term():
    tty.setcbreak(sys.stdin.fileno())
def restore_term():
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)

# 边缘模糊预处理：CLAHE + 双边滤波
def preprocess(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)
    blurred = cv2.bilateralFilter(eq, d=9, sigmaColor=75, sigmaSpace=75)
    return blurred

# 角点可视化并发布
def publish_corners(img, corners, ret, pub):
    vis = img.copy()
    if ret:
        cv2.drawChessboardCorners(vis, PATTERN_SIZE, corners, ret)
    pub.publish(bridge.cv2_to_imgmsg(vis, "bgr8"))

# 图像拼接函数，返回 stitched 和 debug 图
def warp_and_stitch(img_l, img_r, H):
    h_l, w_l = img_l.shape[:2]
    h_r, w_r = img_r.shape[:2]
    corners_r = np.array([[0,0],[w_r,0],[w_r,h_r],[0,h_r]], dtype=np.float32).reshape(-1,1,2)
    corners_r_tr = cv2.perspectiveTransform(corners_r, H)
    corners_l = np.array([[0,0],[w_l,0],[w_l,h_l],[0,h_l]], dtype=np.float32).reshape(-1,1,2)
    all_c = np.concatenate((corners_l, corners_r_tr), axis=0)
    xmin, ymin = np.int32(all_c.min(axis=0).ravel() - 0.5)
    xmax, ymax = np.int32(all_c.max(axis=0).ravel() + 0.5)
    tr = [-xmin, -ymin]
    size = (xmax - xmin, ymax - ymin)
    H_tr = np.array([[1,0,tr[0]],[0,1,tr[1]],[0,0,1]]) @ H

    warped_r = cv2.warpPerspective(img_r, H_tr, size)
    stitched = warped_r.copy()
    stitched[tr[1]:tr[1]+h_l, tr[0]:tr[0]+w_l] = img_l

    debug = stitched.copy()
    # 左图绿框
    cv2.rectangle(debug, (tr[0], tr[1]), (tr[0]+w_l, tr[1]+h_l), (0,255,0), 2)
    # 右图红框
    pts_r = np.int32(cv2.perspectiveTransform(corners_r, H_tr))
    cv2.polylines(debug, [pts_r], True, (0,0,255), 2)
    return stitched, debug

# 同步回调：预处理后检测，发布角点和基于 H_avg 发布 debug 图
def callback(img_left_msg, img_right_msg):
    global img_left, img_right, corners_left, corners_right, ret_l, ret_r, H_avg
    img_left = bridge.imgmsg_to_cv2(img_left_msg, "bgr8")
    img_right = bridge.imgmsg_to_cv2(img_right_msg, "bgr8")

    gray_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    proc_l = preprocess(gray_l)
    proc_r = preprocess(gray_r)

    ret_l, corners_left = cv2.findChessboardCorners(proc_l, PATTERN_SIZE, FLAGS)
    ret_r, corners_right = cv2.findChessboardCorners(proc_r, PATTERN_SIZE, FLAGS)

    publish_corners(img_left, corners_left, ret_l, pub_left_corners)
    publish_corners(img_right, corners_right, ret_r, pub_right_corners)

    if H_avg is not None:
        _, debug = warp_and_stitch(img_left, img_right, H_avg)
        pub_debug.publish(bridge.cv2_to_imgmsg(debug, "bgr8"))

# 键盘处理：Terminal 读取 'c'/'q'
def process_key():
    global H_avg
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    if dr:
        c = sys.stdin.read(1)
        if c == 'c' and ret_l and ret_r:
            idx = len(Hs)
            cv2.imwrite(os.path.join(SAVE_DIR, f"left_{idx}.png"), img_left)
            cv2.imwrite(os.path.join(SAVE_DIR, f"right_{idx}.png"), img_right)
            rospy.loginfo(f"[SAVE] 图像对 {idx} 保存到 {SAVE_DIR}")

            H, _ = cv2.findHomography(corners_right, corners_left)
            Hs.append(H)
            H_avg = np.mean(np.stack(Hs), axis=0)
            rospy.loginfo(f"[CALC] 新单应加入，共 {len(Hs)} 张，已更新平均 H")

        elif c == 'q':
            if H_avg is not None:
                np.save(os.path.join(SAVE_DIR, "H_average.npy"), H_avg)
                rospy.loginfo(f"[DONE] 平均单应矩阵保存为 {SAVE_DIR}/H_right_to_left.npy")
            rospy.signal_shutdown("用户退出，标定完成")

# 主入口
if __name__ == '__main__':
    rospy.init_node('double_camera_calib')
    init_term()

    pub_left_corners  = rospy.Publisher('/left_corners',  Image, queue_size=1)
    pub_right_corners = rospy.Publisher('/right_corners', Image, queue_size=1)
    pub_debug         = rospy.Publisher('/debug_image',   Image, queue_size=1)

    left_sub  = Subscriber('/hk_camera_left/image_raw',  Image)
    right_sub = Subscriber('/hk_camera_right/image_raw', Image)
    sync = ApproximateTimeSynchronizer([left_sub, right_sub], queue_size=10, slop=0.1)
    sync.registerCallback(callback)

    rospy.loginfo("[INFO] 双相机标定启动: 终端键入 'c' 保存并更新 H, 'q' 退出并生成 npy")
    rate = rospy.Rate(30)
    try:
        while not rospy.is_shutdown():
            process_key()
            rate.sleep()
    finally:
        restore_term()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
