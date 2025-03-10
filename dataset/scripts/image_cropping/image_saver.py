#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import argparse

# 初始化全局变量
image_counter = 0  # 图片计数器
save_path = "./images"  # 保存图片的目录
current_frame = None  # 当前帧图像

# 回调函数：接收并处理图像
def image_callback(msg):
    global current_frame
    try:
        # 将ROS图像消息转换为OpenCV格式
        current_frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow("Camera Feed", current_frame)
    except Exception as e:
        rospy.logerr(f"Error converting image: {e}")

    # 按键监听
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # 如果按下's'键
        save_image()
    elif key == ord('q'):  # 如果按下'q'键
        rospy.signal_shutdown("User requested shutdown")

# 保存图片函数
def save_image():
    global image_counter
    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 构造图片文件名
    filename = os.path.join(save_path, f"raw_{image_counter}.jpg")
    # 保存当前帧
    if current_frame is not None:
        cv2.imwrite(filename, current_frame)
        rospy.loginfo(f"Saved image: {filename}")
        image_counter += 1
    else:
        rospy.logwarn("No image to save")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Save images from a camera topic')
    parser.add_argument('--start', type=int, default=1, help='Starting image number')
    parser.add_argument('--path', type=str, default="./images", help='Path to save images')
    parser.add_argument('--topic', type=str, default="/hk_camera/image_raw", help='Image topic to subscribe')

    # 在ROS环境中，命令行参数需要特殊处理
    # 提取ROS节点之后的参数
    args = rospy.myargv()[1:]
    return parser.parse_args(args)

if __name__ == "__main__":
    # 初始化ROS节点
    rospy.init_node("image_saver", anonymous=True)

    # 解析命令行参数
    args = parse_arguments()

    # 从参数中读取设置
    image_counter = args.start
    save_path = args.path
    image_topic = args.topic

    # 从ROS参数服务器读取参数，如果命令行未提供则使用
    image_counter = rospy.get_param("~start_number", image_counter)
    save_path = rospy.get_param("~save_path", save_path)
    image_topic = rospy.get_param("~image_topic", image_topic)

    # 创建CvBridge实例
    bridge = CvBridge()

    # 订阅图像话题
    rospy.Subscriber(image_topic, Image, image_callback)

    # 打开窗口并显示操作说明
    rospy.loginfo("Press 's' to save an image, 'q' to quit.")
    rospy.loginfo(f"Images will be saved to: {save_path}")
    rospy.loginfo(f"Starting from raw_{image_counter}.jpg")
    rospy.loginfo(f"Subscribing to topic: {image_topic}")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        cv2.destroyAllWindows()