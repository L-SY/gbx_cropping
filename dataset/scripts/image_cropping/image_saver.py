#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import argparse

# 初始化全局变量
front_counter = 1  # front文件夹的图片计数器
back_counter = 1   # back文件夹的图片计数器
save_path = "./images"  # 保存图片的基础目录
current_frame = None  # 当前帧图像
last_saved_file = None  # 最后保存的文件路径
last_saved_type = None  # 最后保存的文件类型 ('front' 或 'back')

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
    if key == ord('f'):  # 如果按下'f'键，保存到front文件夹
        save_image("front")
    elif key == ord('b'):  # 如果按下'b'键，保存到back文件夹
        save_image("back")
    elif key == ord('z'):  # 如果按下'z'键，撤销上一次保存操作
        undo_last_save()
    elif key == ord('q'):  # 如果按下'q'键
        rospy.signal_shutdown("User requested shutdown")

# 保存图片函数
def save_image(folder):
    global front_counter, back_counter, last_saved_file, last_saved_type

    # 根据文件夹选择对应的计数器
    if folder == "front":
        counter = front_counter
        front_counter += 1
        prefix = "front"
    else:  # folder == "back"
        counter = back_counter
        back_counter += 1
        prefix = "back"

    # 构建保存路径
    target_dir = os.path.join(save_path, folder)

    # 确保保存路径存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 构造图片文件名，使用前缀_编号的格式
    filename = os.path.join(target_dir, f"{prefix}_{counter}.jpg")

    # 保存当前帧
    if current_frame is not None:
        cv2.imwrite(filename, current_frame)
        rospy.loginfo(f"Saved image to {folder}: {filename} (Counter: {counter})")

        # 记录本次保存的文件信息，用于撤销操作
        last_saved_file = filename
        last_saved_type = folder
    else:
        rospy.logwarn("No image to save")

# 撤销上一次保存操作
def undo_last_save():
    global front_counter, back_counter, last_saved_file, last_saved_type

    if last_saved_file and os.path.exists(last_saved_file):
        try:
            # 删除文件
            os.remove(last_saved_file)

            # 根据文件类型减少对应的计数器
            if last_saved_type == "front":
                front_counter -= 1
                rospy.loginfo(f"Undid last front save: {last_saved_file}. Next front counter: {front_counter}")
            elif last_saved_type == "back":
                back_counter -= 1
                rospy.loginfo(f"Undid last back save: {last_saved_file}. Next back counter: {back_counter}")

            # 重置最后保存的文件信息
            last_saved_file = None
            last_saved_type = None
        except Exception as e:
            rospy.logerr(f"Error undoing last save: {e}")
    else:
        rospy.logwarn("No previous save to undo or file no longer exists")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Save images from a camera topic')
    parser.add_argument('--front_start', type=int, default=1, help='Starting image number for front folder')
    parser.add_argument('--back_start', type=int, default=1, help='Starting image number for back folder')
    parser.add_argument('--path', type=str, default="./images", help='Base path to save images')
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
    front_counter = args.front_start
    back_counter = args.back_start
    save_path = args.path
    image_topic = args.topic

    # 从ROS参数服务器读取参数，如果命令行未提供则使用
    front_counter = rospy.get_param("~front_start", front_counter)
    back_counter = rospy.get_param("~back_start", back_counter)
    save_path = rospy.get_param("~save_path", save_path)
    image_topic = rospy.get_param("~image_topic", image_topic)

    # 创建CvBridge实例
    bridge = CvBridge()

    # 订阅图像话题
    rospy.Subscriber(image_topic, Image, image_callback)

    # 打开窗口并显示操作说明
    rospy.loginfo("Press 'f' to save an image to front folder (as front_N.jpg)")
    rospy.loginfo("Press 'b' to save an image to back folder (as back_N.jpg)")
    rospy.loginfo("Press 'z' to undo the last save operation")
    rospy.loginfo("Press 'q' to quit.")
    rospy.loginfo(f"Images will be saved to: {save_path}/front or {save_path}/back")
    rospy.loginfo(f"Front folder starting from image number: {front_counter}")
    rospy.loginfo(f"Back folder starting from image number: {back_counter}")
    rospy.loginfo(f"Subscribing to topic: {image_topic}")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        cv2.destroyAllWindows()