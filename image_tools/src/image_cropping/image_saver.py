#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

# 初始化全局变量
image_counter = 0  # 图片计数器
save_path = "./images"  # 保存图片的目录
start_number = 0  # 起始编号

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

# 保存图片函数
def save_image():
    global image_counter
    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 构造图片文件名
    filename = os.path.join(save_path, f"raw_{image_counter}.jpg")
    # 保存当前帧
    cv2.imwrite(filename, current_frame)
    rospy.loginfo(f"Saved image: {filename}")
    image_counter += 1

if __name__ == "__main__":
    rospy.init_node("image_saver", anonymous=True)

    # 从参数服务器读取起始编号
    start_number = rospy.get_param("~start_number", 72)
    image_counter = start_number
    save_path = rospy.get_param("~save_path", "/home/lsy/gbx_cropping_ws/src/image_tools/second_batch_of_samples/raw_images_90")

    # 创建CvBridge实例
    bridge = CvBridge()

    # 订阅图像话题
    rospy.Subscriber("/hk_camera/image_raw", Image, image_callback)

    # 打开窗口
    rospy.loginfo("Press 's' to save an image.")
    rospy.loginfo(f"Images will be saved to: {save_path}")
    rospy.loginfo(f"Starting from image_{start_number}.jpg")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        cv2.destroyAllWindows()