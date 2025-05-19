#!/usr/bin/env python3
import rospy
import cv2
import os
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import numpy as np

# 参数配置
save_dir = "/runner/blackpoint/images"
os.makedirs(save_dir, exist_ok=True)

bridge = CvBridge()
current_image = None

def image_callback(msg):
    global current_image
    try:
        # 解码 compressed image
        np_arr = np.frombuffer(msg.data, np.uint8)
        current_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv2.imshow("Live Feed", current_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('a') and current_image is not None:
            filename = os.path.join(save_dir, f"{rospy.Time.now().to_nsec()}.jpg")
            cv2.imwrite(filename, current_image)
            rospy.loginfo(f"Image saved: {filename}")

    except Exception as e:
        rospy.logerr(f"Error processing image: {e}")

def main():
    rospy.init_node("image_saver_and_stitcher", anonymous=True)
    rospy.Subscriber("/hk_camera/image_raw/compressed", CompressedImage, image_callback)
    rospy.loginfo("Press 'a' to save images, 'y' to stitch and show saved images.")
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
