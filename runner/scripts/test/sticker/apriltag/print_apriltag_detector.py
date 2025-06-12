#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import apriltag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def balance_white(img):
    """简易灰度世界白平衡"""
    img = img.astype(np.float32)
    mb, mg, mr = cv2.mean(img)[:3]
    mg_gray = (mb + mg + mr) / 3.0
    sb = mg_gray / (mb + 1e-6)
    sg = mg_gray / (mg + 1e-6)
    sr = mg_gray / (mr + 1e-6)
    out = cv2.merge([
        np.clip(img[:,:,0] * sb, 0,255),
        np.clip(img[:,:,1] * sg, 0,255),
        np.clip(img[:,:,2] * sr, 0,255)
    ]).astype(np.uint8)
    return out

class MultiPassAprilTagDetector:
    def __init__(self):
        rospy.init_node('multi_pass_apriltag', anonymous=True)
        self.bridge = CvBridge()

        # 参数
        topic   = rospy.get_param('~camera_topic', '/stitched_image')
        family  = rospy.get_param('~tag_family', 'tag16h5')
        self.inv= rospy.get_param('~invert_image', False)

        # DetectorOptions 兼容写法
        try:
            opts = apriltag.DetectorOptions(
                families      = family,
                border        = 1,
                nthreads      = 4,
                quad_decimate = 1.0,
                quad_sigma    = 0.8,
                refine_edges  = True,
                refine_decode = True,
                refine_pose   = False
            )
        except TypeError:
            opts = apriltag.DetectorOptions(families=family, border=1, nthreads=4)
            opts.quad_decimate = 1.0
            opts.refine_edges  = True
            opts.refine_decode = True
            if hasattr(opts, 'quad_sigma'):
                opts.quad_sigma = 0.8
            if hasattr(opts, 'refine_pose'):
                opts.refine_pose = False

        self.detector = apriltag.Detector(opts)

        # ROS
        self.sub = rospy.Subscriber(topic, Image, self.callback, queue_size=1)
        self.pub_pre = rospy.Publisher('/apriltag_multi_pre', Image, queue_size=1)
        self.pub_out = rospy.Publisher('/apriltag_multi_out', Image, queue_size=1)
        rospy.loginfo(f"[AprilTag] topic={topic}, family={family}, invert={self.inv}")
        rospy.spin()

    def preprocess(self, img):
        """白平衡 + LAB CLAHE + 双边滤波"""
        wb   = balance_white(img)
        lab  = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
        l,a,b= cv2.split(lab)
        clahe= cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l2   = clahe.apply(l)
        lab2 = cv2.merge((l2,a,b))
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    def get_binaries(self, gray):
        """生成多种二值化图并去重"""
        bins = []

        # 1) 自适应阈值
        th_adapt = cv2.adaptiveThreshold(
            gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV if not self.inv else cv2.THRESH_BINARY,
            blockSize=51, C=10)
        bins.append(th_adapt)

        # 2) 固定多阈值
        for t in (30, 60, 90, 120, 150, 180):
            _, bw = cv2.threshold(
                gray, t, 255,
                cv2.THRESH_BINARY_INV if not self.inv else cv2.THRESH_BINARY)
            bins.append(bw)

        # 3) 闭运算 & 开运算 变体
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        more = []
        for b in bins:
            more.append(cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel))
            more.append(cv2.morphologyEx(b, cv2.MORPH_OPEN,  kernel))
        bins.extend(more)

        # 去重：用 Python hash 对 bytes 去重
        uniq = []
        seen = set()
        for b in bins:
            h = hash(b.tobytes())
            if h not in seen:
                seen.add(h)
                uniq.append(b)
        return uniq

    def callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        proc = self.preprocess(img)
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

        binaries = self.get_binaries(gray)
        # 发布第一张二值图，观察二值化效果
        self.pub_pre.publish(self.bridge.cv2_to_imgmsg(binaries[0], 'mono8'))

        # 多次检测，合并所有 detections
        all_dets = []
        for b in binaries:
            dets = self.detector.detect(b)
            if not dets:
                # fallback: 试灰度图
                dets = self.detector.detect(gray)
            all_dets += dets

        # 去重同一个 tag（按 id + 近似中心格子）
        uniq = {}
        for d in all_dets:
            key = (d.tag_id, int(d.center[0]//5), int(d.center[1]//5))
            if key not in uniq:
                uniq[key] = d
        final = list(uniq.values())

        # 绘图并发布
        out = img.copy()
        for d in final:
            c = np.int32(d.corners)
            cv2.polylines(out, [c], True, (0,255,0), 2)
            ctr = tuple(np.int32(d.center))
            cv2.putText(out, f"ID:{d.tag_id}", ctr,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)

        self.pub_out.publish(self.bridge.cv2_to_imgmsg(out, 'bgr8'))

if __name__ == '__main__':
    try:
        MultiPassAprilTagDetector()
    except rospy.ROSInterruptException:
        pass
