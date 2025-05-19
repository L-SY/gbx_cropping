#!/usr/bin/env python
import rospy
from nodelet import Nodelet
from ros_high_undistort_extract_pp import PanelDetector  # 原始类保持不变

class PanelDetectorNodelet(Nodelet):
    def onInit(self):
        rospy.loginfo("✅ PanelDetectorNodelet initialized.")
        self.detector = PanelDetector()

