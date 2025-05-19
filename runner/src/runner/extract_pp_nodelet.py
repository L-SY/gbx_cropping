#!/usr/bin/env python
import rospy
from nodelet import Nodelet
from runner.ros_high_undistort_extract_pp import PanelDetector

class ExtractPPNodelet(Nodelet):
    def onInit(self):
        rospy.loginfo("✅ ExtractPPNodelet initialized.")
        self.detector = PanelDetector()

