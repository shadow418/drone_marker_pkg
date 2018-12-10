#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError

if __name__ == '__main__':
    rospy.init_node('video_to_ros', anonymous=True)
    image_pub = rospy.Publisher("video/image_raw", Image, queue_size=1)
    bridge = CvBridge()

    cap = cv2.VideoCapture(os.environ["HOME"]+"/catkin_ws/src/drone_marker_pkg/resource/video/mono1.mp4")
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        image_pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))
        cv2.waitKey(1)