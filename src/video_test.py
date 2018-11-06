#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def bf_match(original_image):
    after_image = original_image

    gray_image = cv2.cvtColor(after_image, cv2.COLOR_RGB2GRAY)
    kp1, des1 = detector.detectAndCompute(gray_image, None)
    kp2, des2 = detector.detectAndCompute(temp_image, None)

    if des1 is None or des2 is None:
        return after_image

    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    
    points = []
    for match in matches[:10]:
        if match.trainIdx < len(kp1):
            points.append(kp1[match.trainIdx].pt)

    for point in points:
        after_image = cv2.circle(after_image, (int(point[0]), int(point[1])), 5, (0,0,255), -1)

    upper_left, lower_right = find_end_point(points)
    if upper_left is not None and lower_right is not None:
        after_image = cv2.rectangle(after_image, (int(upper_left[0]),int(upper_left[1])), (int(lower_right[0]),int(lower_right[1])), (255,0,0), 5)
        after_image = cv2.circle(after_image, (int(upper_left[0]), int(upper_left[1])), 10, (255,0,0), -1)
        after_image = cv2.circle(after_image, (int(lower_right[0]), int(lower_right[1])), 10, (255,0,0), -1)
    after_image = cv2.drawMatches(original_image, kp1, temp_image, kp2, matches[:10], None, flags=2)
    return after_image

#マッチングした特徴点から左上と右下の点を見つける
def find_end_point(points):
    upper_left = None
    lower_right = None
    for point in points:
        if upper_left is None:
            upper_left = point
        else:
            if point[0] < upper_left[0] and point[1] < upper_left[1]:
                upper_left = point

        if lower_right is None:
            lower_right = point
        else:
            if point[0] > lower_right[0] and point[1] > lower_right[1]:
                lower_right = point
    return upper_left,lower_right

if __name__ == '__main__':
    rospy.init_node('marker_detect_node', anonymous=True)
    image_pub = rospy.Publisher("marker_detect/image_raw", Image, queue_size=1)
    bridge = CvBridge()

    cap = cv2.VideoCapture(os.environ["HOME"]+"/catkin_ws/src/drone_marker_pkg/resource/1.mp4")
    temp_image = cv2.imread(os.environ["HOME"]+"/catkin_ws/src/drone_marker_pkg/resource/temp1_50.jpg",0) #第2引数が0でグレースケールで読み込むという意味
    detector = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    while(cap.isOpened()):
        ret, frame = cap.read()
        after_image = bf_match(frame)
        image_pub.publish(bridge.cv2_to_imgmsg(after_image, "bgr8"))