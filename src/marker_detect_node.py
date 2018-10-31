#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def callback(data):
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    #cv_image = markerRecognition(cv_image)
    cv_image = featureMatching(cv_image)
    image_pub.publish(bridge.cv2_to_imgmsg(cv_image, "bgr8"))

def featureMatching(original_image):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

    kp1, des1 = detector.detectAndCompute(gray_image, None)
    kp2, des2 = detector.detectAndCompute(temp_image, None)

    if des1 is None or des2 is None:
        return original_image

    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    match_param = 0.6
    for m,n in matches:
        if m.distance < match_param * n.distance:
            good.append([m])

    after_image = cv2.drawMatchesKnn(original_image,kp1,temp_image,kp2,good, None,flags=2)
    return after_image

def markerRecognition(original_image):
    after_image = original_image

    height, width, channels = after_image.shape
    image_size = height * width

    #グレースケール変換
    gray_image = cv2.cvtColor(after_image, cv2.COLOR_RGB2GRAY)

    retval, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_TOZERO_INV )
    binary_image = cv2.bitwise_not(binary_image)
    retval, binary_image= cv2.threshold(binary_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("binary",binary_image)
    
    countours_image, contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow("countour", countours_image)

    cv2.waitKey(1)

    for contour in contours:
        area = cv2.contourArea(contour)

        #過度に大小なエリアは弾く
        if area > image_size * 0.4 and area < image_size * 0.01:
            continue

        #輪郭を直線近似
        arclen = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.005 * arclen, True)

        #直線近似した輪郭を辺の数で場合分け
        if(len(approx) >= 3 and len(approx) <= 5):
            x,y,w,h = cv2.boundingRect(approx)
            #cv2.rectangle(after_image, (x,y), (x+w,y+h), (0,0,255), 3)
            cv2.circle(after_image, (x+w/2, y+h/2), 5, (0,0,255), -1)
            #cv2.drawContours(after_image, approx, -1, (0,0,255), 3)

    return after_image
    
if __name__ == '__main__':
    rospy.init_node('marker_detect_node', anonymous=True)
    image_pub = rospy.Publisher("marker_detect/image_raw", Image, queue_size=1)
    rospy.Subscriber("usb_cam/image_raw", Image, callback)
    bridge = CvBridge()
    temp_image = cv2.imread(os.environ["HOME"]+"/catkin_ws/src/drone_marker_pkg/resource/square.png",0) #第2引数が0でグレースケールで読み込むという意味
    detector = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    rospy.spin()
