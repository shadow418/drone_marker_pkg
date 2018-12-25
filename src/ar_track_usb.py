#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def callback(data):
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    cv_image = ar_track(cv_image)
    image_pub.publish(bridge.cv2_to_imgmsg(cv_image, "bgr8"))

def ar_track(image):
    image_size = image.shape[0] * image.shape[1]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary_image= cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

    for contour in contours:
        area = cv2.contourArea(contour)
        #過度に大小なエリアは弾く
        if area > image_size * 0.8 or area < image_size * 0.01:
            continue

        #輪郭を直線近似
        arclen = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.005 * arclen, True)

        #直線近似した輪郭を辺の数で場合分け
        if len(approx) == 4:
            cv2.line(image, (approx[0][0][0],approx[0][0][1]), (approx[1][0][0],approx[1][0][1]), (255,0,0), 5)
            cv2.line(image, (approx[1][0][0],approx[1][0][1]), (approx[2][0][0],approx[2][0][1]), (255,0,0), 5)
            cv2.line(image, (approx[2][0][0],approx[2][0][1]), (approx[3][0][0],approx[3][0][1]), (255,0,0), 5)
            cv2.line(image, (approx[3][0][0],approx[3][0][1]), (approx[0][0][0],approx[0][0][1]), (255,0,0), 5)
            cv2.drawContours(image, approx, -1, (0,0,255), 10)
            
    return image

if __name__ == '__main__':
    rospy.init_node('ar_track_usb', anonymous=True)
    image_pub = rospy.Publisher("ar_track/image_raw", Image, queue_size=1)
    points_pub = rospy.Publisher("ar_track/points", Float32MultiArray, queue_size=1)   
    bridge = CvBridge()
    rospy.Subscriber("usb_cam/image_raw", Image, callback)
    rospy.spin()
