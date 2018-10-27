#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def callback(data):
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    cv_image = markerRecognition(cv_image)
    image_pub.publish(bridge.cv2_to_imgmsg(cv_image, "bgr8"))

def markerRecognition(original_image):
    after_image = original_image

    height, width, channels = after_image.shape
    image_size = height * width

    gray_image = cv2.cvtColor(after_image, cv2.COLOR_RGB2GRAY)

    retval, dst = cv2.threshold(gray_image, 127, 255, cv2.THRESH_TOZERO_INV )
    dst = cv2.bitwise_not(dst)
    retval, dst = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    dst, contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 500:
            continue
            
        if image_size * 0.99 < area:
            continue
            
        x,y,w,h = cv2.boundingRect(contour)
        dst = cv2.rectangle(after_image,(x,y),(x+w,y+h),(0,255,0),2)

    return after_image
    
if __name__ == '__main__':
    rospy.init_node('pattern_matching_node', anonymous=True)
    image_pub = rospy.Publisher("recognition/image_raw", Image, queue_size=1)
    rospy.Subscriber("usb_cam/image_raw", Image, callback)
    bridge = CvBridge()
    rospy.spin()
