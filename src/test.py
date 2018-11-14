#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rospy
import cv2
import numpy as np

if __name__ == '__main__':
    temp_image = cv2.imread("../resource/temp_color50.jpg")
    temp_gray_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)

    temp_image2 = cv2.imread("../resource/temp1_50.jpg")
    temp_gray_image2 = cv2.cvtColor(temp_image2, cv2.COLOR_RGB2GRAY)

    
    detector = cv2.AKAZE_create()
    kp1 = detector.detect(temp_gray_image)
    kp2 = detector.detect(temp_gray_image2)

    result = cv2.drawKeypoints(temp_gray_image, kp1, None)
    result2 = cv2.drawKeypoints(temp_gray_image2, kp2, None)
    while(1):
        cv2.imshow("test",result)
        cv2.imshow("test2",result2)
        cv2.waitKey(1)