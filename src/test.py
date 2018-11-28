#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

if __name__ == '__main__':
    detector = cv2.AKAZE_create()
    
    temp_image = cv2.imread("../resource/en.jpg")
    temp_gray_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)

    temp_image2 = cv2.imread("../resource/1.png")
    temp_gray_image2 = cv2.cvtColor(temp_image2, cv2.COLOR_RGB2GRAY)

    temp_image3 = cv2.imread("../resource/2.png")
    temp_gray_image3 = cv2.cvtColor(temp_image3, cv2.COLOR_RGB2GRAY)

    ret = cv2.matchShapes(temp_gray_image, temp_gray_image2, 1, 0)
    ret2 = cv2.matchShapes(temp_gray_image, temp_gray_image3, 1, 0)

    print ret
    print ret2

    while(1):
        cv2.imshow("1", temp_gray_image)
        cv2.imshow("2", temp_gray_image2)
        cv2.imshow("3", temp_gray_image3)
        cv2.waitKey(1)

    #kp1 = detector.detect(temp_gray_image)
    #kp2 = detector.detect(temp_gray_image2)

    #result = cv2.drawKeypoints(temp_gray_image, kp1, None)
    #result2 = cv2.drawKeypoints(temp_gray_image2, kp2, None)

    #while(1):
    #    cv2.imshow("test",result)
    #    cv2.imshow("test2",result2)
    #    cv2.waitKey(1)
    
    

    """    
    cap = cv2.VideoCapture("../resource/video/color2.mp4")
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        kp1 = detector.detect(gray_image)
        result = cv2.drawKeypoints(gray_image, kp1, None)
        cv2.imshow("test",result)
        cv2.waitKey(1)
    """
    