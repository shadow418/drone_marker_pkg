#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

if __name__ == '__main__':
    detector = cv2.AKAZE_create()

    """
    cap = cv2.VideoCapture("../resource/video/mono_full_cut.mp4")
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        kp1 = detector.detect(gray_image)
        result = cv2.drawKeypoints(frame, kp1, None)
        cv2.imshow("test",result)
        cv2.waitKey(1)
    """
    
    temp_image = cv2.imread("../resource/20181211/C2/temp_mono2.jpg")
    gray_image = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)
    kp1 = detector.detect(gray_image)
    result = cv2.drawKeypoints(temp_image, kp1, None)
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.imshow("test", result)
    cv2.waitKey(0)